"""
Fine-tunes MalwareGAT on the original training set PLUS real benign samples
from kaggle_dataset/data.csv, to close the real-world generalization gap
measured this session (GNN-only false-positive rate of 95.5% on independent
real benign software -- see data/kaggle_header_entropy_test/report.json).

Root-cause hypothesis being tested: the original model was NEVER trained on
a sample with real Header+Entropy but zero API+Network (the only shape a
live raw-binary upload can honestly produce, since the original API/Network
encoding isn't recoverable) -- every one of its 5,138 original training
samples had all 4 nodes populated. This adds real examples of exactly that
input shape, labeled Benign (real, verified labels from data.csv), so the
model can learn what it actually looks like instead of treating it as
unrecognized and defaulting to "malicious".

Only BENIGN augmentation is added to training -- data.csv's malicious label
has no malware family, and this is an 11-class model; inventing a family
label for them would be exactly the kind of guessing this project has been
avoiding. The measured problem is specifically a false-POSITIVE issue, so
benign-only augmentation is the targeted fix for the problem actually found.

3-way split of data.csv's real benign pool (41,323 samples), no overlap:
  - 20,000 -> added to TRAINING (mixed with the original 3,596 train samples)
  - 5,000  -> added to the VALIDATION set used for checkpoint selection
  - ~16,000 -> held out completely, touched only for the FINAL report below

Plus 5,000 real malicious (data.csv legitimate=0) samples, held out
completely, used only to confirm the fix doesn't overcorrect into "always
benign".

Never touches: original data/*.pt, data/*.npy, data/norm_stats.pt, or the
paper. Writes to model_augmented_candidate.pt, NOT model.pt -- promotion to
the live model only happens if scripts/promote_augmented_model.py's gates
all pass, run separately after reviewing this script's real numbers.

Usage: python scripts/train_with_kaggle_augmentation.py
"""
import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_pacx.gnn_classification import (  # noqa: E402
    MalwareGAT, build_weighted_sampler, evaluate_model, set_seed, train_model,
)
from core.graph_constructor import (  # noqa: E402
    D_FEATURE, EDGE_INDEX, NODE_HEADER, NODE_ENTROPY, NODE_API, NODE_NETWORK,
    NATIVE_PE_HEADER_COLS, NATIVE_ENTROPY_COLS, get_one_hot_type,
    native_row_to_node_vectors,
)
from core.input_adapter import UniversalInputAdapter  # noqa: E402

_RENAME_OVERRIDES = {"SectionMaxRawsize": "SectionsMaxRawsize"}
BENIGN_CLASS_INDEX = 0


def _their_col_to_row_dict(row, header_entropy_cols, their_cols):
    row_dict = {}
    for our_col, their_col in zip(header_entropy_cols, their_cols):
        if their_col in row:
            row_dict[our_col] = float(row[their_col])
    return row_dict


def build_graph(row_dict: dict, label_idx: int, adapter: UniversalInputAdapter) -> Data:
    vectors = native_row_to_node_vectors(row_dict.get, D_FEATURE)
    tensors = []
    for i in range(4):
        one_hot = np.array(get_one_hot_type(i), dtype=np.float32)
        comb = np.concatenate([vectors[i], one_hot])
        tensors.append(torch.tensor(comb, dtype=torch.float32))
    normed = adapter._apply_normalization(tensors)
    X = torch.stack(normed)
    y = torch.tensor([label_idx], dtype=torch.long)
    return Data(x=X, edge_index=EDGE_INDEX, y=y)


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = UniversalInputAdapter()  # loads data/norm_stats.pt, unchanged

    print("[+] Loading original dataset + splits (untouched)...")
    full_dataset = torch.load("data/pyg_dataset_norm.pt", weights_only=False)
    train_indices = np.load("data/train_indices.npy")
    val_indices = np.load("data/val_indices.npy")
    test_indices = np.load("data/test_indices.npy")
    orig_train = [full_dataset[int(i)] for i in train_indices]
    orig_val = [full_dataset[int(i)] for i in val_indices]
    orig_test = [full_dataset[int(i)] for i in test_indices]
    print(f"    original train={len(orig_train)} val={len(orig_val)} test={len(orig_test)}")

    print("[+] Loading kaggle_dataset/data.csv (real, independent PE samples)...")
    df = pd.read_csv("kaggle_dataset/data.csv", sep="|", low_memory=False)
    header_entropy_cols = NATIVE_PE_HEADER_COLS + NATIVE_ENTROPY_COLS
    their_cols = [c.replace("f_", "").rsplit("_0", 1)[0] for c in header_entropy_cols]
    their_cols = [_RENAME_OVERRIDES.get(c, c) if _RENAME_OVERRIDES.get(c, c) in df.columns else c for c in their_cols]

    rng = np.random.RandomState(123)  # different seed from other scripts -- independent split
    benign_df = df[df["legitimate"] == 1].sample(frac=1.0, random_state=123).reset_index(drop=True)
    malicious_df = df[df["legitimate"] == 0].sample(frac=1.0, random_state=123).reset_index(drop=True)

    n_train_aug, n_val_aug = 20000, 5000
    benign_train_aug = benign_df.iloc[:n_train_aug]
    benign_val_aug = benign_df.iloc[n_train_aug:n_train_aug + n_val_aug]
    benign_final_holdout = benign_df.iloc[n_train_aug + n_val_aug:]
    print(f"    real benign pool: {len(benign_df)} -> train_aug={len(benign_train_aug)} "
          f"val_aug={len(benign_val_aug)} final_holdout={len(benign_final_holdout)}")

    # First attempt (see git history / session notes) trained on benign-only
    # augmentation: the model learned "zero API+Network -> Benign" as a
    # shortcut instead of real Header+Entropy discrimination -- real benign
    # FP rate dropped to 0.02% but real malicious FN rate collapsed to 100%.
    # Fix: also augment with real malicious samples in the same zero-node
    # shape, so the model has to actually discriminate. data.csv has no
    # malware FAMILY label (only binary), and this is an 11-class model --
    # rather than invent a family, each augmented malicious sample gets a
    # family label drawn proportionally to the ORIGINAL training set's real
    # class distribution, so no single family's decision boundary gets
    # artificially distorted. This only affects binary malicious/benign
    # learning for these samples; per-family accuracy is never measured on
    # augmented data (only the original, untouched test split reports that).
    orig_family_labels = [int(g.y.item()) for g in orig_train if int(g.y.item()) != BENIGN_CLASS_INDEX]
    malicious_family_pool = np.array(orig_family_labels)

    # Abandoned: assigning augmented malicious samples a RANDOM specific
    # family label is not a scale problem, it's a fundamentally broken
    # target -- the same "zero API+Network" input distribution ends up
    # pointing at 10 different, mutually-contradictory family labels with
    # no real header/entropy signal to actually distinguish them (a data.csv
    # row has no basis for "this looks like gandcrab specifically"). Tried
    # at 15,000 (collapsed in-distribution accuracy to 84.65%) and again at
    # 1,000 (collapsed further, to 63.27%, and never stabilized during
    # training) -- less data made it WORSE, confirming this is a broken
    # approach, not a tuning problem. Reverted to benign-only augmentation;
    # see session notes on why the FN-on-header-only-malicious trade-off
    # from that approach is treated as acceptable for the real-world use
    # case (raw binary uploads, where PAC-X has real signal and the fusion
    # layer provides the actual safety net), not silently accepted.
    n_train_mal_aug, n_val_mal_aug = 0, 0
    malicious_train_aug = malicious_df.iloc[:n_train_mal_aug]
    malicious_val_aug = malicious_df.iloc[n_train_mal_aug:n_train_mal_aug + n_val_mal_aug]
    malicious_final_holdout = malicious_df.iloc[n_train_mal_aug + n_val_mal_aug:n_train_mal_aug + n_val_mal_aug + 5000]
    print(f"    real malicious pool: {len(malicious_df)} -> train_aug={len(malicious_train_aug)} "
          f"val_aug={len(malicious_val_aug)} final_holdout={len(malicious_final_holdout)}")

    mal_train_labels = rng.choice(malicious_family_pool, size=len(malicious_train_aug))
    mal_val_labels = rng.choice(malicious_family_pool, size=len(malicious_val_aug))

    print("[+] Building augmented training graphs (Header+Entropy real, API+Network honestly zero)...")
    aug_train_graphs = [
        build_graph(_their_col_to_row_dict(row, header_entropy_cols, their_cols), BENIGN_CLASS_INDEX, adapter)
        for _, row in benign_train_aug.iterrows()
    ] + [
        build_graph(_their_col_to_row_dict(row, header_entropy_cols, their_cols), int(label), adapter)
        for (_, row), label in zip(malicious_train_aug.iterrows(), mal_train_labels)
    ]
    aug_val_graphs = [
        build_graph(_their_col_to_row_dict(row, header_entropy_cols, their_cols), BENIGN_CLASS_INDEX, adapter)
        for _, row in benign_val_aug.iterrows()
    ] + [
        build_graph(_their_col_to_row_dict(row, header_entropy_cols, their_cols), int(label), adapter)
        for (_, row), label in zip(malicious_val_aug.iterrows(), mal_val_labels)
    ]

    # Save the final-holdout splits (as raw row_dicts, not graphs) so the
    # promotion/validation script can rebuild them identically without
    # re-running this whole script.
    os.makedirs("data/kaggle_augmentation", exist_ok=True)
    benign_final_holdout.to_csv("data/kaggle_augmentation/benign_final_holdout.csv", sep="|", index=False)
    malicious_final_holdout.to_csv("data/kaggle_augmentation/malicious_final_holdout.csv", sep="|", index=False)
    print(f"[+] Saved final-holdout splits to data/kaggle_augmentation/ (untouched by training)")

    combined_train = orig_train + aug_train_graphs
    combined_val = orig_val + aug_val_graphs
    print(f"[+] Combined train={len(combined_train)} (orig {len(orig_train)} + aug {len(aug_train_graphs)}), "
          f"combined val={len(combined_val)} (orig {len(orig_val)} + aug {len(aug_val_graphs)})")

    sampler, class_weights = build_weighted_sampler(combined_train, num_classes=11)
    train_loader = DataLoader(combined_train, batch_size=32, sampler=sampler)
    val_loader = DataLoader(combined_val, batch_size=32, shuffle=False)
    orig_test_loader = DataLoader(orig_test, batch_size=64, shuffle=False)

    print("[+] Loading current model.pt weights (fine-tuning, not training from scratch)...")
    model = MalwareGAT(num_node_features=260, num_classes=11).to(device)
    model.load_state_dict(torch.load("model.pt", map_location=device))

    pre_test_metrics = evaluate_model(model, orig_test_loader, torch.nn.CrossEntropyLoss(), device)
    print(f"[*] Pre-finetune accuracy on ORIGINAL held-out test set: {pre_test_metrics['accuracy']*100:.2f}%")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0007, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    print("\n--- Fine-tuning on original + real-benign-augmented data ---")
    model, history = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, criterion=criterion, scheduler=scheduler,
        device=device, epochs=15, patience=5,
    )

    post_test_metrics = evaluate_model(model, orig_test_loader, torch.nn.CrossEntropyLoss(), device)
    print(f"\n[*] Post-finetune accuracy on ORIGINAL held-out test set: {post_test_metrics['accuracy']*100:.2f}%")
    print(f"[*] Change: {(post_test_metrics['accuracy'] - pre_test_metrics['accuracy'])*100:+.2f} points")

    torch.save(model.state_dict(), "model_augmented_candidate.pt")
    with open("data/kaggle_augmentation/finetune_history.json", "w") as f:
        json.dump({
            "pre_finetune_original_test_accuracy": round(pre_test_metrics["accuracy"] * 100, 2),
            "post_finetune_original_test_accuracy": round(post_test_metrics["accuracy"] * 100, 2),
            "epoch_history": history,
        }, f, indent=2)
    print("[+] Saved model_augmented_candidate.pt (NOT model.pt) and data/kaggle_augmentation/finetune_history.json")
    print("[+] Next: python scripts/validate_augmented_model.py")


if __name__ == "__main__":
    main()
