"""
Zero-day generalization experiment: hold out one entire malware family from
training (not just from the test split -- from train AND validation too),
train fresh on the remaining classes, then measure how well the model
flags the held-out family's samples as malicious despite never having seen
that family's static-analysis signature. This is the experiment the
standard stratified evaluation in agentic_pacx/gnn_classification.py does
NOT run -- every one of the 11 classes there appears in both train and
test, so that number measures known-family classification accuracy, not
zero-day generalization.

Everything here is self-contained under data/zeroday_<family>/ and
model_zeroday_<family>.pt -- it never touches the main pipeline's
data/pyg_dataset*.pt, data/*_indices.npy, data/norm_stats.pt, or model.pt.

Usage: python scripts/build_zeroday_experiment.py --holdout emotet
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph_constructor import (
    D_FEATURE, EDGE_INDEX, NODE_HEADER, NODE_ENTROPY, NODE_API, NODE_NETWORK,
    get_one_hot_type, native_row_to_node_vectors,
)
from agentic_pacx.gnn_classification import (
    MalwareGAT, build_weighted_sampler, evaluate_model, train_model, set_seed,
)


def row_to_graph(row, label):
    vectors = native_row_to_node_vectors(row.get, D_FEATURE)
    final = [
        np.concatenate([vectors[node_idx], get_one_hot_type(node_idx)])
        for node_idx in (NODE_HEADER, NODE_ENTROPY, NODE_API, NODE_NETWORK)
    ]
    X = torch.tensor(np.stack(final), dtype=torch.float)
    y = torch.tensor([label], dtype=torch.long) if label is not None else None
    return Data(x=X, edge_index=EDGE_INDEX, y=y)


def stratified_split_3way(labels, ratios=(0.7, 0.1, 0.2), seed=42):
    train_ratio, val_ratio, _ = ratios
    grouped = {}
    for idx, label in enumerate(labels):
        grouped.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    train_idx, val_idx, test_idx = [], [], []
    for label, indices in grouped.items():
        indices = list(indices)
        rng.shuffle(indices)
        n = len(indices)
        if n < 3:
            train_idx.extend(indices)
            continue
        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio))
        n_train = min(n_train, n - 2)
        n_val = min(n_val, n - n_train - 1)
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def apply_normalize(dataset, mean_std_by_node):
    dim_features = dataset[0].x.shape[1] - 4
    for i, g in enumerate(dataset):
        for node_idx in range(4):
            mean, std = mean_std_by_node[node_idx]
            g.x[node_idx, :dim_features] = (g.x[node_idx, :dim_features] - mean.squeeze(0)) / std.squeeze(0)
    return dataset


def compute_norm_stats(dataset, train_indices):
    dim_features = dataset[0].x.shape[1] - 4
    train_set = set(train_indices)
    train_mask = [i in train_set for i in range(len(dataset))]
    stats = {}
    for node_idx in range(4):
        all_feats = torch.stack([g.x[node_idx, :dim_features] for g in dataset])
        train_feats = all_feats[train_mask]
        mean = train_feats.mean(dim=0, keepdim=True)
        std = train_feats.std(dim=0, keepdim=True)
        std[std == 0] = 1.0
        stats[node_idx] = (mean, std)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", required=True, help="Malware family label to hold out entirely (e.g. emotet)")
    parser.add_argument("--input", default="data/cleaned_data.csv")
    parser.add_argument("--out-dir", default=None, help="Defaults to data/zeroday_<holdout>/")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    holdout = args.holdout.strip().lower()
    out_dir = args.out_dir or f"data/zeroday_{holdout}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[*] Loading {args.input} ...")
    df = pd.read_csv(args.input, low_memory=False)
    if holdout not in set(df["label"].str.lower()):
        raise ValueError(f"'{holdout}' not found in label column. Available: {sorted(df['label'].unique())}")

    df["label"] = df["label"].str.lower()
    known_df = df[df["label"] != holdout].reset_index(drop=True)
    heldout_df = df[df["label"] == holdout].reset_index(drop=True)

    class_names = sorted(known_df["label"].unique().tolist())
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}
    benign_idx = label_to_idx["benign"]
    print(f"[+] Holding out '{holdout}' ({len(heldout_df)} samples) entirely from train/val/test.")
    print(f"[+] Remaining {len(class_names)} classes (sorted): {class_names}")

    print("[*] Building graphs for the known-class pool ...")
    known_graphs = [
        row_to_graph(row, label_to_idx[row["label"]]) for _, row in known_df.iterrows()
    ]
    print("[*] Building graphs for the held-out family (no valid class index -- label kept separately) ...")
    heldout_graphs = [row_to_graph(row, None) for _, row in heldout_df.iterrows()]

    labels = [int(g.y.item()) for g in known_graphs]
    train_idx, val_idx, test_idx = stratified_split_3way(labels, seed=args.seed)
    print(f"[+] Known-class split: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")

    norm_stats = compute_norm_stats(known_graphs, train_idx)
    known_graphs = apply_normalize(known_graphs, norm_stats)
    heldout_graphs = apply_normalize(heldout_graphs, norm_stats)

    train_dataset = [known_graphs[i] for i in train_idx]
    val_dataset = [known_graphs[i] for i in val_idx]
    test_dataset = [known_graphs[i] for i in test_idx]

    sampler, class_weights = build_weighted_sampler(train_dataset, num_classes=len(class_names))
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MalwareGAT(num_node_features=260, num_classes=len(class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    print("\n--- Training on known classes only (holdout family fully excluded) ---")
    model, history = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, criterion=criterion, scheduler=scheduler,
        device=device, epochs=args.epochs, patience=8,
    )

    known_test_metrics = evaluate_model(model, test_loader, criterion, device)
    print(
        f"[+] Known-class held-out test (sanity check, {len(class_names)}-class): "
        f"Accuracy {known_test_metrics['accuracy']*100:.2f}% | F1(macro) {known_test_metrics['f1_macro']*100:.2f}%"
    )

    # ---- The actual zero-day measurement ----
    print(f"\n--- Evaluating on {len(heldout_graphs)} held-out '{holdout}' samples (never seen in training) ---")
    model.eval()
    heldout_loader = DataLoader(heldout_graphs, batch_size=64, shuffle=False)
    predicted_benign = 0
    predicted_malicious = 0
    confidences = []
    below_tau = 0
    predicted_family_counts = {}
    with torch.no_grad():
        for batch in heldout_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)
            for p, c in zip(pred.tolist(), conf.tolist()):
                confidences.append(c)
                if p == benign_idx:
                    predicted_benign += 1
                else:
                    predicted_malicious += 1
                if c < 0.60:
                    below_tau += 1
                fam = class_names[p]
                predicted_family_counts[fam] = predicted_family_counts.get(fam, 0) + 1

    n = len(heldout_graphs)
    zero_day_catch_rate = predicted_malicious / n if n else 0.0
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    median_conf = float(np.median(confidences)) if confidences else 0.0
    agent2_trigger_rate = below_tau / n if n else 0.0

    report = {
        "holdout_family": holdout,
        "holdout_n_samples": n,
        "known_classes": class_names,
        "known_class_split": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "known_class_test_accuracy": round(known_test_metrics["accuracy"] * 100, 2),
        "known_class_test_f1_macro": round(known_test_metrics["f1_macro"] * 100, 2),
        "zero_day_catch_rate": round(zero_day_catch_rate * 100, 2),
        "predicted_benign_count": predicted_benign,
        "predicted_malicious_count": predicted_malicious,
        "mean_confidence": round(mean_conf * 100, 2),
        "median_confidence": round(median_conf * 100, 2),
        "agent2_trigger_rate": round(agent2_trigger_rate * 100, 2),
        "agent2_trigger_count": below_tau,
        "predicted_family_distribution": predicted_family_counts,
    }

    print(f"\n[RESULT] Zero-day catch rate (flagged malicious, not benign): {report['zero_day_catch_rate']}% "
          f"({predicted_malicious}/{n})")
    print(f"[RESULT] Mean confidence on held-out family: {report['mean_confidence']}% "
          f"(median {report['median_confidence']}%)")
    print(f"[RESULT] Agent 2 trigger rate (confidence < 60%): {report['agent2_trigger_rate']}% "
          f"({below_tau}/{n})")
    print(f"[RESULT] Predicted-family distribution: {predicted_family_counts}")

    report_path = os.path.join(out_dir, "zeroday_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[+] Full report saved to {report_path}")

    model_path = os.path.join(out_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[+] Zero-day-holdout model weights saved to {model_path} (not used by the live app)")


if __name__ == "__main__":
    main()
