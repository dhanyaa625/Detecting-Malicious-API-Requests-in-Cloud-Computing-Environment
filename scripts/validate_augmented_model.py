"""
Validates model_augmented_candidate.pt against every gate before it's
allowed to replace the live model.pt:
  1. Original held-out test set (data/test_indices.npy) -- must not
     meaningfully regress from 99.61%.
  2. Real benign final-holdout (data/kaggle_augmentation/benign_final_holdout.csv,
     ~16,323 samples, never used in training or checkpoint selection) --
     false-positive rate should measurably improve from the 95.5% baseline.
  3. Real malicious final-holdout (5,000 samples, never trained on) --
     confirms no overcorrection into "always benign".

Reports real numbers for both model_augmented_candidate.pt and the
original model.pt.pre_augmentation_backup, side by side, on the same data.
Does NOT touch model.pt -- promotion is a separate, explicit step after
reviewing these numbers.

Usage: python scripts/validate_augmented_model.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_pacx.gnn_classification import MalwareGAT, evaluate_model  # noqa: E402
from core.graph_constructor import NATIVE_PE_HEADER_COLS, NATIVE_ENTROPY_COLS  # noqa: E402
from core.input_adapter import UniversalInputAdapter  # noqa: E402
from scripts.train_with_kaggle_augmentation import build_graph, _their_col_to_row_dict, BENIGN_CLASS_INDEX  # noqa: E402

_RENAME_OVERRIDES = {"SectionMaxRawsize": "SectionsMaxRawsize"}


def load_model(path, device):
    model = MalwareGAT(num_node_features=260, num_classes=11).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def predict_binary(model, graphs, device):
    """Returns list of (predicted_binary, true_binary) -- 1=malicious, 0=benign."""
    preds = []
    with torch.no_grad():
        for g in graphs:
            g = g.to(device)
            logits = model(g.x, g.edge_index, torch.zeros(g.x.size(0), dtype=torch.long, device=device))
            pred_class = torch.argmax(logits, dim=1).item()
            preds.append((0 if pred_class == BENIGN_CLASS_INDEX else 1, int(g.y.item())))
    return preds


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = UniversalInputAdapter()

    header_entropy_cols = NATIVE_PE_HEADER_COLS + NATIVE_ENTROPY_COLS
    their_cols = [c.replace("f_", "").rsplit("_0", 1)[0] for c in header_entropy_cols]

    print("[+] Loading held-out real benign/malicious samples (built once, reused for both models)...")
    benign_holdout = pd.read_csv("data/kaggle_augmentation/benign_final_holdout.csv", sep="|", low_memory=False)
    malicious_holdout = pd.read_csv("data/kaggle_augmentation/malicious_final_holdout.csv", sep="|", low_memory=False)

    def to_graphs(sub_df, label_idx):
        graphs = []
        for _, row in sub_df.iterrows():
            their_cols_present = [_RENAME_OVERRIDES.get(c, c) if _RENAME_OVERRIDES.get(c, c) in row else c for c in their_cols]
            row_dict = _their_col_to_row_dict(row, header_entropy_cols, their_cols_present)
            graphs.append(build_graph(row_dict, label_idx, adapter))
        return graphs

    benign_graphs = to_graphs(benign_holdout, BENIGN_CLASS_INDEX)
    malicious_graphs = to_graphs(malicious_holdout, 1)  # placeholder non-benign label; only binary comparison used below
    print(f"    benign holdout: {len(benign_graphs)}   malicious holdout: {len(malicious_graphs)}")

    print("[+] Loading original held-out test set (in-distribution, unaffected by augmentation)...")
    full_dataset = torch.load("data/pyg_dataset_norm.pt", weights_only=False)
    test_indices = np.load("data/test_indices.npy")
    orig_test = [full_dataset[int(i)] for i in test_indices]
    orig_test_loader = DataLoader(orig_test, batch_size=64, shuffle=False)

    results = {}
    for tag, path in [("original", "model.pt.pre_augmentation_backup"), ("augmented_candidate", "model_augmented_candidate.pt")]:
        print(f"\n=== {tag} ({path}) ===")
        model = load_model(path, device)

        orig_metrics = evaluate_model(model, orig_test_loader, torch.nn.CrossEntropyLoss(), device)
        print(f"Original in-distribution test accuracy: {orig_metrics['accuracy']*100:.2f}%")

        benign_preds = predict_binary(model, benign_graphs, device)
        fp = sum(1 for pred, true in benign_preds if pred == 1)
        fp_rate = round(100 * fp / len(benign_preds), 2)
        print(f"Real held-out BENIGN samples ({len(benign_preds)}): false-positive rate = {fp_rate}%")

        malicious_preds_raw = []
        with torch.no_grad():
            for g in malicious_graphs:
                g = g.to(device)
                logits = model(g.x, g.edge_index, torch.zeros(g.x.size(0), dtype=torch.long, device=device))
                pred_class = torch.argmax(logits, dim=1).item()
                malicious_preds_raw.append(0 if pred_class == BENIGN_CLASS_INDEX else 1)
        fn = sum(1 for pred in malicious_preds_raw if pred == 0)  # predicted benign but is real malware
        fn_rate = round(100 * fn / len(malicious_preds_raw), 2)
        print(f"Real held-out MALICIOUS samples ({len(malicious_preds_raw)}): false-negative rate = {fn_rate}%")

        results[tag] = {
            "in_distribution_test_accuracy": round(orig_metrics["accuracy"] * 100, 2),
            "real_benign_false_positive_rate": fp_rate,
            "real_malicious_false_negative_rate": fn_rate,
            "n_benign_tested": len(benign_preds),
            "n_malicious_tested": len(malicious_preds_raw),
        }

    print("\n=== COMPARISON ===")
    print(json.dumps(results, indent=2))

    with open("data/kaggle_augmentation/validation_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[+] Saved data/kaggle_augmentation/validation_report.json")

    # Gate check
    orig, aug = results["original"], results["augmented_candidate"]
    gate1 = aug["in_distribution_test_accuracy"] >= orig["in_distribution_test_accuracy"] - 2.0
    gate2 = aug["real_benign_false_positive_rate"] < orig["real_benign_false_positive_rate"]
    gate3 = aug["real_malicious_false_negative_rate"] < 50.0
    print(f"\nGate 1 (in-distribution test not regressed >2pts): {'PASS' if gate1 else 'FAIL'}")
    print(f"Gate 2 (real benign FP rate improved): {'PASS' if gate2 else 'FAIL'}")
    print(f"Gate 3 (real malicious FN rate < 50%, no overcorrection): {'PASS' if gate3 else 'FAIL'}")
    print(f"\n{'ALL GATES PASSED -- safe to promote' if (gate1 and gate2 and gate3) else 'NOT ALL GATES PASSED -- do not promote'}")


if __name__ == "__main__":
    main()
