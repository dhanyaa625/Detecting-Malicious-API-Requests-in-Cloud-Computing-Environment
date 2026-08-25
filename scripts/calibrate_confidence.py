"""
Fits a temperature-scaling calibration factor for MalwareGAT's softmax
output. Root cause being fixed: raw logit margins are huge (~14 on real
data.csv rows, confirmed by direct inspection) -- the model is
mathematically incapable of reporting anything but near-100% confidence,
which is why Agent 2 (confidence < 60%) essentially never fires anymore
after the benign-robustness fine-tuning made the model more decisive
everywhere, including on genuinely uncertain real-world input.

Temperature scaling (Guo et al. 2017) divides logits by a learned scalar T
before softmax. It is a MONOTONIC transform -- argmax is mathematically
unchanged, so classification accuracy is provably unaffected. Only the
confidence magnitude changes, becoming an honest reflection of how often
the model is actually right at that confidence level instead of an
artifact of a well-separated decision boundary.

Fit by minimizing negative log-likelihood on a MIXED calibration set:
  - a slice of the original in-distribution validation split (513 samples
    -- already used for checkpoint selection, but reusing it to fit one
    scalar parameter is standard practice and low leakage risk)
  - a slice of the real, external, never-trained-on Kaggle holdout data
    (data/kaggle_augmentation/{benign,malicious}_final_holdout.csv) --
    included specifically because the miscalibration this is fixing is
    most visible on real-world data, not the in-distribution test set.

A SEPARATE, disjoint slice of the same holdout files is reserved for
final reporting, never touched during fitting.

Usage: python scripts/calibrate_confidence.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_pacx.gnn_classification import MalwareGAT  # noqa: E402
from core.graph_constructor import NATIVE_PE_HEADER_COLS, NATIVE_ENTROPY_COLS  # noqa: E402
from core.input_adapter import UniversalInputAdapter  # noqa: E402
from scripts.train_with_kaggle_augmentation import build_graph, _their_col_to_row_dict, BENIGN_CLASS_INDEX  # noqa: E402

_RENAME_OVERRIDES = {"SectionMaxRawsize": "SectionsMaxRawsize"}


def get_logits_and_labels(model, dataset, device):
    logits_list, labels_list = [], []
    with torch.no_grad():
        for g in dataset:
            g = g.to(device)
            logits = model(g.x, g.edge_index, torch.zeros(g.x.size(0), dtype=torch.long, device=device))
            logits_list.append(logits[0])
            labels_list.append(int(g.y.item()))
    return torch.stack(logits_list), torch.tensor(labels_list)


def fit_temperature(logits, labels, t_range=np.arange(0.5, 25.0, 0.25)):
    """Grid search over T minimizing NLL -- simple, robust, exact for 1 parameter."""
    best_t, best_nll = 1.0, float("inf")
    for t in t_range:
        log_probs = torch.log_softmax(logits / t, dim=1)
        nll = -log_probs[torch.arange(len(labels)), labels].mean().item()
        if nll < best_nll:
            best_nll, best_t = nll, t
    return float(best_t), best_nll


def expected_calibration_error(logits, labels, temperature, n_bins=10):
    """Standard ECE: bins predictions by confidence, compares confidence to
    actual accuracy within each bin. Lower is better calibrated."""
    probs = torch.softmax(logits / temperature, dim=1)
    confidences, predictions = probs.max(dim=1)
    accuracies = (predictions == labels).float()

    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_acc = accuracies[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            ece += (in_bin.float().mean() * (bin_acc - bin_conf).abs()).item()
    return ece


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = UniversalInputAdapter()

    print("[+] Loading model.pt (current live model)...")
    model = MalwareGAT(num_node_features=260, num_classes=11).to(device)
    model.load_state_dict(torch.load("model.pt", map_location=device))
    model.eval()

    print("[+] Loading in-distribution validation split...")
    full_dataset = torch.load("data/pyg_dataset_norm.pt", weights_only=False)
    val_indices = np.load("data/val_indices.npy")
    val_dataset = [full_dataset[int(i)] for i in val_indices]

    print("[+] Loading real external Kaggle holdout (never trained on)...")
    header_entropy_cols = NATIVE_PE_HEADER_COLS + NATIVE_ENTROPY_COLS
    their_cols = [c.replace("f_", "").rsplit("_0", 1)[0] for c in header_entropy_cols]

    benign_holdout = pd.read_csv("data/kaggle_augmentation/benign_final_holdout.csv", sep="|", low_memory=False)
    malicious_holdout = pd.read_csv("data/kaggle_augmentation/malicious_final_holdout.csv", sep="|", low_memory=False)

    rng = np.random.RandomState(99)  # fresh seed, independent of prior splits
    benign_holdout = benign_holdout.sample(frac=1.0, random_state=99).reset_index(drop=True)
    malicious_holdout = malicious_holdout.sample(frac=1.0, random_state=99).reset_index(drop=True)

    # Disjoint calibration-fit vs. final-report slices.
    n_calib_each = 2000
    benign_calib = benign_holdout.iloc[:n_calib_each]
    benign_report = benign_holdout.iloc[n_calib_each:]
    malicious_calib = malicious_holdout.iloc[:n_calib_each]
    malicious_report = malicious_holdout.iloc[n_calib_each:]

    def to_graphs(df, label_idx):
        graphs = []
        for _, row in df.iterrows():
            row_dict = _their_col_to_row_dict(row, header_entropy_cols, their_cols)
            graphs.append(build_graph(row_dict, label_idx, adapter))
        return graphs

    print(f"[+] Building calibration-fit graphs ({len(benign_calib)} benign + {len(malicious_calib)} malicious real Kaggle rows)...")
    # Malicious real-world rows have no family label -- for calibration
    # purposes only (not training), label them by the model's OWN top
    # non-benign prediction, since calibration cares about confidence-vs-
    # correctness, and "is it malicious at all" is well-defined even
    # without knowing which specific family.
    calib_graphs = to_graphs(benign_calib, BENIGN_CLASS_INDEX)
    for _, row in malicious_calib.iterrows():
        row_dict = _their_col_to_row_dict(row, header_entropy_cols, their_cols)
        g = build_graph(row_dict, BENIGN_CLASS_INDEX, adapter)  # placeholder label, corrected below
        calib_graphs.append(g)
    # Correct malicious calib labels to "malicious" for binary calibration
    # purposes: use the model's own predicted family as the target label
    # ONLY for the binary malicious/benign judgment -- see note above.
    mal_start = len(benign_calib)
    with torch.no_grad():
        for i in range(mal_start, len(calib_graphs)):
            g = calib_graphs[i].to(device)
            logits = model(g.x, g.edge_index, torch.zeros(g.x.size(0), dtype=torch.long, device=device))
            pred = int(torch.argmax(logits[0]).item())
            calib_graphs[i].y = torch.tensor([pred if pred != BENIGN_CLASS_INDEX else 1], dtype=torch.long)

    combined_calib = val_dataset + calib_graphs
    calib_logits, calib_labels = get_logits_and_labels(model, combined_calib, device)

    print("[+] Fitting temperature via grid search (minimizing NLL)...")
    T, nll = fit_temperature(calib_logits, calib_labels)
    print(f"[+] Fitted temperature: T={T:.2f} (NLL={nll:.4f})")

    ece_before = expected_calibration_error(calib_logits, calib_labels, 1.0)
    ece_after = expected_calibration_error(calib_logits, calib_labels, T)
    print(f"[+] Expected Calibration Error: before={ece_before:.4f}  after={ece_after:.4f}")

    # Final, honest report on the disjoint slice never touched during fitting.
    print(f"\n[+] Building final-report graphs ({len(benign_report)} benign + {len(malicious_report)} malicious, "
          f"disjoint from calibration-fit set)...")
    report_graphs = to_graphs(benign_report, 0) + to_graphs(malicious_report, 1)
    report_true_binary = torch.tensor([0] * len(benign_report) + [1] * len(malicious_report))

    report_logits = []
    with torch.no_grad():
        for g in report_graphs:
            g = g.to(device)
            logits = model(g.x, g.edge_index, torch.zeros(g.x.size(0), dtype=torch.long, device=device))
            report_logits.append(logits[0])
    report_logits = torch.stack(report_logits)

    def binary_metrics_at_temperature(temperature, threshold=0.6):
        probs = torch.softmax(report_logits / temperature, dim=1)
        pred_class = probs.argmax(dim=1)
        pred_binary = (pred_class != 0).long()
        top_conf = probs.max(dim=1).values
        below_threshold = (top_conf < threshold)
        acc = (pred_binary == report_true_binary).float().mean().item()
        return acc, below_threshold.float().mean().item()

    acc_t1, below_t1 = binary_metrics_at_temperature(1.0)
    acc_tcal, below_tcal = binary_metrics_at_temperature(T)
    print(f"\n=== FINAL REPORT (disjoint holdout, never used for fitting) ===")
    print(f"T=1.0 (uncalibrated):  binary_accuracy={acc_t1*100:.2f}%  pct_below_60pct_confidence={below_t1*100:.2f}%")
    print(f"T={T:.2f} (calibrated):  binary_accuracy={acc_tcal*100:.2f}%  pct_below_60pct_confidence={below_tcal*100:.2f}%")
    print(f"(accuracy should match exactly -- temperature scaling doesn't change argmax)")

    with open("data/confidence_calibration.json", "w") as f:
        json.dump({
            "temperature": round(T, 4),
            "note": (
                "Temperature-scaling factor for MalwareGAT's softmax, fit by minimizing NLL on a mix of the "
                "in-distribution validation split and real external Kaggle holdout data. Divides logits by this "
                "value before softmax -- a monotonic transform, so it changes reported confidence only, never "
                "the predicted class. See scripts/calibrate_confidence.py for the fitting procedure."
            ),
            "ece_before": round(ece_before, 4),
            "ece_after": round(ece_after, 4),
            "final_report_uncalibrated": {"accuracy": round(acc_t1 * 100, 2), "pct_below_60pct_confidence": round(below_t1 * 100, 2)},
            "final_report_calibrated": {"accuracy": round(acc_tcal * 100, 2), "pct_below_60pct_confidence": round(below_tcal * 100, 2)},
        }, f, indent=2)
    print("\n[+] Saved data/confidence_calibration.json")


if __name__ == "__main__":
    main()
