"""
Computes real, data-derived out-of-distribution thresholds for
core/pe_feature_extractor.py's check_node_ood(), used by the live raw-PE-
binary upload path (core/input_adapter.py's _parse_pe_binary).

For each of the 4 graph nodes, this measures mean|z-score| across the 256
real feature dims for every genuine TRAINING sample (data/train_indices.npy),
after the same normalization used everywhere else (data/norm_stats.pt), and
saves the 99th percentile as data/ood_node_thresholds.json. That's the bar
99% of real training data actually clears -- not a guessed cutoff.

Usage: python scripts/build_ood_thresholds.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NODE_NAMES = ["header", "entropy", "api", "network"]


def main():
    dataset = torch.load("data/pyg_dataset_norm.pt", weights_only=False)
    train_indices = np.load("data/train_indices.npy")
    train_samples = [dataset[int(i)] for i in train_indices]

    per_node_values = {name: [] for name in NODE_NAMES}
    for graph in train_samples:
        x = graph.x  # [4, 260]: 256 real dims + 4 one-hot dims per node
        for node_idx, name in enumerate(NODE_NAMES):
            per_node_values[name].append(float(x[node_idx, :256].abs().mean().item()))

    thresholds = {}
    for name in NODE_NAMES:
        arr = np.array(per_node_values[name])
        thresholds[name] = {
            "p99": round(float(np.percentile(arr, 99)), 4),
            "max": round(float(arr.max()), 4),
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4),
        }
        print(f"{name:8s} p99={thresholds[name]['p99']:.4f}  max={thresholds[name]['max']:.4f}  "
              f"mean={thresholds[name]['mean']:.4f}  std={thresholds[name]['std']:.4f}")

    report = {
        "note": (
            "Per-node mean-absolute-z-score thresholds, computed from the REAL "
            "training split (data/train_indices.npy) after normalization "
            "(data/norm_stats.pt). Used by core/pe_feature_extractor.py's "
            "check_node_ood() to flag a live PE-binary upload's node as "
            "statistically implausible relative to what the model actually saw "
            "during training, instead of trusting it blindly. p99 = 99th "
            "percentile of genuine training samples' own mean|z| per node; "
            "anything past this is more extreme than 99% of real training data."
        ),
        "n_train_samples": len(train_samples),
        "thresholds": thresholds,
    }
    with open("data/ood_node_thresholds.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[+] Saved data/ood_node_thresholds.json")


if __name__ == "__main__":
    main()
