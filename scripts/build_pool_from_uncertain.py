"""
Builds a legitimate Agent-2 retrain pool from the genuinely uncertain
predictions (confidence < 0.6) recorded in gnn_outputs.json by the last
training run's Phase-2 inference pass. Each sample carries its real
ground-truth label (not a live-analysis pseudo-label), recovered by
reversing the test_loader's batch_idx/item_idx back to the original
dataset index (test_loader was built with batch_size=32, shuffle=False).

Usage: python scripts/build_pool_from_uncertain.py
"""
import json

import numpy as np
import torch

BATCH_SIZE = 32


def main():
    with open("gnn_outputs.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    uncertain = data["uncertain_predictions_zero_day"]

    dataset = torch.load("data/pyg_dataset_norm.pt", weights_only=False)
    test_indices = np.load("data/test_indices.npy")

    pool = []
    for entry in uncertain:
        position = entry["batch_idx"] * BATCH_SIZE + entry["item_idx"]
        dataset_index = int(test_indices[position])
        graph = dataset[dataset_index]
        # Sanity check: the graph's real label must match what was recorded.
        assert int(graph.y.item()) == entry["actual_label"], (
            f"Mismatch at position {position}: graph.y={graph.y.item()} "
            f"vs recorded actual_label={entry['actual_label']}"
        )
        pool.append(graph)

    torch.save(pool, "data/retrain_pool.pt")
    print(f"[+] Built retrain pool with {len(pool)} genuinely uncertain, real-labeled samples.")


if __name__ == "__main__":
    main()
