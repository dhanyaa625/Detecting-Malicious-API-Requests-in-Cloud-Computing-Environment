import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data


class DatasetManager:
    """
    Professional manager for graph dataset splitting and normalization.

    Pipeline order matters: the stratified train/val/test split is computed
    *before* normalization, and normalization statistics are derived from the
    train split only. Computing stats over the whole dataset (as the previous
    version of this file did) leaks test-set information into every feature's
    mean/std -- mild but real train/test leakage, separate from (and in
    addition to) not having a validation split at all.
    """

    def __init__(self, raw_graph_path="data/pyg_dataset.pt",
                 output_path="data/pyg_dataset_norm.pt",
                 stats_path="data/norm_stats.pt",
                 index_dir="data"):
        self.raw_graph_path = raw_graph_path
        self.output_path = output_path
        self.stats_path = stats_path
        self.index_dir = index_dir

    def load_raw(self) -> List[Data]:
        print(f"[*] Loading raw graphs from {self.raw_graph_path} ...")
        dataset = torch.load(self.raw_graph_path, weights_only=False)
        missing_labels = sum(1 for g in dataset if g.y is None)
        if missing_labels:
            raise ValueError(
                f"{missing_labels} graphs have no label -- build the dataset "
                f"with scripts/build_graph_dataset.py, which labels every graph "
                f"at construction time."
            )
        return dataset

    def split(self, dataset: List[Data], ratios: Tuple[float, float, float] = (0.7, 0.1, 0.2),
              seed: int = 42) -> Dict[str, np.ndarray]:
        """
        Per-class stratified split into train/val/test. A class with only one
        sample goes entirely to train (with a warning) rather than being
        silently duplicated into both train and test, which is what the
        previous 2-way split in agentic_pacx/gnn_classification.py did.
        """
        train_ratio, val_ratio, _test_ratio = ratios
        assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"

        grouped: Dict[int, List[int]] = {}
        for idx, sample in enumerate(dataset):
            label = int(sample.y.item())
            grouped.setdefault(label, []).append(idx)

        rng = random.Random(seed)
        train_idx, val_idx, test_idx = [], [], []
        for label, indices in grouped.items():
            indices = list(indices)
            rng.shuffle(indices)
            n = len(indices)

            if n < 3:
                print(
                    f"[!] Class {label} has only {n} sample(s) -- too few to "
                    f"split across train/val/test; assigning entirely to train."
                )
                train_idx.extend(indices)
                continue

            n_train = max(1, round(n * train_ratio))
            n_val = max(1, round(n * val_ratio))
            # Keep at least 1 sample in test; clamp so we never overrun n.
            n_train = min(n_train, n - 2)
            n_val = min(n_val, n - n_train - 1)

            train_idx.extend(indices[:n_train])
            val_idx.extend(indices[n_train:n_train + n_val])
            test_idx.extend(indices[n_train + n_val:])

        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)

        train_arr = np.array(train_idx, dtype=np.int64)
        val_arr = np.array(val_idx, dtype=np.int64)
        test_arr = np.array(test_idx, dtype=np.int64)

        assert len(set(train_arr.tolist()) & set(val_arr.tolist())) == 0
        assert len(set(train_arr.tolist()) & set(test_arr.tolist())) == 0
        assert len(set(val_arr.tolist()) & set(test_arr.tolist())) == 0

        print(
            f"[+] Split: {len(train_arr)} train / {len(val_arr)} val / "
            f"{len(test_arr)} test (of {len(dataset)} total)."
        )
        return {"train": train_arr, "val": val_arr, "test": test_arr}

    def normalize(self, dataset: List[Data], train_indices: np.ndarray) -> Tuple[List[Data], dict]:
        """
        Standardizes each node type's features to train-set mean=0, std=1.
        Stats are computed from train_indices only, then applied to every
        sample (train/val/test alike) -- this is what makes the val/test
        numbers an honest estimate of generalization instead of leaking
        their own distribution into the normalization.
        """
        print("[*] Normalizing topological features (train-set statistics only) ...")
        dim_features = dataset[0].x.shape[1] - 4  # last 4 cols are the node-type one-hot

        stats = {}
        train_set = set(int(i) for i in train_indices)
        train_mask = [i in train_set for i in range(len(dataset))]

        for node_idx in range(4):
            all_feats = torch.stack([g.x[node_idx, :dim_features] for g in dataset])
            train_feats = all_feats[train_mask]

            mean = train_feats.mean(dim=0, keepdim=True)
            std = train_feats.std(dim=0, keepdim=True)
            std[std == 0] = 1.0

            stats[node_idx] = {"mean": mean, "std": std}
            norm_feats = (all_feats - mean) / std
            for i in range(len(dataset)):
                dataset[i].x[node_idx, :dim_features] = norm_feats[i]

        print("[+] Normalization complete.")
        return dataset, stats

    def save(self, dataset: List[Data], splits: Dict[str, np.ndarray], stats: dict):
        torch.save(dataset, self.output_path)
        print(f"[+] Normalized dataset saved to {self.output_path}")

        for name, indices in splits.items():
            path = os.path.join(self.index_dir, f"{name}_indices.npy")
            np.save(path, indices)
            print(f"[+] {name} indices ({len(indices)}) saved to {path}")

        torch.save(stats, self.stats_path)
        print(f"[+] Normalization stats saved to {self.stats_path}")

    def process_complete_pipeline(self, ratios=(0.7, 0.1, 0.2), seed=42):
        print("--- Initiating Data Pipeline: split -> normalize -> save ---")
        dataset = self.load_raw()
        splits = self.split(dataset, ratios=ratios, seed=seed)
        dataset, stats = self.normalize(dataset, splits["train"])
        self.save(dataset, splits, stats)
        return {"success": True, "message": "Pipeline completed successfully.", "splits": {k: len(v) for k, v in splits.items()}}


if __name__ == "__main__":
    manager = DatasetManager()
    result = manager.process_complete_pipeline()
    print(json.dumps(result, indent=2))
