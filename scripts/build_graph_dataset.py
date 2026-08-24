"""
Builds the 4-node PyG graph dataset from data/cleaned_data.csv.

Replaces _legacy_archive/graph_generation.py, which built the previous
data/pyg_dataset.pt by treating f_ImportsList_*/f_ExportsList_*/f_URLs_*
(etc.) as literal API-name / string tokens and MD5-hashing them into
buckets. Direct inspection of every CSV variant in data/ shows those
columns are already pre-computed embedding floats (e.g. "0.2359422..."),
not strings -- so the old pipeline was hashing stringified floats, and
the resulting "API Import/Export" and "Network" node features carried no
real API/network semantics at all, just a stable-but-meaningless bucket
pattern.

Fix: treat all four nodes the same principled way Header/Entropy already
were -- pad_numeric_features() applied directly to the real numeric
columns for that node group, reusing core/graph_constructor.py's shared
helpers instead of a separate hashing path. Node column groups:

  Header  (0): the 30 raw PE header fields (unchanged from before).
  Entropy (1): the 10 section entropy/size stats *plus* 13 structural
               summary scalars (Imports/Exports/Resources counts and
               sizes) that the old pipeline computed but never used.
  API     (2): all 128 ImportsList + 128 ExportsList embedding values
               (256 total -- exactly fills D_FEATURE, no padding needed).
  Network (3): all 10 STRING_GROUPS x 16 embedding values (160 real,
               padded to 256).

The live raw-text/API-log path (core/input_adapter.py) is unaffected --
it keeps hashing real string tokens from uploaded logs, which is the one
modality where literal API-name strings genuinely exist. That live path
approximates this dataset's structured features from unstructured text;
it was never a byte-for-byte match and isn't now either.

Usage: python scripts/build_graph_dataset.py [--input data/cleaned_data.csv] [--output data/pyg_dataset.pt]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph_constructor import (
    D_FEATURE, EDGE_INDEX, NODE_HEADER, NODE_ENTROPY, NODE_API, NODE_NETWORK,
    NATIVE_PE_HEADER_COLS, NATIVE_ENTROPY_COLS,
    get_one_hot_type, native_row_to_node_vectors,
)
from torch_geometric.data import Data


def create_sample_graph(row: pd.Series, label: int = None) -> Data:
    vectors = native_row_to_node_vectors(row.get, D_FEATURE)
    final = [
        np.concatenate([vectors[node_idx], get_one_hot_type(node_idx)])
        for node_idx in (NODE_HEADER, NODE_ENTROPY, NODE_API, NODE_NETWORK)
    ]
    X = torch.tensor(np.stack(final), dtype=torch.float)
    y = torch.tensor([label], dtype=torch.long) if label is not None else None
    return Data(x=X, edge_index=EDGE_INDEX, y=y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", default="data/cleaned_data.csv")
    parser.add_argument("--output", "-o", default="data/pyg_dataset.pt")
    args = parser.parse_args()

    print(f"[*] Loading {args.input} ...")
    df = pd.read_csv(args.input, low_memory=False)

    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in the input CSV.")

    class_names = sorted(df["label"].dropna().unique().tolist())
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}
    print(f"[+] {len(class_names)} classes (sorted): {class_names}")

    missing = [c for c in NATIVE_PE_HEADER_COLS + NATIVE_ENTROPY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected numeric columns: {missing}")

    graphs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building graphs"):
        label_idx = label_to_idx.get(row["label"])
        g = create_sample_graph(row, label=label_idx)
        graphs.append(g)

    print(f"[+] Built {len(graphs)} graphs. Feature dim per node: {graphs[0].x.shape[1]}")
    torch.save(graphs, args.output)
    print(f"[+] Saved to {args.output}")

    labels_path = os.path.join(os.path.dirname(args.output) or ".", "class_labels.json")
    import json
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"[+] Saved class label order to {labels_path}")


if __name__ == "__main__":
    main()
