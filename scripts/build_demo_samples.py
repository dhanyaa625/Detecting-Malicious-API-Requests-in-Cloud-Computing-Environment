"""
Builds a small, legitimate demo set for the live-analysis page.

Pulls real rows from data/cleaned_data.csv that fall in data/test_indices.npy
-- i.e. samples the model was never trained on -- one per class (up to
`per_class`), and writes each as a standalone CSV file (header + one row)
under demo_samples/. Each file can be uploaded directly to /live_analysis.html
with input_type=csv: UniversalInputAdapter.parse() reads exactly this schema.

Because ground truth is known and the split is provably held-out, this is a
legitimate way to demo the pipeline without fabricating input data.

Usage:
    python scripts/build_demo_samples.py [--per-class N] [--out DIR]
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=2, help="Samples to pull per class")
    parser.add_argument("--out", default="demo_samples", help="Output directory")
    args = parser.parse_args()

    csv_path = os.path.join("data", "cleaned_data.csv")
    test_index_path = os.path.join("data", "test_indices.npy")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found -- run from the project root.")
    if not os.path.exists(test_index_path):
        raise FileNotFoundError(
            f"{test_index_path} not found -- run agentic_pacx/gnn_classification.py "
            "at least once to generate the train/test split."
        )

    test_indices = set(int(i) for i in np.load(test_index_path))
    df = pd.read_csv(csv_path, low_memory=False)

    os.makedirs(args.out, exist_ok=True)

    manifest = []
    for label, group in df.groupby("label"):
        held_out = group[group.index.isin(test_indices)]
        picked = held_out.head(args.per_class)
        for row_idx, row in picked.iterrows():
            filename = f"{label}_{row_idx}.csv"
            out_path = os.path.join(args.out, filename)
            pd.DataFrame([row]).to_csv(out_path, index=False)
            manifest.append({"file": filename, "true_label": label, "csv_row_index": row_idx})

    manifest_path = os.path.join(args.out, "manifest.csv")
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)

    print(f"[+] Wrote {len(manifest)} held-out demo samples to {args.out}/")
    print(f"[+] Manifest (true labels) saved to {manifest_path}")
    print("[*] Upload any file to /live_analysis.html with input_type=csv and compare")
    print("    the prediction against the manifest's true_label -- these were never")
    print("    seen during training, so a correct prediction is a legitimate result.")


if __name__ == "__main__":
    main()
