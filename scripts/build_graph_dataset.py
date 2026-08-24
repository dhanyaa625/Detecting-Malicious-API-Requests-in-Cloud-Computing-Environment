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
    get_one_hot_type, pad_numeric_features,
)
from torch_geometric.data import Data

FEATURE_SLOTS = 16

PE_HEADER_COLS = [
    "f_Machine_0", "f_SizeOfOptionalHeader_0", "f_Characteristics_0",
    "f_MajorLinkerVersion_0", "f_MinorLinkerVersion_0", "f_SizeOfCode_0",
    "f_SizeOfInitializedData_0", "f_SizeOfUninitializedData_0",
    "f_AddressOfEntryPoint_0", "f_BaseOfCode_0", "f_BaseOfData_0",
    "f_ImageBase_0", "f_SectionAlignment_0", "f_FileAlignment_0",
    "f_MajorOperatingSystemVersion_0", "f_MinorOperatingSystemVersion_0",
    "f_MajorImageVersion_0", "f_MinorImageVersion_0",
    "f_MajorSubsystemVersion_0", "f_MinorSubsystemVersion_0",
    "f_SizeOfImage_0", "f_SizeOfHeaders_0", "f_CheckSum_0",
    "f_Subsystem_0", "f_DllCharacteristics_0",
    "f_SizeOfStackReserve_0", "f_SizeOfStackCommit_0",
    "f_SizeOfHeapReserve_0", "f_SizeOfHeapCommit_0",
    "f_LoaderFlags_0", "f_NumberOfRvaAndSizes_0",
]

ENTROPY_COLS = [
    "f_SectionsNb_0", "f_SectionsMeanEntropy_0", "f_SectionsMinEntropy_0",
    "f_SectionsMaxEntropy_0", "f_SectionsMeanRawsize_0", "f_SectionsMinRawsize_0",
    "f_SectionsMaxRawsize_0", "f_SectionsMeanVirtualsize_0", "f_SectionsMinVirtualsize_0",
    "f_SectionMaxVirtualsize_0",
    # Structural summary scalars the old pipeline computed but never fed to any node.
    "f_ImportsNbDLL_0", "f_ImportsNb_0", "f_ImportsNbOrdinal_0", "f_ExportNb_0",
    "f_ResourcesNb_0", "f_ResourcesMeanEntropy_0", "f_ResourcesMinEntropy_0",
    "f_ResourcesMaxEntropy_0", "f_ResourcesMeanSize_0", "f_ResourcesMinSize_0",
    "f_ResourcesMaxSize_0", "f_LoadConfigurationSize_0", "f_VersionInformationSize_0",
]

IMPORT_CATS = ["open", "close", "create", "resume", "kill", "call", "delete", "other"]
EXPORT_CATS = ["open", "close", "create", "resume", "kill", "call", "delete", "other"]

STRING_GROUPS = {
    "URL": "f_URLs", "DIR": "f_DIRs", "Email": "f_emails", "InvEmail": "f_inValEmails",
    "LongWord": "f_longWord", "Keyword": "f_specialKeyword", "IP": "f_ipaddresses",
    "Sentence": "f_sentences", "Filename": "f_fileName", "Garbage": "f_garbage",
}


def _safe_float(row, col):
    val = row.get(col, 0.0)
    try:
        val = float(val)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if pd.isna(val) else val


def create_sample_graph(row: pd.Series, label: int = None) -> Data:
    header_vals = [_safe_float(row, c) for c in PE_HEADER_COLS]
    header_vec = pad_numeric_features(header_vals, D_FEATURE)
    header_final = np.concatenate([header_vec, get_one_hot_type(NODE_HEADER)])

    entropy_vals = [_safe_float(row, c) for c in ENTROPY_COLS]
    entropy_vec = pad_numeric_features(entropy_vals, D_FEATURE)
    entropy_final = np.concatenate([entropy_vec, get_one_hot_type(NODE_ENTROPY)])

    api_vals = []
    for cat in IMPORT_CATS:
        for i in range(FEATURE_SLOTS):
            api_vals.append(_safe_float(row, f"f_ImportsList_{cat}_{i}"))
    for cat in EXPORT_CATS:
        for i in range(FEATURE_SLOTS):
            api_vals.append(_safe_float(row, f"f_ExportsList_{cat}_{i}"))
    api_vec = pad_numeric_features(api_vals, D_FEATURE)
    api_final = np.concatenate([api_vec, get_one_hot_type(NODE_API)])

    net_vals = []
    for prefix in STRING_GROUPS.values():
        for i in range(FEATURE_SLOTS):
            net_vals.append(_safe_float(row, f"{prefix}_{i}"))
    net_vec = pad_numeric_features(net_vals, D_FEATURE)
    net_final = np.concatenate([net_vec, get_one_hot_type(NODE_NETWORK)])

    X = torch.tensor(
        np.stack([header_final, entropy_final, api_final, net_final]), dtype=torch.float
    )
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

    missing = [c for c in PE_HEADER_COLS + ENTROPY_COLS if c not in df.columns]
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
