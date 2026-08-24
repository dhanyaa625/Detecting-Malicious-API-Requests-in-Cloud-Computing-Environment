"""
==============================================================================
  Malware PE Dataset → Bidirectional Feature-Group Graph Construction
==============================================================================
This script implements Part 1 of the malware detection pipeline.
It generates a PyTorch Geometric (PyG) homogenous dataset from CSV.
For each sample, it creates a graph G_s = (V, E) where:
  -> V = 4 Nodes: [Header, Entropy, API, Network]
  -> E = Logical dependencies between the groups (bidirectional)
"""

import os
import hashlib
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import torch
    from torch_geometric.data import Data
except ImportError:
    print("[!] PyTorch or torch_geometric not found.")
    print("    Please install them to proceed: pip install torch torch-geometric")
    exit(1)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

D_FEATURE = 256 # Uniform feature dimension for all nodes
D_NODE_TYPE = 4 # Size of the one-hot encoded node type
TOTAL_DIM = D_FEATURE + D_NODE_TYPE # 260

# Define Node indices
NODE_HEADER  = 0
NODE_ENTROPY = 1
NODE_API     = 2
NODE_NETWORK = 3

# Define Edges: Header <-> Entropy, Header <-> API, API <-> Network, Entropy <-> API
edge_sources = [NODE_HEADER, NODE_ENTROPY, NODE_HEADER, NODE_API, NODE_API, NODE_NETWORK, NODE_ENTROPY, NODE_API]
edge_targets = [NODE_ENTROPY, NODE_HEADER, NODE_API, NODE_HEADER, NODE_NETWORK, NODE_API, NODE_API, NODE_ENTROPY]
EDGE_INDEX = torch.tensor([edge_sources, edge_targets], dtype=torch.long)

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN DEFINITIONS (From original script)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_SLOTS = 16

PE_HEADER_COLS = [
    "f_Machine_0","f_SizeOfOptionalHeader_0","f_Characteristics_0",
    "f_MajorLinkerVersion_0","f_MinorLinkerVersion_0","f_SizeOfCode_0",
    "f_SizeOfInitializedData_0","f_SizeOfUninitializedData_0",
    "f_AddressOfEntryPoint_0","f_BaseOfCode_0","f_BaseOfData_0",
    "f_ImageBase_0","f_SectionAlignment_0","f_FileAlignment_0",
    "f_MajorOperatingSystemVersion_0","f_MinorOperatingSystemVersion_0",
    "f_MajorImageVersion_0","f_MinorImageVersion_0",
    "f_MajorSubsystemVersion_0","f_MinorSubsystemVersion_0",
    "f_SizeOfImage_0","f_SizeOfHeaders_0","f_CheckSum_0",
    "f_Subsystem_0","f_DllCharacteristics_0",
    "f_SizeOfStackReserve_0","f_SizeOfStackCommit_0",
    "f_SizeOfHeapReserve_0","f_SizeOfHeapCommit_0",
    "f_LoaderFlags_0","f_NumberOfRvaAndSizes_0",
]

SECTION_COLS = [
    "f_SectionsNb_0", "f_SectionsMeanEntropy_0", "f_SectionsMinEntropy_0", 
    "f_SectionsMaxEntropy_0", "f_SectionsMeanRawsize_0", "f_SectionsMinRawsize_0", 
    "f_SectionsMaxRawsize_0", "f_SectionsMeanVirtualsize_0", "f_SectionsMinVirtualsize_0", 
    "f_SectionMaxVirtualsize_0"
]

IMPORT_CATS  = ["open","close","create","resume","kill","call","delete","other"]
EXPORT_CATS  = ["open","close","create","resume","kill","call","delete","other"]

STRING_GROUPS = {
    "URL": "f_URLs", "DIR": "f_DIRs", "Email": "f_emails", "InvEmail": "f_inValEmails",
    "LongWord": "f_longWord", "Keyword": "f_specialKeyword", "IP": "f_ipaddresses",
    "Sentence": "f_sentences", "Filename": "f_fileName", "Garbage": "f_garbage",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_one_hot_type(node_type: int) -> list:
    """Returns a 4-element one-hot list."""
    vec = [0.0] * D_NODE_TYPE
    vec[node_type] = 1.0
    return vec

def hash_feature_vector(string_list: list, dim: int = D_FEATURE) -> np.ndarray:
    """
    Simulates API2Vec via a fixed-length hashing frequency method.
    Hashes string tokens into 'dim' buckets.
    """
    vec = np.zeros(dim, dtype=np.float32)
    for s in string_list:
        if isinstance(s, str) and s.strip():
            idx = int(hashlib.md5(s.strip().encode()).hexdigest(), 16) % dim
            vec[idx] += 1.0
    
    # Normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

def pad_numeric_features(vals: list, dim: int = D_FEATURE) -> np.ndarray:
    """
    Pads or truncates a list of numerical features to the specified dimension.
    """
    vec = np.zeros(dim, dtype=np.float32)
    length = min(len(vals), dim)
    vec[:length] = vals[:length]
    # Simple MinMax-like normalization if std > 0 to prevent huge variances
    # (assuming raw features from CSV can be huge)
    std = np.std(vec)
    if std > 0:
        vec = (vec - np.mean(vec)) / std
    return vec

# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHMS
# ─────────────────────────────────────────────────────────────────────────────

def create_sample_graph(row: pd.Series, label: int = None) -> Data:
    """
    Executes Algorithm 1, 2, and 3 for a single sample row.
    """
    # ── Algorithm 3: Header Node ──
    header_vals = [float(row.get(col, 0.0) if pd.notna(row.get(col)) else 0.0) for col in PE_HEADER_COLS]
    header_vec = pad_numeric_features(header_vals)
    header_final = np.concatenate([header_vec, get_one_hot_type(NODE_HEADER)])

    # ── Algorithm 3: Entropy Node ──
    entropy_vals = [float(row.get(col, 0.0) if pd.notna(row.get(col)) else 0.0) for col in SECTION_COLS]
    entropy_vec = pad_numeric_features(entropy_vals)
    entropy_final = np.concatenate([entropy_vec, get_one_hot_type(NODE_ENTROPY)])

    # ── Algorithm 2: API Node ──
    api_strings = []
    for cat in IMPORT_CATS:
        for i in range(FEATURE_SLOTS):
            val = str(row.get(f"f_ImportsList_{cat}_{i}", "")).strip()
            if val and val != "0" and val.lower() != "nan":
                api_strings.append(val)
    for cat in EXPORT_CATS:
        for i in range(FEATURE_SLOTS):
            val = str(row.get(f"f_ExportsList_{cat}_{i}", "")).strip()
            if val and val != "0" and val.lower() != "nan":
                api_strings.append(val)
    
    api_vec = hash_feature_vector(api_strings)
    api_final = np.concatenate([api_vec, get_one_hot_type(NODE_API)])

    # ── Algorithm 3: Network Node ──
    net_strings = []
    for prefix in STRING_GROUPS.values():
        for i in range(FEATURE_SLOTS):
            val = str(row.get(f"{prefix}_{i}", "")).strip()
            if val and val != "0" and val.lower() != "nan":
                net_strings.append(val)

    net_vec = hash_feature_vector(net_strings)
    net_final = np.concatenate([net_vec, get_one_hot_type(NODE_NETWORK)])

    # Combine into node feature matrix X
    X = torch.tensor([header_final, entropy_final, api_final, net_final], dtype=torch.float)

    # Optional Label
    y = torch.tensor([label], dtype=torch.long) if label is not None else None

    # Construct PyG Data object
    data = Data(x=X, edge_index=EDGE_INDEX, y=y)
    return data

# ─────────────────────────────────────────────────────────────────────────────
# EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def process_dataset(csv_path: str, out_path: str, max_samples: int = None):
    print(f"[*] Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Simple check for label column (often named "label", "malware", "class")
    label_col = None
    for col in ["label", "class", "malware", "is_malware"]:
        if col in df.columns:
            label_col = col
            break
            
    if max_samples:
        df = df.iloc[:max_samples]
        
    print(f"[*] Generating graphs for {len(df)} samples...")
    graphs = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Constructing Graphs"):
        # Retrieve label if available
        y = None
        if label_col:
            val = row[label_col]
            try:
                y = int(float(val))
            except ValueError:
                y = 0
            
        g = create_sample_graph(row, label=y)
        graphs.append(g)
        
    print(f"\n[+] Processing complete.")
    print(f"    - Generated {len(graphs)} graphs.")
    print(f"    - Nodes per graph: {graphs[0].num_nodes}")
    print(f"    - Edges per graph: {graphs[0].num_edges}")
    print(f"    - Feature dimension: {graphs[0].x.shape[1]}")
    
    print(f"\n[*] Saving dataset to {out_path}...")
    torch.save(graphs, out_path)
    print(f"[✓] Successfully saved to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate PyG Dataset from Malware CSV")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input CSV dataset")
    parser.add_argument("--output", "-o", type=str, default="data/pyg_dataset.pt", help="Path to output .pt file")
    parser.add_argument("--max_samples", "-m", type=int, default=None, help="Max rows to process (for testing)")
    
    args = parser.parse_args()
    process_dataset(args.input, args.output, args.max_samples)
