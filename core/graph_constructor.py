import torch
import hashlib
import numpy as np
import pandas as pd
from torch_geometric.data import Data

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

# Define Edges: Bidirectional dependencies between the 4 logical groups
edge_sources = [NODE_HEADER, NODE_ENTROPY, NODE_HEADER, NODE_API, NODE_API, NODE_NETWORK, NODE_ENTROPY, NODE_API]
edge_targets = [NODE_ENTROPY, NODE_HEADER, NODE_API, NODE_HEADER, NODE_NETWORK, NODE_API, NODE_API, NODE_ENTROPY]
EDGE_INDEX = torch.tensor([edge_sources, edge_targets], dtype=torch.long)

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
    Hashes string tokens into 'dim' buckets.
    Used for API names and Network artefacts.
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
    """Pads or truncates numerical features to fixed dimension."""
    vec = np.zeros(dim, dtype=np.float32)
    length = min(len(vals), dim)
    vec[:length] = vals[:length]
    std = np.std(vec)
    if std > 0:
        vec = (vec - np.mean(vec)) / std
    return vec

# ─────────────────────────────────────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def create_sample_graph(row: pd.Series, label: int = None) -> Data:
    """Constructs the standard 4-node graph from a tabular row block."""
    # This is a simplified version of the full generator for real-time use
    # In a full-scale scenario, we'd use the exact column mapping from the training script.
    
    # Placeholder for actual feature extraction (real code would use PE_HEADER_COLS etc.)
    # For now, we return a zero-placeholder that the Adapter will fill with Ghost Nodes.
    header_vec = np.zeros(D_FEATURE, dtype=np.float32)
    header_final = np.concatenate([header_vec, get_one_hot_type(NODE_HEADER)])

    entropy_vec = np.zeros(D_FEATURE, dtype=np.float32)
    entropy_final = np.concatenate([entropy_vec, get_one_hot_type(NODE_ENTROPY)])

    api_vec = np.zeros(D_FEATURE, dtype=np.float32)
    api_final = np.concatenate([api_vec, get_one_hot_type(NODE_API)])

    net_vec = np.zeros(D_FEATURE, dtype=np.float32)
    net_final = np.concatenate([net_vec, get_one_hot_type(NODE_NETWORK)])

    X = torch.tensor([header_final, entropy_final, api_final, net_final], dtype=torch.float)
    y = torch.tensor([label], dtype=torch.long) if label is not None else None

    return Data(x=X, edge_index=EDGE_INDEX, y=y)
