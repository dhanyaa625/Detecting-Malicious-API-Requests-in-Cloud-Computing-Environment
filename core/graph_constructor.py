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
# NATIVE DATASET SCHEMA (data/cleaned_data.csv column groups)
# ─────────────────────────────────────────────────────────────────────────────
# Shared by scripts/build_graph_dataset.py (offline training-set construction)
# and core/input_adapter.py (live inference). Kept in one place so a CSV
# upload of one of this dataset's own rows gets reconstructed into the exact
# same features the model was trained on, instead of falling through to the
# generic text-heuristic path meant for unstructured logs.

NATIVE_FEATURE_SLOTS = 16

NATIVE_PE_HEADER_COLS = [
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

NATIVE_ENTROPY_COLS = [
    "f_SectionsNb_0", "f_SectionsMeanEntropy_0", "f_SectionsMinEntropy_0",
    "f_SectionsMaxEntropy_0", "f_SectionsMeanRawsize_0", "f_SectionsMinRawsize_0",
    "f_SectionsMaxRawsize_0", "f_SectionsMeanVirtualsize_0", "f_SectionsMinVirtualsize_0",
    "f_SectionMaxVirtualsize_0",
    "f_ImportsNbDLL_0", "f_ImportsNb_0", "f_ImportsNbOrdinal_0", "f_ExportNb_0",
    "f_ResourcesNb_0", "f_ResourcesMeanEntropy_0", "f_ResourcesMinEntropy_0",
    "f_ResourcesMaxEntropy_0", "f_ResourcesMeanSize_0", "f_ResourcesMinSize_0",
    "f_ResourcesMaxSize_0", "f_LoadConfigurationSize_0", "f_VersionInformationSize_0",
]

NATIVE_IMPORT_CATS = ["open", "close", "create", "resume", "kill", "call", "delete", "other"]
NATIVE_EXPORT_CATS = ["open", "close", "create", "resume", "kill", "call", "delete", "other"]

NATIVE_STRING_GROUPS = {
    "URL": "f_URLs", "DIR": "f_DIRs", "Email": "f_emails", "InvEmail": "f_inValEmails",
    "LongWord": "f_longWord", "Keyword": "f_specialKeyword", "IP": "f_ipaddresses",
    "Sentence": "f_sentences", "Filename": "f_fileName", "Garbage": "f_garbage",
}

NATIVE_API_COLS = [
    f"f_ImportsList_{cat}_{i}" for cat in NATIVE_IMPORT_CATS for i in range(NATIVE_FEATURE_SLOTS)
] + [
    f"f_ExportsList_{cat}_{i}" for cat in NATIVE_EXPORT_CATS for i in range(NATIVE_FEATURE_SLOTS)
]

NATIVE_NETWORK_COLS = [
    f"{prefix}_{i}" for prefix in NATIVE_STRING_GROUPS.values() for i in range(NATIVE_FEATURE_SLOTS)
]

NATIVE_SCHEMA_COLUMNS = set(NATIVE_PE_HEADER_COLS) | set(NATIVE_ENTROPY_COLS) | set(NATIVE_API_COLS) | set(NATIVE_NETWORK_COLS)


# ─────────────────────────────────────────────────────────────────────────────
# AMAURICIO/KAGGLE PE-HEADER SCHEMA (e.g. kaggle_dataset/data.csv, the
# "Benign & Malicious PE Files" Kaggle dataset) -- a plain-name variant of
# this project's own header+entropy columns (no "f_" prefix, no "_0"
# suffix), with one confirmed naming difference: their "SectionMaxRawsize"
# is our "f_SectionsMaxRawsize_0" (plural "Sections"). Verified by directly
# diffing kaggle_dataset/data.csv's real header against data/features.txt --
# 53 of 54 columns match exactly.
#
# This dataset has no import/export/string data at all, so uploading a raw
# row only ever fills the Header+Entropy nodes -- API+Network stay honestly
# unfilled (completeness < 1.0), same treatment as a live raw-PE-binary
# upload whose API/Network nodes get OOD-flagged out.
# ─────────────────────────────────────────────────────────────────────────────
_AMAURICIO_RENAME_OVERRIDES = {"SectionMaxRawsize": "SectionsMaxRawsize"}

AMAURICIO_COLUMN_MAP = {}  # their column name -> our NATIVE_* column name
for _our_col in NATIVE_PE_HEADER_COLS + NATIVE_ENTROPY_COLS:
    _their_col = _our_col.replace("f_", "").rsplit("_0", 1)[0]
    _their_col = _AMAURICIO_RENAME_OVERRIDES.get(_their_col, _their_col)
    AMAURICIO_COLUMN_MAP[_their_col] = _our_col


def is_amauricio_schema(columns) -> bool:
    """True if `columns` looks like the amauricio/Kaggle plain-name PE-header
    schema (e.g. kaggle_dataset/data.csv) rather than this project's own
    native "f_..._0" schema. Same threshold logic as is_native_schema: most
    of the header columns present, small tolerance for minor version drift.
    """
    cols = set(columns)
    their_header_cols = set(AMAURICIO_COLUMN_MAP.keys()) - {"legitimate", "Name", "md5"}
    return len(their_header_cols & cols) >= len(their_header_cols) - 4


def amauricio_row_to_native_dict(row_get) -> dict:
    """Maps a single amauricio-schema row into this project's own column
    names, ready for native_row_to_node_vectors(). `row_get` is a
    (column_name, default) -> value callable, same contract as
    native_row_to_node_vectors's row_get. Only Header+Entropy columns are
    populated -- API+Network are simply absent (this schema has no such
    data), which native_row_to_node_vectors already handles via row_get's
    default value.
    """
    return {
        our_col: row_get(their_col, 0.0)
        for their_col, our_col in AMAURICIO_COLUMN_MAP.items()
    }


def is_native_schema(columns) -> bool:
    """True if `columns` looks like this dataset's own native CSV schema
    (data/cleaned_data.csv), not an arbitrary user log. Requires most of the
    PE header columns to be present -- those are the smallest, most specific
    group, so a coincidental partial match on API/network columns alone
    (which are only generic "_0".."_15" suffixes) can't trigger a false
    positive.
    """
    cols = set(columns)
    return len(set(NATIVE_PE_HEADER_COLS) & cols) >= len(NATIVE_PE_HEADER_COLS) - 2


def _safe_float(row_get, col):
    try:
        val = float(row_get(col, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if val != val else val  # NaN check without importing pandas/numpy isnan


def native_row_to_node_vectors(row_get, dim: int = D_FEATURE) -> dict:
    """
    Reconstructs the exact 4 raw (pre-one-hot) node feature vectors that
    scripts/build_graph_dataset.py builds at training time, from a single
    native-schema row. `row_get` is a `(column_name, default) -> value`
    callable (a pandas Series' `.get`, or a plain dict's `.get`).
    """
    header_vals = [_safe_float(row_get, c) for c in NATIVE_PE_HEADER_COLS]
    entropy_vals = [_safe_float(row_get, c) for c in NATIVE_ENTROPY_COLS]
    api_vals = [_safe_float(row_get, c) for c in NATIVE_API_COLS]
    network_vals = [_safe_float(row_get, c) for c in NATIVE_NETWORK_COLS]

    return {
        NODE_HEADER: pad_numeric_features(header_vals, dim),
        NODE_ENTROPY: pad_numeric_features(entropy_vals, dim),
        NODE_API: pad_numeric_features(api_vals, dim),
        NODE_NETWORK: pad_numeric_features(network_vals, dim),
    }

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
