"""
==============================================================================
  Malware PE/APK Dataset → Static Feature Extraction & Graph Construction
==============================================================================
"""

import os
import re
import json
import hashlib
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_SLOTS = 16          # _0 … _15 per feature group
ENTROPY_HIGH  = 7.0         # threshold → packed / encrypted section
ENTROPY_MED   = 5.0         # threshold → compressed section

# PE header numeric columns (single value each)
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

SECTION_COLS = {
    "nb":           "f_SectionsNb_0",
    "mean_entropy": "f_SectionsMeanEntropy_0",
    "min_entropy":  "f_SectionsMinEntropy_0",
    "max_entropy":  "f_SectionsMaxEntropy_0",
    "mean_raw":     "f_SectionsMeanRawsize_0",
    "min_raw":      "f_SectionsMinRawsize_0",
    "max_raw":      "f_SectionsMaxRawsize_0",
    "mean_virt":    "f_SectionsMeanVirtualsize_0",
    "min_virt":     "f_SectionsMinVirtualsize_0",
    "max_virt":     "f_SectionMaxVirtualsize_0",
}

IMPORT_CATS  = ["open","close","create","resume","kill","call","delete","other"]
EXPORT_CATS  = ["open","close","create","resume","kill","call","delete","other"]

# API co-occurrence threshold (Phase 1: conservative; Phase 2: tune higher)
CO_OCCURRENCE_THRESHOLD = 2  # Only create SHARES_API edge if co-appear in 2+ samples

STRING_GROUPS = {
    "URL":            "f_URLs",
    "DIR":            "f_DIRs",
    "Email":          "f_emails",
    "InvEmail":       "f_inValEmails",
    "LongWord":       "f_longWord",
    "Keyword":        "f_specialKeyword",
    "IP":             "f_ipaddresses",
    "Sentence":       "f_sentences",
    "Filename":       "f_fileName",
    "Garbage":        "f_garbage",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING  (supports CSV / Excel / JSON)
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from CSV, Excel, or JSON."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path, low_memory=False)
    elif ext in (".xls", ".xlsx"):
        df = pd.read_excel(path)
    elif ext == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    print(f"[+] Loaded  {df.shape[0]:,} rows × {df.shape[1]:,} cols  from '{path}'")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Drop fully-empty rows / columns
    2. Fill numeric NaNs with 0
    3. Fill string NaNs with ''
    4. Coerce numeric columns
    5. Remove duplicate rows
    6. Report cleaning summary
    """
    original_shape = df.shape

    # Drop columns that are 100 % NaN
    df.dropna(axis=1, how="all", inplace=True)

    # Drop rows that are 100 % NaN
    df.dropna(axis=0, how="all", inplace=True)

    # Identify numeric vs string columns
    num_cols = [c for c in df.columns if _is_numeric_col(c, df)]
    str_cols  = [c for c in df.columns if c not in num_cols]

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    for c in str_cols:
        df[c] = df[c].fillna("").astype(str).str.strip()

    # Remove exact duplicates
    before_dedup = len(df)
    df.drop_duplicates(inplace=True)
    after_dedup  = len(df)

    print(f"[✓] Cleaning summary")
    print(f"    Original : {original_shape[0]:,} rows × {original_shape[1]:,} cols")
    print(f"    After    : {df.shape[0]:,} rows × {df.shape[1]:,} cols")
    print(f"    Dupes removed : {before_dedup - after_dedup:,}")

    return df.reset_index(drop=True)


def _is_numeric_col(col: str, df: pd.DataFrame) -> bool:
    """Heuristic: column name starts with 'f_' and content looks numeric."""
    if not col.startswith("f_"):
        return False
    sample = df[col].dropna().head(50)
    try:
        pd.to_numeric(sample, errors="raise")
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 3.  STATIC FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(row: pd.Series) -> dict:
    """Extract structured features from one sample row."""

    features = {}

    # ── PE header ──────────────────────────────────────────────────────────
    features["pe_header"] = {
        col: float(row.get(col, 0))
        for col in PE_HEADER_COLS
        if col in row.index
    }

    # ── Section stats ───────────────────────────────────────────────────────
    features["sections"] = {
        k: float(row.get(v, 0))
        for k, v in SECTION_COLS.items()
        if v in row.index
    }
    # Classify section entropy profile
    max_ent = features["sections"].get("max_entropy", 0)
    if max_ent >= ENTROPY_HIGH:
        features["sections"]["profile"] = "packed"
    elif max_ent >= ENTROPY_MED:
        features["sections"]["profile"] = "compressed"
    else:
        features["sections"]["profile"] = "normal"

    # ── Imports ──────────────────────────────────────────────────────────────
    imports = defaultdict(list)
    for cat in IMPORT_CATS:
        for i in range(FEATURE_SLOTS):
            val = str(row.get(f"f_ImportsList_{cat}_{i}", "")).strip()
            if val and val != "0" and val.lower() != "nan":
                imports[cat].append(val)
    features["imports"] = dict(imports)

    # ── Exports ──────────────────────────────────────────────────────────────
    exports = defaultdict(list)
    for cat in EXPORT_CATS:
        for i in range(FEATURE_SLOTS):
            val = str(row.get(f"f_ExportsList_{cat}_{i}", "")).strip()
            if val and val != "0" and val.lower() != "nan":
                exports[cat].append(val)
    features["exports"] = dict(exports)

    # ── String artefacts ─────────────────────────────────────────────────────
    strings = defaultdict(list)
    for label, prefix in STRING_GROUPS.items():
        for i in range(FEATURE_SLOTS):
            val = str(row.get(f"{prefix}_{i}", "")).strip()
            if val and val != "0" and val.lower() != "nan":
                strings[label].append(val)
    features["strings"] = dict(strings)

    return features


# ─────────────────────────────────────────────────────────────────────────────
# 4.  GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

NODE_COLORS = {
    "SAMPLE":     "#FF4757",
    "API_CALL":   "#2ED573",
    "STRING":     "#FFA502",
    "SECTION":    "#1E90FF",
    "PE_HEADER":  "#A855F7",
}


def build_graph(df: pd.DataFrame, max_samples: int = None) -> nx.DiGraph:
    """Convert cleaned DataFrame into a directed property graph."""
    G = nx.DiGraph()
    subset = df.iloc[:max_samples] if max_samples else df

    # Track API co-occurrence per sample for SHARES_API edges
    sample_apis: dict[str, set] = {}

    for idx, row in subset.iterrows():
        sample_id = f"sample::{idx}"
        feats     = extract_features(row)

        # ── SAMPLE node ──────────────────────────────────────────────────
        G.add_node(
            sample_id,
            node_type   = "SAMPLE",
            label       = f"S{idx}",
            color       = NODE_COLORS["SAMPLE"],
            **{k: v for k, v in feats["pe_header"].items()},
            section_nb      = feats["sections"].get("nb", 0),
            section_profile = feats["sections"].get("profile", "normal"),
            mean_entropy    = feats["sections"].get("mean_entropy", 0),
            max_entropy     = feats["sections"].get("max_entropy", 0),
        )

        apis_this_sample: set[str] = set()

        # ── IMPORT edges ──────────────────────────────────────────────────
        for cat, calls in feats["imports"].items():
            for api in calls:
                api_node = f"api::{api}"
                if not G.has_node(api_node):
                    G.add_node(
                        api_node,
                        node_type = "API_CALL",
                        label     = api,
                        color     = NODE_COLORS["API_CALL"],
                        category  = cat,
                        direction = "import",
                    )
                else:
                    # API node exists - update direction if needed
                    existing_dir = G.nodes[api_node].get("direction", "")
                    if existing_dir == "export":
                        G.nodes[api_node]["direction"] = "both"
                w = G[sample_id][api_node]["weight"] + 1 \
                    if G.has_edge(sample_id, api_node) else 1
                G.add_edge(sample_id, api_node,
                           edge_type="HAS_IMPORT", weight=w, category=cat)
                apis_this_sample.add(api_node)

        # ── EXPORT edges ──────────────────────────────────────────────────
        for cat, calls in feats["exports"].items():
            for api in calls:
                api_node = f"api::{api}"
                if not G.has_node(api_node):
                    G.add_node(
                        api_node,
                        node_type = "API_CALL",
                        label     = api,
                        color     = NODE_COLORS["API_CALL"],
                        category  = cat,
                        direction = "export",
                    )
                else:
                    # API node exists - update direction if needed
                    existing_dir = G.nodes[api_node].get("direction", "")
                    if existing_dir == "import":
                        G.nodes[api_node]["direction"] = "both"
                w = G[sample_id][api_node]["weight"] + 1 \
                    if G.has_edge(sample_id, api_node) else 1
                G.add_edge(sample_id, api_node,
                           edge_type="HAS_EXPORT", weight=w, category=cat)
                apis_this_sample.add(api_node)

        # ── STRING edges ──────────────────────────────────────────────────
        for stype, values in feats["strings"].items():
            for val in values:
                key = val if len(val) <= 60 else hashlib.md5(val.encode()).hexdigest()
                str_node = f"str::{stype}::{key}"
                if not G.has_node(str_node):
                    G.add_node(
                        str_node,
                        node_type   = "STRING",
                        label       = f"{stype}",
                        color       = NODE_COLORS["STRING"],
                        string_type = stype,
                        value       = val[:120],
                    )
                G.add_edge(sample_id, str_node,
                           edge_type="HAS_STRING", string_type=stype)

        # ── SECTION node ──────────────────────────────────────────────────
        profile    = feats["sections"].get("profile", "normal")
        sec_node   = f"sec::{profile}"
        if not G.has_node(sec_node):
            G.add_node(
                sec_node,
                node_type = "SECTION",
                label     = profile.upper(),
                color     = NODE_COLORS["SECTION"],
                profile   = profile,
            )
        G.add_edge(sample_id, sec_node,
                   edge_type    = "HAS_SECTION",
                   max_entropy  = feats["sections"].get("max_entropy", 0))

        sample_apis[sample_id] = apis_this_sample

    # ── SHARES_API edges (co-occurrence) ──────────────────────────────────
    api_to_samples: dict[str, set] = defaultdict(set)
    for sid, apis in sample_apis.items():
        for a in apis:
            api_to_samples[a].add(sid)

    for api_node, samples in api_to_samples.items():
        if len(samples) < CO_OCCURRENCE_THRESHOLD:
            continue
        peer_counts: dict[str, int] = defaultdict(int)
        for sid in samples:
            for peer in sample_apis[sid]:
                if peer != api_node:
                    peer_counts[peer] += 1
        # Filter by threshold instead of arbitrary top-10
        valid_peers = [p for p, cnt in peer_counts.items() if cnt >= CO_OCCURRENCE_THRESHOLD]
        for peer in valid_peers:
            if not G.has_edge(api_node, peer):
                G.add_edge(api_node, peer,
                           edge_type="SHARES_API",
                           co_count=peer_counts[peer])

    return G


# ─────────────────────────────────────────────────────────────────────────────
# 5.  GRAPH STATISTICS & EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def graph_stats(G: nx.DiGraph):
    """Print graph statistics."""
    node_types = defaultdict(int)
    edge_types = defaultdict(int)
    for _, d in G.nodes(data=True):
        node_types[d.get("node_type", "?")] += 1
    for _, _, d in G.edges(data=True):
        edge_types[d.get("edge_type", "?")] += 1

    print("\n" + "═"*55)
    print("  GRAPH STATISTICS")
    print("═"*55)
    print(f"  Total nodes : {G.number_of_nodes():,}")
    print(f"  Total edges : {G.number_of_edges():,}")
    print()
    print("  NODE BREAKDOWN")
    for nt, cnt in sorted(node_types.items()):
        bar = "█" * min(cnt, 40)
        print(f"    {nt:<14} {cnt:>6,}  {bar}")
    print()
    print("  EDGE BREAKDOWN")
    for et, cnt in sorted(edge_types.items()):
        bar = "█" * min(cnt, 40)
        print(f"    {et:<16} {cnt:>6,}  {bar}")
    print("═"*55 + "\n")


def export_graph_json(G: nx.DiGraph, path: str):
    data = nx.node_link_data(G)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[✓] Graph JSON saved → '{path}'")


def export_edgelist(G: nx.DiGraph, path: str):
    rows = []
    for u, v, d in G.edges(data=True):
        rows.append({"source": u, "target": v, **d})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[✓] Edge list  saved → '{path}'")


def export_nodelist(G: nx.DiGraph, path: str):
    rows = []
    for n, d in G.nodes(data=True):
        rows.append({"node_id": n, **d})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[✓] Node list  saved → '{path}'")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  GRAPH VISUALISATION (matplotlib fallback)
# ─────────────────────────────────────────────────────────────────────────────

def visualize_graph(G: nx.DiGraph, max_nodes: int = 120, out_path: str = None):
    nodes = list(G.nodes())[:max_nodes]
    H     = G.subgraph(nodes)

    colors  = [H.nodes[n].get("color", "#888888") for n in H.nodes()]
    sizes   = []
    for n in H.nodes():
        nt = H.nodes[n].get("node_type", "")
        sizes.append(600 if nt == "SAMPLE" else 300 if nt == "API_CALL" else 150)

    fig, ax = plt.subplots(figsize=(18, 12), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")

    pos = nx.spring_layout(H, seed=42, k=1.8)
    nx.draw_networkx_nodes(H, pos, node_color=colors, node_size=sizes, alpha=0.92, ax=ax)
    nx.draw_networkx_edges(H, pos, edge_color="#444C56", arrows=True, arrowsize=10, width=0.6, alpha=0.6, ax=ax)
    nx.draw_networkx_labels(H, pos, labels={n: H.nodes[n].get("label", n.split("::")[-1])[:12] for n in H.nodes()}, font_size=6, font_color="#E6EDF3", ax=ax)

    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in NODE_COLORS.items()]
    ax.legend(handles=legend_patches, loc="upper left", facecolor="#161B22", edgecolor="#30363D", labelcolor="#E6EDF3", fontsize=9)
    ax.set_title("Malware PE Static Feature Graph", color="#E6EDF3", fontsize=14, pad=12)
    ax.axis("off")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
        print(f"[✓] Graph PNG  saved → '{out_path}'")
    else:
        try:
            plt.show()
        except:
            pass
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  DEMO  ─  runs when dataset is absent; builds synthetic samples
# ─────────────────────────────────────────────────────────────────────────────

def _build_demo_dataframe(n: int = 30) -> pd.DataFrame:
    rng  = np.random.default_rng(0)
    rows = []
    apis_pool = {
        "open":   ["OpenFile","OpenRegistry","OpenProcess","OpenMutex"],
        "close":  ["CloseHandle","CloseSocket"],
        "create": ["CreateFile","CreateProcess","CreateThread","CreateMutex"],
        "kill":   ["TerminateProcess","KillTimer"],
        "call":   ["VirtualAlloc","WriteProcessMemory","LoadLibrary","GetProcAddress"],
        "delete": ["DeleteFile","RegDeleteKey"],
        "resume": ["ResumeThread"],
        "other":  ["GetSystemInfo","Sleep","GetTickCount"],
    }
    urls_pool    = ["http://evil.ru/gate.php","https://c2.malware.net/cmd"]
    ip_pool      = ["192.168.1.100","10.0.0.5"]
    kw_pool      = ["cmd.exe","powershell"]

    for i in range(n):
        row: dict = {}
        row.update({
            "f_Machine_0": 332, "f_SizeOfOptionalHeader_0": 224,
            "f_Characteristics_0": rng.integers(0, 0xFFFF),
            "f_SectionsNb_0": rng.integers(3, 8),
            "f_SectionsMeanEntropy_0": rng.uniform(4.5, 7.5),
        })
        for cat, pool in apis_pool.items():
            chosen = rng.choice(pool, size=rng.integers(0, len(pool)), replace=False)
            for j, api in enumerate(chosen[:16]):
                row[f"f_ImportsList_{cat}_{j}"] = api
        rows.append(row)

    return pd.DataFrame(rows).fillna("")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main(dataset_path: str = None, output_dir: str = "graph_output",
         max_samples: int = None):

    os.makedirs(output_dir, exist_ok=True)

    if dataset_path and os.path.exists(dataset_path):
        df_raw = load_dataset(dataset_path)
    else:
        print("[!] No dataset path provided — running with synthetic demo data (30 samples)")
        df_raw = _build_demo_dataframe(30)

    df_clean = clean_dataset(df_raw)
    print(f"\n[~] Building graph (max_samples={max_samples or 'all'}) …")
    G = build_graph(df_clean, max_samples=max_samples)
    graph_stats(G)
    
    export_graph_json(G, os.path.join(output_dir, "graph.json"))
    export_edgelist  (G, os.path.join(output_dir, "edges.csv"))
    export_nodelist  (G, os.path.join(output_dir, "nodes.csv"))

    visualize_graph(G, max_nodes=100, out_path=os.path.join(output_dir, "graph_preview.png"))
    print(f"\n[✓] All outputs saved to '{output_dir}/'")
    return G


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    main(dataset_path=path, output_dir="graph_output")
