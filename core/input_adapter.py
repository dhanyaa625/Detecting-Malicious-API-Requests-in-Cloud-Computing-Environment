import os
import re
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

# Safely import strict constraints from Core
try:
    from core.graph_constructor import (
        EDGE_INDEX, get_one_hot_type, hash_feature_vector,
        pad_numeric_features, D_FEATURE, NODE_HEADER, NODE_ENTROPY, NODE_API, NODE_NETWORK,
        is_native_schema, native_row_to_node_vectors,
    )
except ImportError:
    # Fallback to local import if needed
    from graph_constructor import (
        EDGE_INDEX, get_one_hot_type, hash_feature_vector,
        pad_numeric_features, D_FEATURE, NODE_HEADER, NODE_ENTROPY, NODE_API, NODE_NETWORK,
        is_native_schema, native_row_to_node_vectors,
    )

# Common real WinAPI names that end in 'a'/'w' but have no A/W ABI-suffixed
# sibling -- stripping the trailing letter would mangle these specifically.
# Not exhaustive; see the docstring on canonicalize_api().
_NON_ABI_SUFFIXED_NAMES = {
    "showwindow", "destroywindow", "updatewindow", "iswindow", "movewindow",
    "getdesktopwindow", "getforegroundwindow", "setforegroundwindow",
    "getactivewindow", "closewindow", "flushview", "flushviewoffilebuffer",
}


class UniversalInputAdapter:
    def __init__(self, raw_dataset_path='data/pyg_dataset.pt', norm_stats_path='data/norm_stats.pt'):
        """
        `norm_stats_path` points at the per-node mean/std saved by
        core/dataset_manager.py, computed from the TRAIN split only. This
        adapter used to re-derive stats by loading the whole (already
        normalized) dataset and computing mean/std over it again here --
        redundant, and it meant "the training stats" and "the stats this
        adapter uses" could silently drift apart. Loading the one file both
        sides agree on removes that risk. `raw_dataset_path` is kept only as
        a fallback for older setups that haven't regenerated norm_stats.pt yet.
        """
        self.raw_dataset_path = raw_dataset_path
        self.norm_stats_path = norm_stats_path
        self.norm_params = None

    def _load_normalization_params(self):
        if self.norm_params is not None:
            return self.norm_params

        if os.path.exists(self.norm_stats_path):
            stats = torch.load(self.norm_stats_path, weights_only=False)
            self.norm_params = {
                node_idx: (entry["mean"], entry["std"]) for node_idx, entry in stats.items()
            }
            return self.norm_params

        # Fallback: derive stats from the full dataset (pre-dataset_manager.py setups only).
        if not os.path.exists(self.raw_dataset_path):
            raise FileNotFoundError(
                f"[!] Neither {self.norm_stats_path} nor {self.raw_dataset_path} exist. "
                f"Run `python -m core.dataset_manager` to generate normalization stats."
            )

        dataset = torch.load(self.raw_dataset_path, weights_only=False)
        dim_features = D_FEATURE

        header_feats = torch.stack([g.x[0, :dim_features] for g in dataset])
        entropy_feats = torch.stack([g.x[1, :dim_features] for g in dataset])
        api_feats = torch.stack([g.x[2, :dim_features] for g in dataset])
        network_feats = torch.stack([g.x[3, :dim_features] for g in dataset])

        def get_params(tensor):
            mean = tensor.mean(dim=0, keepdim=True)
            std = tensor.std(dim=0, keepdim=True)
            std[std == 0] = 1.0
            return mean, std

        self.norm_params = {
            NODE_HEADER: get_params(header_feats),
            NODE_ENTROPY: get_params(entropy_feats),
            NODE_API: get_params(api_feats),
            NODE_NETWORK: get_params(network_feats)
        }
        return self.norm_params

    def _apply_normalization(self, node_tensors: list) -> list:
        params = self._load_normalization_params()

        normalized_tensors = []
        for i in range(4):
            # Scale only the features, ignore the identifying one-hot ending
            feat_block = node_tensors[i][:D_FEATURE].unsqueeze(0)
            mean, std = params[i]
            scaled = (feat_block - mean) / std

            # Repackage the one-hot tail end
            one_hot = node_tensors[i][-4:].unsqueeze(0)
            full_node = torch.cat([scaled, one_hot], dim=1).squeeze(0)
            normalized_tensors.append(full_node)

        return normalized_tensors

    def canonicalize_api(self, api_string: str) -> str:
        """
        MANDATORY ADAPTER OBFUSCATION PREVENTION.
        e.g., kernel32.dll!CreateFileW -> createfile
        e.g., NtCreateFile -> createfile

        Both stripping rules are heuristics and can still misfire on
        ordinary names (e.g. an all-caps "NTDLL" module string, or a real
        unsuffixed API that just happens to end in 'a'/'w'); the guards
        below only narrow the two concrete false-positive cases found
        during review (`ShowWindow` -> `showwindo`, `ntdll` -> `dll`),
        they don't eliminate the ambiguity entirely.
        """
        original = api_string.strip()
        api = original.lower()

        # Strip Library Declarations
        if "!" in api:
            original = original.split("!")[-1]
            api = api.split("!")[-1]

        # Strip Nt/Zw kernel wrappers -- only when the source used real
        # Hungarian-notation casing (NtXxx/ZwXxx: third character uppercase),
        # which is how genuine syscall wrappers are written in logs/tools.
        # A lowercase module name like "ntdll" doesn't match this and is
        # left alone.
        if len(original) > 2 and original[:2].lower() in ("nt", "zw") and original[2:3].isupper():
            api = api[2:]

        # Strip Windows ABI tags ('A', 'W', 'Ex') -- skip a small set of
        # common real API names that end in 'a'/'w' but were never
        # ABI-suffixed to begin with.
        if len(api) > 4 and api not in _NON_ABI_SUFFIXED_NAMES and (api.endswith('a') or api.endswith('w')):
            api = api[:-1]

        if len(api) > 4 and api.endswith('ex'):
            api = api[:-2]

        return api

    def _extract_tokens(self, raw_text: str) -> list:
        if not raw_text:
            return []
        return [tok for tok in re.split(r'[\s,;|\r\n\t]+', raw_text) if tok]

    def _build_text_heuristics(self, raw_text: str) -> tuple[dict, dict]:
        if not isinstance(raw_text, str):
            raw_text = str(raw_text or "")

        tokens = self._extract_tokens(raw_text)
        lowered = raw_text.lower()

        api_like_tokens = []
        network_tokens = []
        header_markers = []

        for token in tokens:
            cleaned = token.strip()
            lowered_token = cleaned.lower()
            if not cleaned:
                continue

            if re.search(r'(https?://|www\.|(?:\d{1,3}\.){3}\d{1,3}|[a-z0-9.-]+\.(com|net|org|ru|xyz|io))', lowered_token):
                network_tokens.append(lowered_token)

            if re.search(r'(dll!|^[a-z_][a-z0-9_]+(?:file|process|thread|alloc|load|exec|connect|socket|http|crypt|reg|inject|write|read))', lowered_token):
                api_like_tokens.append(self.canonicalize_api(cleaned))

        for marker in ["mz", "pe", ".text", ".rsrc", ".idata", ".reloc", "dos stub", "rich"]:
            if marker in lowered:
                header_markers.append(marker)

        suspicious_terms = [
            term for term in [
                "virtualalloc", "writeprocessmemory", "createremotethread",
                "powershell", "cmd.exe", "http://", "https://", "rundll32",
                "regsvr32", "connectnetwork", "winexec", "cryptdecrypt"
            ]
            if term in lowered
        ]

        char_count = max(len(raw_text), 1)
        unique_ratio = len(set(raw_text)) / char_count
        digit_ratio = sum(ch.isdigit() for ch in raw_text) / char_count
        upper_ratio = sum(ch.isupper() for ch in raw_text) / char_count
        symbol_ratio = sum(not ch.isalnum() and not ch.isspace() for ch in raw_text) / char_count

        entropy = 0.0
        for ch in set(raw_text):
            prob = raw_text.count(ch) / char_count
            entropy -= prob * np.log2(prob)

        entropy_features = pad_numeric_features([
            entropy,
            np.log1p(char_count),
            unique_ratio,
            digit_ratio,
            upper_ratio,
            symbol_ratio,
            len(api_like_tokens),
            len(network_tokens),
            lowered.count("http"),
            lowered.count("powershell"),
            lowered.count("cmd.exe"),
            lowered.count("virtualalloc"),
            lowered.count("createremotethread"),
            lowered.count("writeprocessmemory"),
        ], D_FEATURE)

        return (
            {
                NODE_HEADER: hash_feature_vector(header_markers, D_FEATURE) if header_markers else None,
                NODE_ENTROPY: entropy_features,
                NODE_API: hash_feature_vector(api_like_tokens, D_FEATURE) if api_like_tokens else None,
                NODE_NETWORK: hash_feature_vector(network_tokens, D_FEATURE) if network_tokens else None,
            },
            {
                "token_count": len(tokens),
                "api_count": len(api_like_tokens),
                "network_count": len(network_tokens),
                "header_count": len(header_markers),
                "suspicious_count": len(suspicious_terms),
                "entropy": round(float(entropy), 4),
                "suspicious_terms": suspicious_terms,
            }
        )

    def parse(self, input_data, input_type=None) -> dict:
        """
        Intercepts input, categorizes it, and formally outputs a strict PyG block.
        Allows for an explicit input_type hint (json, csv, api).
        """
        import json
        import pandas as pd
        import io

        raw_text = input_data if isinstance(input_data, str) else json.dumps(input_data, default=str)

        # Initiate the GNN Ghost Nodes Space
        nodes = {
            NODE_HEADER: np.zeros(D_FEATURE, dtype=np.float32),
            NODE_ENTROPY: np.zeros(D_FEATURE, dtype=np.float32),
            NODE_API: np.zeros(D_FEATURE, dtype=np.float32),
            NODE_NETWORK: np.zeros(D_FEATURE, dtype=np.float32)
        }
        nodes_filled = 0

        # FORMAT AWARENESS LOGIC
        if input_type == "json":
            try:
                if isinstance(input_data, str):
                    input_data = json.loads(input_data)
                # Extract behavior from JSON structure
                if "api" in input_data or "api_calls" in input_data:
                    calls = input_data.get("api", []) or input_data.get("api_calls", [])
                    nodes[NODE_API] = hash_feature_vector([self.canonicalize_api(str(t)) for t in calls], D_FEATURE)
                    nodes_filled += 1
                if "network" in input_data or "ip" in input_data:
                    net = input_data.get("network", []) or input_data.get("ip", [])
                    nodes[NODE_NETWORK] = hash_feature_vector([str(x) for x in net], D_FEATURE)
                    nodes_filled += 1
            except:
                input_type = "unknown" # Fallback

        elif input_type == "csv":
            try:
                df = pd.read_csv(io.StringIO(input_data))
                if is_native_schema(df.columns) and len(df) > 0:
                    # This dataset's own native schema (data/cleaned_data.csv) --
                    # reconstruct the exact same 4 feature vectors used at
                    # training time instead of falling through to the generic
                    # api/network column check below (which never matches this
                    # schema, since its columns are all f_..._N, not literally
                    # "api"/"network"/"ip") or the raw-text heuristic fallback.
                    row = df.iloc[0]
                    vectors = native_row_to_node_vectors(row.get, D_FEATURE)
                    for node_idx, vec in vectors.items():
                        nodes[node_idx] = vec.astype(np.float32)
                    nodes_filled = 4
                    input_type = "csv_native_schema"
                else:
                    # Heuristic: Check common security columns for arbitrary/unknown CSV logs
                    cols = df.columns.str.lower()
                    if "api" in cols:
                        nodes[NODE_API] = hash_feature_vector([self.canonicalize_api(str(x)) for x in df.iloc[:,0].tolist()], D_FEATURE)
                        nodes_filled += 1
                    if "network" in cols or "ip" in cols:
                        nodes[NODE_NETWORK] = hash_feature_vector([str(x) for x in df.iloc[:,0].tolist()], D_FEATURE)
                        nodes_filled += 1
            except:
                input_type = "unknown"

        heuristic_nodes, signal_summary = self._build_text_heuristics(raw_text)
        for node_idx, vec in heuristic_nodes.items():
            if vec is None:
                continue
            if np.count_nonzero(nodes[node_idx]) == 0 and np.count_nonzero(vec) > 0:
                nodes[node_idx] = vec.astype(np.float32) if isinstance(vec, np.ndarray) else vec
                nodes_filled += 1

        # MODE 1: Default/Raw Text API fallback
        if nodes_filled == 0:
            input_type = "api_only"
            tokens = [t.strip() for t in raw_text.split(',') if t.strip()] if isinstance(raw_text, str) else input_data
            nodes[NODE_API] = hash_feature_vector([self.canonicalize_api(str(t)) for t in tokens], D_FEATURE)
            nodes_filled += 1

        # Keeping duplicates is heavily intentional: 
        # Phase 1's `hash_feature_vector()` uses += 1.0 logic to explicitly score API call frequency.

        # Exact ground truth of which nodes carry real signal vs. zero-padding,
        # checked directly on the feature arrays rather than approximated from
        # signal_summary counts (which only reflect the text-heuristic path,
        # not the structured JSON/CSV column fills above).
        nodes_filled_detail = {
            "header": bool(np.count_nonzero(nodes[NODE_HEADER])),
            "entropy": bool(np.count_nonzero(nodes[NODE_ENTROPY])),
            "api": bool(np.count_nonzero(nodes[NODE_API])),
            "network": bool(np.count_nonzero(nodes[NODE_NETWORK])),
        }

        # Stitch One-Hots back onto Vectors safely
        final_tensors = []
        for i in range(4):
            vec = nodes[i]
            one_hot = np.array(get_one_hot_type(i), dtype=np.float32)
            comb = np.concatenate([vec, one_hot])
            final_tensors.append(torch.tensor(comb, dtype=torch.float32))

        # Push the arrays (and zeroes) directly through the original normalization logic
        normed_nodes = self._apply_normalization(final_tensors)
        X = torch.stack(normed_nodes)
        
        data = Data(x=X, edge_index=EDGE_INDEX, y=None)

        # C_graph = (1/4) * sum(indicator(||x_i||_2 > 0)) -- recounted directly
        # from nodes_filled_detail's indicator values, not the `nodes_filled`
        # counter above. The counter increments once per fill *site* (JSON/CSV
        # column fill, then again if the independent text-heuristic merge also
        # touches the same node), so it could count a node twice, or count a
        # fill that produced an all-zero vector (e.g. an empty JSON "api" list)
        # as "filled". The indicator recount is exact: 1 node out of 4 = 25%.
        score = sum(1 for v in nodes_filled_detail.values() if v) / 4.0
        interpretation = "low_due_to_missing_nodes" if score < 1.0 else "high_confidence_structured_input"
        
        return {
            "graph": data,
            "completeness": round(score, 2),
            "type": input_type,
            "confidence_interpretation": interpretation,
            "signal_summary": signal_summary,
            "nodes_filled_detail": nodes_filled_detail,
        }

    def to_pacx_vector(self, graph: Data) -> np.ndarray:
        """
        Legacy Export formatting. Overrides the 4-dim Graph space downwards 
        into a 1040-dim flat tabular list suitable for Phase 1 XGBoost inference.
        """
        return graph.x.flatten().numpy()
