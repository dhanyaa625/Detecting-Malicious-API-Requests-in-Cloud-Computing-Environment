"""
Real feature extraction from a raw PE binary (.exe/.dll bytes), for the live
"upload any file, including ones never seen in training" path.

Unlike core/input_adapter.py's CSV path (which only reconstructs the trained
feature vectors when the upload already IS a precomputed row from this
project's own dataset), this module computes those same 470 native-schema
columns from scratch, straight off the binary:

  - NATIVE_PE_HEADER_COLS (30) and NATIVE_ENTROPY_COLS (23): exact values,
    read directly off pefile's parsed FILE_HEADER/OPTIONAL_HEADER/section
    table/import-export counts/resource directory. No approximation --
    these fields have one unambiguous real value per binary.

  - NATIVE_API_COLS (256) and NATIVE_NETWORK_COLS (160): the training
    dataset's exact original encoding for these columns (behavioral
    import/export-name buckets, string-artifact buckets) isn't known or
    reproducible -- there's no extractor for it anywhere in this repo, only
    the already-featurized CSV. So this module takes the same principled
    approach the paper already documents for raw-text input (Eq. 1-2's
    "deliberately approximate bridge"): real imported/exported function
    names and real strings pulled from the binary are keyword-categorized
    into the same category names the schema expects, then hash-embedded
    with graph_constructor.hash_feature_vector -- the same mechanism
    already used for the unstructured-text fallback path. This is an
    honest approximation of real extracted content, not a fabrication of
    values with no basis in the file.

Usage:
    row = extract_native_row(raw_bytes)          # dict[str, float], 470 keys
    text_summary = extract_text_summary(raw_bytes)  # for the PAC-X pathway
"""
import json
import os
import re

import pefile

from core.graph_constructor import (
    NATIVE_PE_HEADER_COLS, NATIVE_ENTROPY_COLS, NATIVE_IMPORT_CATS,
    NATIVE_EXPORT_CATS, NATIVE_STRING_GROUPS, NATIVE_FEATURE_SLOTS,
    hash_feature_vector,
)

# ─────────────────────────────────────────────────────────────────────────────
# Behavioral keyword buckets for import/export function names.
# Matches the schema's category names (open/close/create/resume/kill/call/
# delete/other) with a reasonable, documented keyword mapping -- not a
# reverse-engineering of the unknown original scheme, a principled bucketing
# of real names into the categories the schema already defines.
# ─────────────────────────────────────────────────────────────────────────────
_CATEGORY_KEYWORDS = {
    "open": ("open",),
    "close": ("close",),
    "create": ("create",),
    "resume": ("resume",),
    "kill": ("terminate", "kill", "exitprocess", "exitthread"),
    "delete": ("delete", "free", "destroy"),
    "call": ("call", "invoke", "exec", "getprocaddress", "loadlibrary"),
}


def _categorize_name(name: str) -> str:
    lowered = name.lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return category
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# Regex classifiers for extracted printable strings, into the schema's 10
# string-artifact groups. Reuses the same suspicious-keyword list already in
# core/input_adapter.py's text-heuristic path for consistency.
# ─────────────────────────────────────────────────────────────────────────────
_URL_RE = re.compile(r'^(https?://|www\.)', re.IGNORECASE)
_IP_RE = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
_EMAIL_RE = re.compile(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
_INVALID_EMAIL_RE = re.compile(r'@')
_DIR_RE = re.compile(r'^[A-Za-z]:\\|\\\\|/[a-zA-Z]')
_FILENAME_RE = re.compile(r'\.[a-zA-Z]{2,4}$')
_SUSPICIOUS_KEYWORDS = (
    "virtualalloc", "writeprocessmemory", "createremotethread", "powershell",
    "cmd.exe", "rundll32", "regsvr32", "winexec", "cryptdecrypt",
)

_MIN_STRING_LEN = 4
_PRINTABLE_RE = re.compile(rb'[\x20-\x7e]{%d,}' % _MIN_STRING_LEN)


def _classify_string(token: str) -> str:
    lowered = token.lower()
    if _URL_RE.match(token):
        return "URL"
    if _IP_RE.match(token):
        return "IP"
    if _EMAIL_RE.match(token):
        return "Email"
    if _INVALID_EMAIL_RE.search(token):
        return "InvEmail"
    if _DIR_RE.match(token):
        return "DIR"
    if _FILENAME_RE.search(token) and len(token) < 64:
        return "Filename"
    if any(kw in lowered for kw in _SUSPICIOUS_KEYWORDS):
        return "Keyword"
    if " " in token and len(token) > 20:
        return "Sentence"
    if " " not in token and len(token) > 12:
        return "LongWord"
    return "Garbage"


def extract_strings(raw_bytes: bytes, limit: int = 20000) -> list:
    """Standard printable-ASCII string extraction (like the `strings` CLI)."""
    matches = _PRINTABLE_RE.findall(raw_bytes[: 1 << 22])  # cap at 4 MiB scanned
    return [m.decode("ascii", errors="ignore") for m in matches[:limit]]


def _walk_resources(pe: "pefile.PE") -> list:
    """Returns raw byte blocks for every leaf resource entry, for entropy/size stats."""
    blocks = []
    if not hasattr(pe, "DIRECTORY_ENTRY_RESOURCE"):
        return blocks
    try:
        for entry in pe.DIRECTORY_ENTRY_RESOURCE.entries:
            _walk_resource_entry(pe, entry, blocks)
    except Exception:
        pass
    return blocks


def _walk_resource_entry(pe, entry, blocks, depth=0):
    if depth > 4:
        return
    if hasattr(entry, "directory"):
        for sub_entry in entry.directory.entries:
            _walk_resource_entry(pe, sub_entry, blocks, depth + 1)
    elif hasattr(entry, "data"):
        try:
            rva = entry.data.struct.OffsetToData
            size = entry.data.struct.Size
            data = pe.get_memory_mapped_image()[rva: rva + size]
            if data:
                blocks.append(data)
        except Exception:
            pass


def _shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    import math
    freq = [0] * 256
    for b in data:
        freq[b] += 1
    length = len(data)
    entropy = 0.0
    for count in freq:
        if count:
            p = count / length
            entropy -= p * math.log2(p)
    return entropy


def extract_native_row(raw_bytes: bytes) -> dict:
    """
    Parses a raw PE binary and returns a dict covering all 470 native-schema
    columns (core.graph_constructor.NATIVE_SCHEMA_COLUMNS), ready to hand to
    native_row_to_node_vectors(row.get, D_FEATURE) exactly like a CSV row.
    """
    row = {}
    pe = pefile.PE(data=raw_bytes, fast_load=False)

    try:
        # ---- PE header (30 exact values) ----
        fh, oh = pe.FILE_HEADER, pe.OPTIONAL_HEADER
        header_values = {
            "f_Machine_0": fh.Machine,
            "f_SizeOfOptionalHeader_0": fh.SizeOfOptionalHeader,
            "f_Characteristics_0": fh.Characteristics,
            "f_MajorLinkerVersion_0": oh.MajorLinkerVersion,
            "f_MinorLinkerVersion_0": oh.MinorLinkerVersion,
            "f_SizeOfCode_0": oh.SizeOfCode,
            "f_SizeOfInitializedData_0": oh.SizeOfInitializedData,
            "f_SizeOfUninitializedData_0": oh.SizeOfUninitializedData,
            "f_AddressOfEntryPoint_0": oh.AddressOfEntryPoint,
            "f_BaseOfCode_0": oh.BaseOfCode,
            "f_BaseOfData_0": getattr(oh, "BaseOfData", 0),  # PE32+ has no BaseOfData
            "f_ImageBase_0": oh.ImageBase,
            "f_SectionAlignment_0": oh.SectionAlignment,
            "f_FileAlignment_0": oh.FileAlignment,
            "f_MajorOperatingSystemVersion_0": oh.MajorOperatingSystemVersion,
            "f_MinorOperatingSystemVersion_0": oh.MinorOperatingSystemVersion,
            "f_MajorImageVersion_0": oh.MajorImageVersion,
            "f_MinorImageVersion_0": oh.MinorImageVersion,
            "f_MajorSubsystemVersion_0": oh.MajorSubsystemVersion,
            "f_MinorSubsystemVersion_0": oh.MinorSubsystemVersion,
            "f_SizeOfImage_0": oh.SizeOfImage,
            "f_SizeOfHeaders_0": oh.SizeOfHeaders,
            "f_CheckSum_0": oh.CheckSum,
            "f_Subsystem_0": oh.Subsystem,
            "f_DllCharacteristics_0": oh.DllCharacteristics,
            "f_SizeOfStackReserve_0": oh.SizeOfStackReserve,
            "f_SizeOfStackCommit_0": oh.SizeOfStackCommit,
            "f_SizeOfHeapReserve_0": oh.SizeOfHeapReserve,
            "f_SizeOfHeapCommit_0": oh.SizeOfHeapCommit,
            "f_LoaderFlags_0": oh.LoaderFlags,
            "f_NumberOfRvaAndSizes_0": oh.NumberOfRvaAndSizes,
        }
        row.update(header_values)
        assert set(header_values) == set(NATIVE_PE_HEADER_COLS)

        # ---- Section entropy/size stats (real, computed per-section) ----
        entropies = [s.get_entropy() for s in pe.sections] or [0.0]
        raw_sizes = [s.SizeOfRawData for s in pe.sections] or [0]
        virt_sizes = [s.Misc_VirtualSize for s in pe.sections] or [0]

        imports = getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])
        import_names = []
        dll_names = set()
        n_ordinal_imports = 0
        for entry in imports:
            dll_names.add(entry.dll.decode(errors="ignore") if entry.dll else "")
            for imp in entry.imports:
                if imp.name:
                    import_names.append(imp.name.decode(errors="ignore"))
                else:
                    n_ordinal_imports += 1

        exports = getattr(pe, "DIRECTORY_ENTRY_EXPORT", None)
        export_symbols = exports.symbols if exports else []
        export_names = [s.name.decode(errors="ignore") for s in export_symbols if s.name]

        resource_blocks = _walk_resources(pe)
        resource_entropies = [_shannon_entropy(b) for b in resource_blocks] or [0.0]
        resource_sizes = [len(b) for b in resource_blocks] or [0]

        load_config = getattr(pe, "DIRECTORY_ENTRY_LOAD_CONFIG", None)
        load_config_size = load_config.struct.Size if load_config else 0

        version_size = 0
        if hasattr(pe, "FileInfo"):
            try:
                version_size = sum(len(str(fi)) for fi in pe.FileInfo)
            except Exception:
                version_size = 0

        entropy_values = {
            "f_SectionsNb_0": len(pe.sections),
            "f_SectionsMeanEntropy_0": sum(entropies) / len(entropies),
            "f_SectionsMinEntropy_0": min(entropies),
            "f_SectionsMaxEntropy_0": max(entropies),
            "f_SectionsMeanRawsize_0": sum(raw_sizes) / len(raw_sizes),
            "f_SectionsMinRawsize_0": min(raw_sizes),
            "f_SectionsMaxRawsize_0": max(raw_sizes),
            "f_SectionsMeanVirtualsize_0": sum(virt_sizes) / len(virt_sizes),
            "f_SectionsMinVirtualsize_0": min(virt_sizes),
            "f_SectionMaxVirtualsize_0": max(virt_sizes),
            "f_ImportsNbDLL_0": len(dll_names),
            "f_ImportsNb_0": len(import_names) + n_ordinal_imports,
            "f_ImportsNbOrdinal_0": n_ordinal_imports,
            "f_ExportNb_0": len(export_symbols),
            "f_ResourcesNb_0": len(resource_blocks),
            "f_ResourcesMeanEntropy_0": sum(resource_entropies) / len(resource_entropies),
            "f_ResourcesMinEntropy_0": min(resource_entropies),
            "f_ResourcesMaxEntropy_0": max(resource_entropies),
            "f_ResourcesMeanSize_0": sum(resource_sizes) / len(resource_sizes),
            "f_ResourcesMinSize_0": min(resource_sizes),
            "f_ResourcesMaxSize_0": max(resource_sizes),
            "f_LoadConfigurationSize_0": load_config_size,
            "f_VersionInformationSize_0": version_size,
        }
        row.update(entropy_values)
        assert set(entropy_values) == set(NATIVE_ENTROPY_COLS)

        # ---- API node: real import/export names, keyword-bucketed + hash-embedded ----
        import_buckets = {cat: [] for cat in NATIVE_IMPORT_CATS}
        for name in import_names:
            import_buckets[_categorize_name(name)].append(name)
        export_buckets = {cat: [] for cat in NATIVE_EXPORT_CATS}
        for name in export_names:
            export_buckets[_categorize_name(name)].append(name)

        for cat in NATIVE_IMPORT_CATS:
            vec = hash_feature_vector(import_buckets[cat], NATIVE_FEATURE_SLOTS)
            for i in range(NATIVE_FEATURE_SLOTS):
                row[f"f_ImportsList_{cat}_{i}"] = float(vec[i])
        for cat in NATIVE_EXPORT_CATS:
            vec = hash_feature_vector(export_buckets[cat], NATIVE_FEATURE_SLOTS)
            for i in range(NATIVE_FEATURE_SLOTS):
                row[f"f_ExportsList_{cat}_{i}"] = float(vec[i])

        # ---- Network node: real strings from the binary, classified + hash-embedded ----
        strings = extract_strings(raw_bytes)
        string_buckets = {group: [] for group in NATIVE_STRING_GROUPS}
        for token in strings:
            group_label = _classify_string(token)
            if group_label in string_buckets:
                string_buckets[group_label].append(token)

        for group_label, prefix in NATIVE_STRING_GROUPS.items():
            vec = hash_feature_vector(string_buckets[group_label], NATIVE_FEATURE_SLOTS)
            for i in range(NATIVE_FEATURE_SLOTS):
                row[f"{prefix}_{i}"] = float(vec[i])

    finally:
        pe.close()

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Out-of-distribution check. A real PE binary that was never part of this
# dataset's collection process can have real, correctly-extracted feature
# values that are still statistically nothing like what the model was
# trained on (verified: real-world system binaries produced mean|z-score|
# values 40-100x past what any genuine training sample ever showed on the
# Network node -- see data/ood_node_thresholds.json's note). The GAT has no
# built-in "I don't recognize this" signal and will confidently misclassify
# such input, so this check is a data-derived (not guessed) circuit breaker:
# thresholds are the 99th-percentile mean|z-score| actually observed across
# every real training sample (scripts/build_ood_thresholds.py), not an
# arbitrary cutoff.
# ─────────────────────────────────────────────────────────────────────────────
_OOD_THRESHOLDS_PATH = "data/ood_node_thresholds.json"
_NODE_ORDER = ["header", "entropy", "api", "network"]
_ood_thresholds_cache = None


def _load_ood_thresholds():
    global _ood_thresholds_cache
    if _ood_thresholds_cache is not None:
        return _ood_thresholds_cache
    if not os.path.exists(_OOD_THRESHOLDS_PATH):
        _ood_thresholds_cache = {}
        return _ood_thresholds_cache
    with open(_OOD_THRESHOLDS_PATH) as f:
        _ood_thresholds_cache = json.load(f)["thresholds"]
    return _ood_thresholds_cache


def check_node_ood(normalized_tensor) -> dict:
    """
    `normalized_tensor`: [4, 260] tensor, post z-score normalization (same
    tensor layout as the graph's node feature matrix). Returns
    {node_idx: bool}, True if that node's mean|z-score| across its 256 real
    dims exceeds the p99 threshold real training samples actually show.
    Missing thresholds file -> no flags raised (fails open, not closed).
    """
    thresholds = _load_ood_thresholds()
    flags = {}
    for node_idx, name in enumerate(_NODE_ORDER):
        if name not in thresholds:
            flags[node_idx] = False
            continue
        mean_abs_z = float(normalized_tensor[node_idx, :256].abs().mean().item())
        flags[node_idx] = mean_abs_z > thresholds[name]["p99"]
    return flags


def extract_text_summary(raw_bytes: bytes, max_items: int = 200) -> str:
    """
    A textual view of the same real extracted content, for the PAC-X
    heuristic pathway (which has signal on raw text/API-log content, not
    numeric PE-header fields -- see core/pacx_analyzer.py). Real imported
    function names and real flagged strings, not synthesized text.
    """
    pe = pefile.PE(data=raw_bytes, fast_load=False)
    try:
        imports = getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])
        api_names = []
        for entry in imports:
            for imp in entry.imports:
                if imp.name:
                    api_names.append(imp.name.decode(errors="ignore"))
    finally:
        pe.close()

    strings = extract_strings(raw_bytes)
    flagged = [s for s in strings if _classify_string(s) in ("URL", "IP", "Email", "Keyword")]

    parts = api_names[:max_items] + flagged[:max_items]
    return "\n".join(parts)
