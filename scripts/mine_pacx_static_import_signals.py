"""
Data-mines evidence-weighted STATIC-import API signals for PAC-X, to replace
the arbitrary `min_weight=2.5` damping in core/pacx_analyzer.py (a
reasonable-but-unvalidated guess made when the only mined weights available
were from DYNAMIC execution traces, applied to STATIC import-table content
with a hand-picked cutoff).

Malware side: real static import-table data (API function names actually
linked by the binary) from 18,551 real malware samples --
yousuf_dataset/API Functions{1-4}.csv, from "A Multi-feature Dataset for
Windows PE Malware Classification" (Yousuf et al., Mendeley
DOI 10.17632/vnj7sxkt53.1, arXiv:2210.16285). Each malware sample's import
list can be split across multiple rows (same SHA256 repeated) when it
exceeds one file's column width -- reconstructed here by grouping on SHA256
and taking the union of all API names across every row/file it appears in.

Benign side: this dataset ships malware only (no benign class), so there is
nothing to pair it against out of the box. Built here from real, genuine
Windows system binaries already present on this machine
(C:\\Windows\\System32\\*.dll and *.exe -- signed Microsoft binaries, the
same kind of real benign PE file already used for the notepad.exe/calc.exe
spot checks earlier in this project) via the exact same pefile import
extraction used at inference time (core/pe_feature_extractor.py's
extract_text_summary), so mining and inference see API names the same way.

Proper 60/20/20 split (stratified by class, fixed seed), no peeking:
  - TRAIN: per-API P(present|malware), P(present|benign), log-likelihood-
    ratio weight log2((p_malware+eps)/(p_benign+eps)).
  - VAL: picks the one free parameter (decision threshold on summed score)
    by balanced accuracy (classes are imbalanced -- more malware samples
    than benign).
  - TEST: touched once, after threshold selection.

Usage: python scripts/mine_pacx_static_import_signals.py
"""
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pefile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EPS = 0.01
YOUSUF_DIR = "yousuf_dataset"
BENIGN_DIRS = [r"C:\Windows\System32"]
OUT_PATH = "data/pacx_mining/mined_static_import_signals.json"


def load_malware_samples():
    """SHA256 -> set(API function names), reconstructed across all 4
    API Functions*.csv files (a sample's row can repeat across files/rows
    when its import list is too wide for one row)."""
    samples = {}
    for i in range(1, 5):
        path = os.path.join(YOUSUF_DIR, f"API Functions{i}.csv")
        df = pd.read_csv(path, header=None, low_memory=False)
        api_cols = df.columns[2:]
        for row in df.itertuples(index=False):
            sha = row[0]
            apis = samples.setdefault(sha, set())
            for val in row[2:]:
                if isinstance(val, str) and val.strip():
                    apis.add(val.strip())
        print(f"[+] {path}: {len(df)} rows loaded")
    print(f"[+] Reconstructed {len(samples)} unique malware samples "
          f"(real dataset has 18,551 -- some rows may be dropped by malformed CSV lines)")
    return samples


def extract_benign_apis(path):
    try:
        pe = pefile.PE(path, fast_load=True)
        try:
            pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
            imports = getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])
            apis = set()
            for entry in imports:
                for imp in entry.imports:
                    if imp.name:
                        apis.add(imp.name.decode(errors="ignore"))
            return apis
        finally:
            pe.close()
    except Exception:
        return None


def load_benign_samples():
    """path -> set(API function names), from real Windows system binaries."""
    files = []
    for d in BENIGN_DIRS:
        files.extend(glob.glob(os.path.join(d, "*.dll")))
        files.extend(glob.glob(os.path.join(d, "*.exe")))
    print(f"[+] Found {len(files)} candidate real benign PE files under {BENIGN_DIRS}")

    samples = {}
    skipped = 0
    for path in files:
        apis = extract_benign_apis(path)
        if apis:
            samples[path] = apis
        else:
            skipped += 1
    print(f"[+] Extracted real imports from {len(samples)} benign files ({skipped} skipped -- "
          f"unparsable/no-import/locked)")
    return samples


def stratified_split(sample_ids, seed=42):
    rng = np.random.RandomState(seed)
    ids = list(sample_ids)
    rng.shuffle(ids)
    n = len(ids)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    return ids[:train_end], ids[train_end:val_end], ids[val_end:]


def presence_rates(train_ids, id_to_apis, id_to_label, vocab):
    malware_ids = [i for i in train_ids if id_to_label[i] == 1]
    benign_ids = [i for i in train_ids if id_to_label[i] == 0]
    n_m, n_b = len(malware_ids), len(benign_ids)
    rates = {}
    for api in vocab:
        p_m = sum(1 for i in malware_ids if api in id_to_apis[i]) / n_m if n_m else 0.0
        p_b = sum(1 for i in benign_ids if api in id_to_apis[i]) / n_b if n_b else 0.0
        rates[api] = (p_m, p_b)
    return rates


def score_sample(apis_present, weights):
    return sum(weights.get(a, 0.0) for a in apis_present)


def main():
    print("[+] Loading malware side (real static imports, Yousuf dataset)...")
    malware = load_malware_samples()
    print("[+] Loading benign side (real static imports, local Windows system binaries)...")
    benign = load_benign_samples()

    id_to_apis = {}
    id_to_label = {}
    for sha, apis in malware.items():
        id_to_apis[("m", sha)] = apis
        id_to_label[("m", sha)] = 1
    for path, apis in benign.items():
        id_to_apis[("b", path)] = apis
        id_to_label[("b", path)] = 0

    print(f"[+] Total samples: {len(id_to_apis)} ({len(malware)} malware, {len(benign)} benign)")

    m_train, m_val, m_test = stratified_split([k for k in id_to_apis if k[0] == "m"])
    b_train, b_val, b_test = stratified_split([k for k in id_to_apis if k[0] == "b"])
    train_ids = m_train + b_train
    val_ids = m_val + b_val
    test_ids = m_test + b_test
    print(f"[+] train={len(train_ids)} ({len(m_train)}m/{len(b_train)}b)  "
          f"val={len(val_ids)} ({len(m_val)}m/{len(b_val)}b)  "
          f"test={len(test_ids)} ({len(m_test)}m/{len(b_test)}b)")

    # Vocabulary: APIs seen in TRAIN only (no peeking at val/test-only names).
    vocab = set()
    for i in train_ids:
        vocab.update(id_to_apis[i])
    print(f"[+] Train vocabulary: {len(vocab)} unique API names")

    print("[+] Computing log-likelihood-ratio weights on TRAIN split...")
    rates = presence_rates(train_ids, id_to_apis, id_to_label, vocab)
    weights = {}
    for api, (p_m, p_b) in rates.items():
        llr = math.log2((p_m + EPS) / (p_b + EPS))
        weights[api] = round(llr, 4)

    top15 = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:15]
    print("[+] Top 15 highest-evidence static-import APIs (train):")
    for name, w in top15:
        p_m, p_b = rates[name]
        print(f"   {name:35s} weight={w:6.2f}  p_malware={p_m:.3f}  p_benign={p_b:.3f}")

    def scores_labels(ids):
        scores = np.array([score_sample(id_to_apis[i], weights) for i in ids])
        labels = np.array([id_to_label[i] for i in ids])
        return scores, labels

    print("[+] Selecting decision threshold on VAL split (balanced accuracy -- "
          "classes are imbalanced, more malware samples than benign)...")
    val_scores, val_labels = scores_labels(val_ids)
    best_threshold, best_balanced_acc = None, -1
    lo, hi = float(val_scores.min()), float(val_scores.max())
    step = max((hi - lo) / 400.0, 0.01)
    for t in np.arange(lo, hi, step):
        preds = (val_scores > t).astype(int)
        tpr = (preds[val_labels == 1] == 1).mean() if (val_labels == 1).any() else 0.0
        tnr = (preds[val_labels == 0] == 0).mean() if (val_labels == 0).any() else 0.0
        balanced_acc = (tpr + tnr) / 2.0
        if balanced_acc > best_balanced_acc:
            best_balanced_acc, best_threshold = balanced_acc, t
    print(f"[+] Best threshold on VAL: {best_threshold:.2f} (val balanced accuracy {best_balanced_acc*100:.2f}%)")

    print("[+] Final check on TEST split (touched once)...")
    test_scores, test_labels = scores_labels(test_ids)
    test_preds = (test_scores > best_threshold).astype(int)
    test_acc = (test_preds == test_labels).mean()
    fp = int(((test_preds == 1) & (test_labels == 0)).sum())
    fn = int(((test_preds == 0) & (test_labels == 1)).sum())
    n_benign = int((test_labels == 0).sum())
    n_malicious = int((test_labels == 1).sum())
    fp_rate = round(100 * fp / max(1, n_benign), 2)
    fn_rate = round(100 * fn / max(1, n_malicious), 2)
    test_summary = {
        "raw_accuracy": round(float(test_acc) * 100, 2),
        "balanced_accuracy": round(100 - (fp_rate + fn_rate) / 2, 2),
        "false_positive_rate": fp_rate,
        "false_negative_rate": fn_rate,
        "n_tested": len(test_ids),
        "n_benign": n_benign,
        "n_malicious": n_malicious,
    }
    print(json.dumps(test_summary, indent=2))

    report = {
        "note": (
            "Log-likelihood-ratio API evidence weights for STATIC import-table content "
            "(a raw PE binary upload's real DIRECTORY_ENTRY_IMPORT names), mined from real "
            "static imports of 18,551 real malware samples (Yousuf et al. multi-feature PE "
            "malware dataset, yousuf_dataset/API Functions*.csv) paired against real static "
            "imports of genuine Windows system binaries (C:\\Windows\\System32\\*.dll/*.exe) "
            "as the benign side, since the source dataset ships malware only. 60% TRAIN / 20% "
            "VAL / 20% TEST split, stratified by class, no peeking. "
            "weight(api) = log2((p_malware+0.01)/(p_benign+0.01))."
        ),
        "decision_threshold": round(float(best_threshold), 2),
        "weights": weights,
        "test_summary": test_summary,
        "n_malware_samples": len(malware),
        "n_benign_samples": len(benign),
    }
    os.makedirs("data/pacx_mining", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[+] Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
