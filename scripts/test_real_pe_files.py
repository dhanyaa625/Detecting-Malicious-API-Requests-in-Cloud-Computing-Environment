"""
Batch-tests the live pipeline (core/pe_feature_extractor.py's raw-binary path
+ full dual-pathway fusion, exactly as app.py's /api/analyze/live runs it)
against a folder of real PE files with known labels -- e.g. amauricio's
"Benign & Malicious PE Files" Kaggle dataset's `samples/` folder + its
samples.csv label file (https://www.kaggle.com/datasets/amauricio/pe-files-malwares).

This project can't fetch that dataset itself (fetching files from an
external host is out of scope here regardless of research intent) -- this
script is what to run once you've downloaded it yourself.

Reports, per real file: the GNN pathway's own verdict, PAC-X's own verdict,
and the FULL fused verdict (with trust arbitration) -- so you can see
whether the safety net (Section IV-D style arbitration) is actually
correcting the GNN's mistakes on real out-of-distribution files, exactly
like it did for notepad.exe/calc.exe.

Usage:
    python scripts/test_real_pe_files.py --samples-dir path/to/samples --labels-csv path/to/samples.csv

The labels CSV needs a filename/id column and a label column. Common
amauricio-style naming (id, legitimate) is auto-detected; pass
--id-col / --label-col to override if your download uses different names.
`legitimate` values of 1/0 or True/False/malicious/benign are all accepted.
"""
import argparse
import asyncio
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app  # noqa: E402


class _FakeUploadFile:
    def __init__(self, raw_bytes: bytes):
        self._data = raw_bytes

    async def read(self):
        return self._data


def _guess_column(columns, candidates):
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def _normalize_label(value) -> str:
    """Returns 'Malicious' or 'Benign' from whatever the CSV encodes."""
    s = str(value).strip().lower()
    if s in ("1", "true", "malicious", "malware", "yes"):
        return "Malicious"
    if s in ("0", "false", "benign", "legitimate", "no"):
        return "Benign"
    raise ValueError(f"Unrecognized label value: {value!r}")


async def _analyze_one(raw_bytes: bytes) -> dict:
    result = await app.analyze_live(
        background_tasks=app.BackgroundTasks(),
        text=None,
        file=_FakeUploadFile(raw_bytes),
        input_type="api",
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", required=True, help="Folder of raw PE files (extension may be stripped)")
    parser.add_argument("--labels-csv", required=True, help="CSV with filename/id + label columns")
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--limit", type=int, default=200, help="Max files to test (dataset can be 19k+ files)")
    args = parser.parse_args()

    df = pd.read_csv(args.labels_csv)
    id_col = args.id_col or _guess_column(df.columns, ["id", "name", "filename", "hash", "md5", "sha256"])
    label_col = args.label_col or _guess_column(df.columns, ["legitimate", "label", "malicious", "class"])
    if not id_col or not label_col:
        print(f"[-] Could not auto-detect columns. Found: {list(df.columns)}. "
              f"Pass --id-col and --label-col explicitly.")
        sys.exit(1)
    print(f"[+] Using id_col={id_col!r}, label_col={label_col!r}")

    rows = df[[id_col, label_col]].dropna().head(args.limit).to_dict("records")

    results = []
    counts = {"gnn_correct": 0, "pacx_correct": 0, "fused_correct": 0, "total": 0, "skipped": 0}

    for row in rows:
        file_id = str(row[id_col])
        path = os.path.join(args.samples_dir, file_id)
        if not os.path.exists(path):
            path = os.path.join(args.samples_dir, file_id + ".exe")
        if not os.path.exists(path):
            counts["skipped"] += 1
            continue

        try:
            true_label = _normalize_label(row[label_col])
            with open(path, "rb") as f:
                raw_bytes = f.read()
            if raw_bytes[:2] != b"MZ":
                counts["skipped"] += 1
                continue

            analysis = asyncio.run(_analyze_one(raw_bytes))
            gnn_label = analysis["path_gnn"]["diagnosis"]
            pacx_label = analysis["path_pacx"]["diagnosis"]
            fused_label = analysis["decision_fusion"]["final_label"]

            counts["total"] += 1
            counts["gnn_correct"] += int(gnn_label == true_label)
            counts["pacx_correct"] += int(pacx_label == true_label)
            counts["fused_correct"] += int(fused_label == true_label)

            results.append({
                "file": file_id,
                "true_label": true_label,
                "gnn_label": gnn_label,
                "pacx_label": pacx_label,
                "fused_label": fused_label,
                "ood_flags": analysis["path_gnn"]["signal_summary"].get("ood_flags", {}),
                "completeness": analysis["completeness"],
            })
            print(f"{file_id}: true={true_label} gnn={gnn_label} pacx={pacx_label} fused={fused_label}")
        except Exception as e:
            print(f"[-] {file_id}: error -- {e}")
            counts["skipped"] += 1

    total = max(1, counts["total"])
    summary = {
        "n_tested": counts["total"],
        "n_skipped": counts["skipped"],
        "gnn_only_accuracy": round(100 * counts["gnn_correct"] / total, 2),
        "pacx_only_accuracy": round(100 * counts["pacx_correct"] / total, 2),
        "fused_accuracy": round(100 * counts["fused_correct"] / total, 2),
    }
    print("\n=== SUMMARY (real, previously-unseen PE files) ===")
    print(json.dumps(summary, indent=2))

    os.makedirs("data/real_pe_test", exist_ok=True)
    with open("data/real_pe_test/report.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print("[+] Saved data/real_pe_test/report.json")


if __name__ == "__main__":
    main()
