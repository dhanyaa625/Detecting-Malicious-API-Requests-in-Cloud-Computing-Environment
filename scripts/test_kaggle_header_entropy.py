"""
Scales up the "does the GNN pathway generalize to real, independently-
sourced software" measurement from an anecdote (2 files) to a real
statistic, using kaggle_dataset/data.csv -- 138,047 real PE samples with
header+entropy features whose column names match this project's own
schema almost exactly (53/54 identical; the one difference,
'SectionMaxRawsize' vs 'SectionsMaxRawsize', is handled below).

This dataset has no import/export/string data (API + Network nodes stay
honestly unfilled, not approximated), so this measures the GNN-only and
PAC-X-default pathways plus fusion -- it does NOT test whether the
arbitration fix (which depends on PAC-X having real extracted text)
generalizes, since there's no raw binary here for PAC-X to read.

Usage:
    python scripts/test_kaggle_header_entropy.py --csv kaggle_dataset/data.csv --limit 1000
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
from core.graph_constructor import NATIVE_PE_HEADER_COLS, NATIVE_ENTROPY_COLS  # noqa: E402
import agents.analyst_agent as analyst_agent  # noqa: E402

# Agent 1's Gemini call only affects the natural-language reasoning text,
# not the actual decision_fusion math this script measures -- disabled here
# purely to avoid ~1000 real network round-trips (falls back to the
# deterministic static-template reasoning, same fusion/arbitration numbers).
analyst_agent.genai_client = None

# Their column name -> our column name. Verified by diffing kaggle_dataset/
# data.csv's real header against data/features.txt: 53/54 match after
# stripping our 'f_' prefix and '_0' suffix; this one differs.
_RENAME_OVERRIDES = {"SectionMaxRawsize": "SectionsMaxRawsize"}


def _our_name(their_name: str) -> str:
    their_name = _RENAME_OVERRIDES.get(their_name, their_name)
    return f"f_{their_name}_0"


class _FakeUploadFile:
    def __init__(self, raw_bytes: bytes):
        self._data = raw_bytes

    async def read(self):
        return self._data


async def _analyze_row(row_dict: dict) -> dict:
    return await app.analyze_live(
        background_tasks=app.BackgroundTasks(),
        text=json.dumps(row_dict),
        file=None,
        input_type="native_features",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="kaggle_dataset/data.csv")
    parser.add_argument("--limit", type=int, default=1000, help="Stratified sample size (dataset has 138k rows)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # A low-confidence row causes app.analyze_live to synchronously append a
    # pseudo-labeled sample to data/retrain_pool.pt (the actual retraining
    # never runs -- BackgroundTasks only fires inside a real ASGI request --
    # but the pool write is real). Snapshot it and restore after this batch
    # so a 1000-row test run doesn't leave synthetic rows mixed into the
    # project's real retrain pool.
    import shutil
    pool_path = "data/retrain_pool.pt"
    pool_backup = None
    if os.path.exists(pool_path):
        pool_backup = pool_path + ".pretest_backup"
        shutil.copy(pool_path, pool_backup)
        print(f"[+] Backed up {pool_path} before running (will restore after)")

    df = pd.read_csv(args.csv, sep="|", low_memory=False)
    assert "legitimate" in df.columns, f"Expected a 'legitimate' column, got: {list(df.columns)}"

    # Stratified sample: half benign, half malicious, so the result isn't
    # dominated by this CSV's own class imbalance (96,724 vs 41,323).
    per_class = args.limit // 2
    benign = df[df["legitimate"] == 1].sample(n=min(per_class, (df["legitimate"] == 1).sum()), random_state=args.seed)
    malicious = df[df["legitimate"] == 0].sample(n=min(per_class, (df["legitimate"] == 0).sum()), random_state=args.seed)
    sample = pd.concat([benign, malicious]).sample(frac=1.0, random_state=args.seed)  # shuffle
    print(f"[+] Testing {len(sample)} samples ({len(benign)} benign, {len(malicious)} malicious)")

    header_entropy_cols = NATIVE_PE_HEADER_COLS + NATIVE_ENTROPY_COLS
    their_cols = [c.replace("f_", "").rsplit("_0", 1)[0] for c in header_entropy_cols]

    counts = {"gnn_correct": 0, "pacx_correct": 0, "fused_correct": 0, "total": 0, "errors": 0}
    results = []

    for _, row in sample.iterrows():
        true_label = "Benign" if row["legitimate"] == 1 else "Malicious"
        row_dict = {}
        for our_col, their_col in zip(header_entropy_cols, their_cols):
            if their_col in row:
                row_dict[our_col] = float(row[their_col])
            elif _RENAME_OVERRIDES.get(their_col) and _RENAME_OVERRIDES[their_col] in row:
                row_dict[our_col] = float(row[_RENAME_OVERRIDES[their_col]])

        try:
            analysis = asyncio.run(_analyze_row(row_dict))
            gnn_label = analysis["path_gnn"]["diagnosis"]
            pacx_label = analysis["path_pacx"]["diagnosis"]
            fused_label = analysis["decision_fusion"]["final_label"]

            counts["total"] += 1
            counts["gnn_correct"] += int(gnn_label == true_label)
            counts["pacx_correct"] += int(pacx_label == true_label)
            counts["fused_correct"] += int(fused_label == true_label)

            results.append({
                "true_label": true_label,
                "gnn_label": gnn_label,
                "pacx_label": pacx_label,
                "fused_label": fused_label,
                "completeness": analysis["completeness"],
            })
        except Exception as e:
            counts["errors"] += 1
            if counts["errors"] <= 5:
                print(f"[-] Row error: {e}")

    total = max(1, counts["total"])
    summary = {
        "n_tested": counts["total"],
        "n_errors": counts["errors"],
        "gnn_only_accuracy": round(100 * counts["gnn_correct"] / total, 2),
        "pacx_default_accuracy": round(100 * counts["pacx_correct"] / total, 2),
        "fused_accuracy": round(100 * counts["fused_correct"] / total, 2),
    }

    # Break down false positive/negative rates for the GNN pathway specifically.
    gnn_fp = sum(1 for r in results if r["true_label"] == "Benign" and r["gnn_label"] == "Malicious")
    gnn_fn = sum(1 for r in results if r["true_label"] == "Malicious" and r["gnn_label"] == "Benign")
    n_benign_tested = sum(1 for r in results if r["true_label"] == "Benign")
    n_malicious_tested = sum(1 for r in results if r["true_label"] == "Malicious")
    summary["gnn_false_positive_rate"] = round(100 * gnn_fp / max(1, n_benign_tested), 2)
    summary["gnn_false_negative_rate"] = round(100 * gnn_fn / max(1, n_malicious_tested), 2)

    fused_fp = sum(1 for r in results if r["true_label"] == "Benign" and r["fused_label"] == "Malicious")
    fused_fn = sum(1 for r in results if r["true_label"] == "Malicious" and r["fused_label"] == "Benign")
    summary["fused_false_positive_rate"] = round(100 * fused_fp / max(1, n_benign_tested), 2)
    summary["fused_false_negative_rate"] = round(100 * fused_fn / max(1, n_malicious_tested), 2)

    print("\n=== SUMMARY (real, independently-sourced Kaggle samples) ===")
    print(json.dumps(summary, indent=2))

    os.makedirs("data/kaggle_header_entropy_test", exist_ok=True)
    with open("data/kaggle_header_entropy_test/report.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print("[+] Saved data/kaggle_header_entropy_test/report.json")

    if pool_backup is not None:
        shutil.move(pool_backup, pool_path)
        print(f"[+] Restored {pool_path} to its pre-test state")


if __name__ == "__main__":
    main()
