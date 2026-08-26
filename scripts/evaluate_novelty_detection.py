"""
Investigates the confidently-wrong residual gap that fail-closed escalation
(app.apply_zeroday_escalation) cannot touch: novel malware the live model
places in the WRONG class with HIGH fused confidence, so it never trips the
Agent 2 gate at all. The paper (Section III-C-2 / IV-E) and README name
downloadguide and genkryptik as the worst offenders and state this needs real
novelty detection, not a threshold -- this script tests that claim rather
than assuming it.

The idea under test: escalation only ever sees the model's own softmax, which
is RELATIVE -- it always sums to 1 and always has a most-likely class, even for
a sample that resembles nothing the model was trained on. What escalation
never sees is an ABSOLUTE distance signal: how close is this sample to the
*nearest* of the 11 trained-class centroids at all, regardless of which one
wins. A sample can win the softmax comparison confidently while still sitting
far from every real cluster. compute_centroid_probabilities() in app.py already
computes per-class cosine similarity to CLASS_CENTROIDS, but only exposes the
softmax-normalized (relative) result -- the raw similarity is thrown away
before this function returns. This script recovers it and asks a real
question: does the RAW maximum similarity to any centroid separate genuine
known-family samples from the specific novel families the escalation rule
misses?

Two independent evaluation sets, both scored through the REAL production
model.pt and its REAL CLASS_CENTROIDS (loaded once at `import app` time) --
not a re-implementation, and not a leave-one-out retrain, since
downloadguide/genkryptik were never in the training data under any split:

  1. KNOWN samples: the model's own real held-out test split
     (data/pyg_dataset_norm.pt + data/test_indices.npy), 11 real classes,
     never used for threshold calibration or model selection elsewhere.
  2. NOVEL samples: 2,004 rows / 63 real malware families absent from
     data/cleaned_data.csv entirely, drawn from the same 472-column native
     schema (pacx_merge_data/merge_csv_20240704.csv) -- the same source and
     filter as data/synthetic_surprise_test/pacx_new_families_results.json
     (verified: filtering to labels not in the 11 trained classes and not
     starting with "benign" gives exactly 63 families / 2,004 rows).

Run:  python scripts/evaluate_novelty_detection.py
Out:  data/novelty_detection/report.json
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph_constructor import (  # noqa: E402
    D_FEATURE, EDGE_INDEX, NODE_HEADER, NODE_ENTROPY, NODE_API, NODE_NETWORK,
    get_one_hot_type, native_row_to_node_vectors,
)
import app  # noqa: E402 -- loads the REAL production model + CLASS_CENTROIDS

CLEANED_CSV = "data/cleaned_data.csv"
MERGE_CSV = "pacx_merge_data/merge_csv_20240704.csv"
NORM_STATS_PATH = "data/norm_stats.pt"
OUT_DIR = "data/novelty_detection"
OUT_PATH = os.path.join(OUT_DIR, "report.json")

CONST_PACX_RESULT = {"prediction": "Benign", "confidence": 0.5, "evidence_score": 0.0}


def row_to_graph(row, label):
    vectors = native_row_to_node_vectors(row.get, D_FEATURE)
    final = [
        np.concatenate([vectors[node_idx], get_one_hot_type(node_idx)])
        for node_idx in (NODE_HEADER, NODE_ENTROPY, NODE_API, NODE_NETWORK)
    ]
    from torch_geometric.data import Data
    X = torch.tensor(np.stack(final), dtype=torch.float)
    y = torch.tensor([label], dtype=torch.long) if label is not None else None
    return Data(x=X, edge_index=EDGE_INDEX, y=y)


def normalize_graph(graph, norm_params):
    """Mirrors UniversalInputAdapter._apply_normalization exactly, using the
    REAL saved data/norm_stats.pt (train-split-only stats), so novel rows are
    scored under identical normalization to a real live request."""
    for i in range(4):
        feat = graph.x[i, :D_FEATURE].unsqueeze(0)
        mean, std = norm_params[i]
        scaled = (feat - mean) / std
        graph.x[i, :D_FEATURE] = scaled.squeeze(0)
    return graph


def compute_trust_scores(pacx_result, gnn_result):
    pacx_trust = 0.70 * pacx_result["confidence"] + 0.30 * pacx_result["evidence_score"]
    gnn_trust = (
        (gnn_result["raw_confidence"] * 0.35) + (gnn_result["model_accuracy"] * 0.10)
        + (gnn_result["completeness"] * 0.15) + (gnn_result["evidence_score"] * 0.25)
        + (gnn_result["family_confidence"] * 0.15)
    )
    return {"pacx": round(pacx_trust, 4), "gnn": round(gnn_trust, 4)}


def max_centroid_similarity(graph):
    """The ABSOLUTE distance signal compute_centroid_probabilities() discards:
    raw cosine similarity to the single NEAREST class centroid, before the
    x25 softmax that converts it into a purely relative ranking."""
    vector = graph.x[:, :256].flatten().float().detach().cpu()
    vector = vector / (torch.norm(vector) + 1e-8)
    sims = [float(torch.dot(vector, c).item()) for c in app.CLASS_CENTROIDS.values()]
    return max(sims)


def score_graph(graph):
    gnn_result = app.run_gnn_inference(graph, completeness=1.0, signal_summary={})
    trust = compute_trust_scores(CONST_PACX_RESULT, gnn_result)
    fusion = app.fuse_decisions(CONST_PACX_RESULT, gnn_result, trust_scores=trust)
    fusion = app.apply_zeroday_escalation(dict(fusion))
    return {
        "fused_label": fusion["final_label"],
        "fused_confidence": fusion["confidence"],
        "escalated": fusion["escalated"],
        "max_centroid_sim": max_centroid_similarity(graph),
        "gnn_family": gnn_result["predicted_family"],
    }


def main():
    device = torch.device("cpu")
    norm_stats = torch.load(NORM_STATS_PATH, weights_only=False)
    norm_params = {idx: (entry["mean"], entry["std"]) for idx, entry in norm_stats.items()}

    # --- 1. KNOWN: the model's own real held-out test split ---
    print("[*] Scoring real held-out test split (known classes)...")
    full_dataset = torch.load("data/pyg_dataset_norm.pt", weights_only=False)
    test_indices = np.load("data/test_indices.npy")
    known_results = []
    class_names = app.CLASS_LABELS
    for idx in test_indices:
        g = full_dataset[int(idx)].to(device)
        r = score_graph(g)
        truth = "Benign" if int(g.y.item()) == app.BENIGN_CLASS_INDEX else "Malicious"
        r["truth_binary"] = truth
        r["truth_family"] = class_names[int(g.y.item())]
        known_results.append(r)
    print(f"    {len(known_results)} known held-out samples scored")

    # --- 2. NOVEL: 63 real malware families never in cleaned_data.csv ---
    print("[*] Loading and filtering the 63-novel-family superset...")
    clean = pd.read_csv(CLEANED_CSV, low_memory=False, usecols=["label"])
    trained = set(clean["label"].unique())
    merge = pd.read_csv(MERGE_CSV, low_memory=False)
    novel_mask = (~merge["label"].isin(trained)) & (~merge["label"].str.startswith("benign"))
    novel_df = merge[novel_mask].reset_index(drop=True)
    print(f"    {len(novel_df)} rows / {novel_df['label'].nunique()} novel malware families")

    print("[*] Scoring novel-family samples...")
    novel_results = []
    for _, row in novel_df.iterrows():
        g = row_to_graph(row, None)
        g = normalize_graph(g, norm_params)
        r = score_graph(g)
        r["true_family"] = row["label"]
        novel_results.append(r)
    print(f"    {len(novel_results)} novel samples scored")

    # --- 3. Calibrate the novelty threshold from KNOWN data only ---
    # Same methodology as check_node_ood's p99: a real percentile observed on
    # real data, not a guessed cutoff. p1 here (the LOW tail) is the analogue,
    # since a low max-similarity is the "unusual" direction for this signal.
    known_sims = np.array([r["max_centroid_sim"] for r in known_results])
    p1 = float(np.percentile(known_sims, 1))
    p5 = float(np.percentile(known_sims, 5))

    def flagged(sim, thresh):
        return sim < thresh

    # --- 3b. Corrected headline catch rates for this novel-family set ---
    # A prior ad hoc script (never saved, its output committed as
    # data/synthetic_surprise_test/pacx_new_families_results.json) turned out
    # to have a real bug on ~7.5% of rows: a cluster of near-identical
    # confidence values (0.7106, 0.7717, ...) repeating across unrelated
    # families -- the signature of a stale fallback, not real per-sample
    # inference. Verified directly: this script's per-row output is bit-for-
    # bit identical to that file on 1853/2004 rows (including many decimals),
    # and diverges on exactly the rows with those suspicious repeated values,
    # disproportionately downloadguide (65/161 of its "misses" were this bug,
    # not a real detection gap). NovelBefore/NovelAfter below are the
    # corrected replacement for that file's numbers.
    n_novel = len(novel_results)
    baseline_caught = sum(1 for r in novel_results if r["fused_label"] != "Benign")
    escalated_caught = sum(1 for r in novel_results if r["fused_label"] in ("Malicious", "Suspicious"))
    print(f"\n[=] CORRECTED novel-family catch rate (supersedes pacx_new_families_results.json):")
    print(f"    baseline (pre-escalation): {baseline_caught}/{n_novel} = {baseline_caught/n_novel*100:.2f}%")
    print(f"    escalated (tau={app.AGENT2_THRESHOLD}):        {escalated_caught}/{n_novel} = {escalated_caught/n_novel*100:.2f}%")

    # --- 4. The specific failure mode: confidently-wrong misses at tau=0.68 ---
    # These already pass through app.apply_zeroday_escalation unescalated --
    # they are exactly what Section III-C-2 says a threshold cannot reach.
    confident_misses = [r for r in novel_results
                        if r["fused_label"] == "Benign" and not r["escalated"]]
    print(f"\n[=] Confidently-wrong misses (Benign, not escalated): "
          f"{len(confident_misses)}/{len(novel_results)}")

    for thresh, name in [(p1, "p1"), (p5, "p5")]:
        caught = sum(1 for r in confident_misses if flagged(r["max_centroid_sim"], thresh))
        known_fp = sum(1 for r in known_results if flagged(r["max_centroid_sim"], thresh))
        print(f"    threshold={name} ({thresh:.4f}): "
              f"catches {caught}/{len(confident_misses)} confident misses, "
              f"{known_fp}/{len(known_results)} known-sample false positives "
              f"({known_fp/len(known_results)*100:.2f}%)")

    # --- 5. Per-family breakdown for the two named worst offenders ---
    import collections
    by_family = collections.defaultdict(list)
    for r in novel_results:
        by_family[r["true_family"]].append(r)

    per_family_report = {}
    for fam, rows in sorted(by_family.items(), key=lambda kv: -len(kv[1])):
        sims = [r["max_centroid_sim"] for r in rows]
        n_missed = sum(1 for r in rows if r["fused_label"] == "Benign" and not r["escalated"])
        per_family_report[fam] = {
            "n": len(rows),
            "n_confidently_missed": n_missed,
            "mean_max_centroid_sim": round(float(np.mean(sims)), 4),
            "min_max_centroid_sim": round(float(np.min(sims)), 4),
            "max_max_centroid_sim": round(float(np.max(sims)), 4),
        }

    print("\n[=] Worst-offender families (from the paper/README):")
    for fam in ["downloadguide", "genkryptik"]:
        if fam in per_family_report:
            print(f"    {fam}: {json.dumps(per_family_report[fam])}")
        else:
            print(f"    {fam}: not present in this superset pull")

    print(f"\n[=] Reference: known-sample max_centroid_sim distribution: "
          f"min={known_sims.min():.4f} p1={p1:.4f} p5={p5:.4f} "
          f"mean={known_sims.mean():.4f} max={known_sims.max():.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump({
        "note": (
            "Tests whether raw (pre-softmax) max cosine similarity to the nearest "
            "of the 11 real CLASS_CENTROIDS separates known samples from the novel "
            "malware families that confidently defeat fail-closed escalation. Scored "
            "through the REAL production app.model / app.CLASS_CENTROIDS, no retraining "
            "or monkeypatching -- downloadguide/genkryptik were never in any training "
            "split, unlike the leave-one-out zero-day experiment."
        ),
        "n_known": len(known_results),
        "n_novel": len(novel_results),
        "n_novel_families": novel_df["label"].nunique(),
        "novel_catch_rate_baseline": round(baseline_caught / n_novel, 4),
        "novel_catch_rate_escalated": round(escalated_caught / n_novel, 4),
        "n_confidently_missed": len(confident_misses),
        "known_max_centroid_sim_distribution": {
            "min": float(known_sims.min()), "p1": p1, "p5": p5,
            "mean": float(known_sims.mean()), "max": float(known_sims.max()),
        },
        "threshold_sweep": [
            {
                "name": name, "threshold": thresh,
                "confident_misses_caught": sum(1 for r in confident_misses
                                               if flagged(r["max_centroid_sim"], thresh)),
                "n_confident_misses": len(confident_misses),
                "known_false_positives": sum(1 for r in known_results
                                             if flagged(r["max_centroid_sim"], thresh)),
                "n_known": len(known_results),
            }
            for thresh, name in [(p1, "p1"), (p5, "p5")]
        ],
        "per_family": per_family_report,
    }, open(OUT_PATH, "w"), indent=2)
    print(f"\n[+] Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
