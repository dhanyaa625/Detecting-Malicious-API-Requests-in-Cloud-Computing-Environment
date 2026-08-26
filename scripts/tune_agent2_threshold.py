"""
Properly tunes Agent 2's retraining-trigger threshold (currently a hardcoded,
never-derived 0.6 in app.py's `should_retrain = fusion_confidence < 0.6`)
using real data from ALL 10 leave-one-family-out zero-day experiments
(data/zeroday_<family>/), not just the 2 (emotet, darkkomet) run in the prior
session.

The prior session's zero-day numbers used raw top-1 GNN softmax confidence
as a stand-in for what Agent 2 actually gates on. That's not quite right --
the real gate is the FUSED decision confidence (app.py's fuse_decisions()),
which blends the GNN's own model+centroid+evidence-weighted binary score
with PAC-X's score, arbitrated by Agent 1's trust scores when they disagree.
This script replicates that exact live pipeline instead of approximating it:

- For native-schema input (what these CSV-derived samples are, and the only
  input type app.py's own accuracy claims are measured on), PAC-X's real
  output is provably CONSTANT: parse_result["text_summary"] is always "" for
  native_features input (core/input_adapter.py), so pacx_analyzer.analyze("")
  always returns prediction="Benign", confidence=0.5, evidence_score=0.0,
  regardless of the sample. That makes PAC-X's trust score constant too
  (0.5*0.7 + 0.0*0.3 = 0.35) -- computed once, not re-derived per sample as
  a shortcut, but because it provably cannot vary for this input type.
- The GNN side is NOT constant and NOT just top-1 softmax: it blends the
  model's own softmax, a class-centroid similarity classifier (rebuilt here
  from each zero-day run's own known-class TRAIN split, mirroring app.py's
  load_class_centroids -- centroids must never see the held-out family
  either, matching the zero-day premise), and an evidence score (always 0
  for native input, since signal_summary lacks suspicious/api/network
  counts for this input type).
- Rather than hand-transcribing app.py's blend formula (real risk of a
  transcription bug silently invalidating the whole exercise), this script
  monkey-patches app.py's module-level `model`/`CLASS_CENTROIDS`/
  `CLASS_LABELS`/`BENIGN_CLASS_INDEX`/`MODEL_OVERALL_ACCURACY` to point at
  each zero-day run's own artifacts, then calls app.run_gnn_inference() and
  app.fuse_decisions() directly -- the exact same functions app.py's live
  /api/analyze/live endpoint calls. Agent 1's trust-score arithmetic is
  copied from agents/analyst_agent.py's ComparisonAgent.compare_results
  (not called directly, to avoid triggering ~9,000 real LLM calls for
  reasoning text that isn't needed here).

For every known-class TEST sample (correctly and incorrectly classified)
across all 10 runs, plus every held-out zero-day sample, computes the real
fused confidence, then sweeps candidate thresholds to show the actual
tradeoff: how much zero-day coverage a given threshold buys vs. how often
it needlessly re-flags already-correct in-distribution predictions.

Usage: python scripts/tune_agent2_threshold.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app  # noqa: E402 -- reuses run_gnn_inference / fuse_decisions / clamp verbatim
from agentic_pacx.gnn_classification import MalwareGAT  # noqa: E402
from scripts.build_zeroday_experiment import (  # noqa: E402
    row_to_graph, stratified_split_3way, apply_normalize, compute_norm_stats,
)

CONST_PACX_RESULT = {"prediction": "Benign", "confidence": 0.5, "evidence_score": 0.0}
CONST_PACX_TRUST = round((0.5 * 0.7) + (0.0 * 0.3), 4)  # 0.35, provably constant for native input


def build_centroids(train_dataset):
    """Mirrors app.py's load_class_centroids(), scoped to one zero-day run's
    own known-class TRAIN split (never the held-out family)."""
    grouped = {}
    for g in train_dataset:
        label = int(g.y.item())
        grouped.setdefault(label, []).append(g.x[:, :256].flatten().float())
    centroids = {}
    for label, vecs in grouped.items():
        stacked = torch.stack(vecs)
        c = stacked.mean(dim=0)
        c = c / (torch.norm(c) + 1e-8)
        centroids[label] = c
    return centroids


def compute_trust_scores(pacx_result, gnn_result):
    """Verbatim arithmetic from agents/analyst_agent.py's
    ComparisonAgent.compare_results (LLM reasoning generation intentionally
    skipped -- not needed for the trust-score numbers this depends on, and
    calling it per-sample would fire ~9,000 real LLM requests)."""
    pacx_trust = (pacx_result.get('confidence', 0) * 0.7) + (pacx_result.get('evidence_score', 0) * 0.3)
    gnn_raw_conf = gnn_result.get('raw_confidence', gnn_result.get('confidence', 0.0))
    gnn_family_conf = gnn_result.get('family_confidence', 0.0)
    gnn_model_accuracy = gnn_result.get('model_accuracy', gnn_raw_conf)
    gnn_completeness = gnn_result.get('completeness', 1.0)
    gnn_evidence = gnn_result.get('evidence_score', gnn_result.get('confidence', 0.0))
    gnn_trust = (
        (gnn_raw_conf * 0.35) + (gnn_model_accuracy * 0.10) + (gnn_completeness * 0.15)
        + (gnn_evidence * 0.25) + (gnn_family_conf * 0.15)
    )
    return {"pacx": round(pacx_trust, 4), "gnn": round(gnn_trust, 4)}


def run_one_family(fam, df, device):
    report_path = f"data/zeroday_{fam}/zeroday_report.json"
    model_path = f"data/zeroday_{fam}/model.pt"
    if not (os.path.exists(report_path) and os.path.exists(model_path)):
        print(f"[!] Skipping {fam}: no saved zero-day run found (run scripts/build_zeroday_experiment.py --holdout {fam} first)")
        return None
    report = json.load(open(report_path))
    model_acc = report["known_class_test_accuracy"] / 100.0

    known_df = df[df["label"] != fam].reset_index(drop=True)
    heldout_df = df[df["label"] == fam].reset_index(drop=True)
    class_names = sorted(known_df["label"].unique().tolist())
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}
    benign_idx = label_to_idx["benign"]

    known_graphs = [row_to_graph(row, label_to_idx[row["label"]]) for _, row in known_df.iterrows()]
    heldout_graphs = [row_to_graph(row, None) for _, row in heldout_df.iterrows()]

    labels = [int(g.y.item()) for g in known_graphs]
    train_idx, val_idx, test_idx = stratified_split_3way(labels, seed=42)  # same seed as build_zeroday_experiment
    norm_stats = compute_norm_stats(known_graphs, train_idx)
    known_graphs = apply_normalize(known_graphs, norm_stats)
    heldout_graphs = apply_normalize(heldout_graphs, norm_stats)

    train_dataset = [known_graphs[i] for i in train_idx]
    test_dataset = [known_graphs[i] for i in test_idx]

    model = MalwareGAT(num_node_features=260, num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    centroids = build_centroids(train_dataset)

    # Monkey-patch app.py's module globals so run_gnn_inference/fuse_decisions
    # (both real, live, unmodified functions) operate on THIS zero-day run's
    # own model/centroids/labels instead of the main live model.
    app.model = model
    app.CLASS_LABELS = class_names
    app.BENIGN_CLASS_INDEX = benign_idx
    app.MODEL_OVERALL_ACCURACY = model_acc
    app.CLASS_CENTROIDS = centroids

    def fused_confidence_and_label(graph):
        graph = graph.to(device)
        gnn_result = app.run_gnn_inference(graph, completeness=1.0, signal_summary={})
        trust_scores = compute_trust_scores(CONST_PACX_RESULT, gnn_result)
        fusion = app.fuse_decisions(CONST_PACX_RESULT, gnn_result, trust_scores=trust_scores)
        return fusion["confidence"], fusion["final_label"]

    known_correct, known_incorrect, zeroday = [], [], []
    for g in test_dataset:
        truth = "Benign" if int(g.y.item()) == benign_idx else "Malicious"
        conf, label = fused_confidence_and_label(g)
        (known_correct if label == truth else known_incorrect).append(conf)
    for g in heldout_graphs:
        conf, label = fused_confidence_and_label(g)
        zeroday.append((conf, label))

    print(f"[+] {fam:12s}: {len(test_dataset)} known-test ({len(known_correct)} correct / {len(known_incorrect)} wrong), "
          f"{len(heldout_graphs)} zero-day held out")
    return {"known_correct": known_correct, "known_incorrect": known_incorrect, "zeroday": zeroday}


def main():
    device = torch.device("cpu")
    df = pd.read_csv("data/cleaned_data.csv", low_memory=False)
    df["label"] = df["label"].str.lower()
    families = sorted(l for l in df["label"].unique() if l != "benign")

    pooled = {"known_correct": [], "known_incorrect": [], "zeroday": []}
    per_family = {}
    for fam in families:
        result = run_one_family(fam, df, device)
        if result is None:
            continue
        per_family[fam] = {
            "n_known_correct": len(result["known_correct"]),
            "n_known_incorrect": len(result["known_incorrect"]),
            "n_zeroday": len(result["zeroday"]),
            "zeroday_caught_directly": sum(1 for _, lbl in result["zeroday"] if lbl == "Malicious"),
        }
        pooled["known_correct"].extend(result["known_correct"])
        pooled["known_incorrect"].extend(result["known_incorrect"])
        pooled["zeroday"].extend(result["zeroday"])

    n_known_correct = len(pooled["known_correct"])
    n_known_incorrect = len(pooled["known_incorrect"])
    n_zeroday = len(pooled["zeroday"])
    print(f"\n[+] Pooled across {len(per_family)} families: {n_known_correct} known-correct, "
          f"{n_known_incorrect} known-incorrect, {n_zeroday} zero-day samples")

    known_correct_arr = np.array(pooled["known_correct"])
    known_incorrect_arr = np.array(pooled["known_incorrect"])
    zeroday_confs = np.array([c for c, _ in pooled["zeroday"]])
    zeroday_caught_directly = np.array([lbl == "Malicious" for _, lbl in pooled["zeroday"]])

    sweep = []
    for tau in np.arange(0.50, 1.0001, 0.01):
        false_trigger_rate = float((known_correct_arr < tau).mean()) if n_known_correct else 0.0
        zeroday_handled = zeroday_caught_directly | (zeroday_confs < tau)
        zeroday_handled_rate = float(zeroday_handled.mean()) if n_zeroday else 0.0
        known_incorrect_caught_rate = float((known_incorrect_arr < tau).mean()) if n_known_incorrect else 0.0
        sweep.append({
            "tau": round(float(tau), 2),
            "zeroday_handled_rate": round(100 * zeroday_handled_rate, 2),
            "known_correct_false_trigger_rate": round(100 * false_trigger_rate, 2),
            "known_incorrect_caught_rate": round(100 * known_incorrect_caught_rate, 2),
            "objective": round(zeroday_handled_rate - false_trigger_rate, 4),
        })

    best = max(sweep, key=lambda r: r["objective"])
    baseline = next(r for r in sweep if r["tau"] == 0.60)

    print("\n=== Baseline (current hardcoded tau=0.60) ===")
    print(json.dumps(baseline, indent=2))
    print("\n=== Best by (zero-day handled - false trigger) objective ===")
    print(json.dumps(best, indent=2))

    report = {
        "note": (
            "Real fused-confidence distributions (app.py's actual run_gnn_inference + "
            "fuse_decisions pipeline, replicated exactly via monkey-patched globals -- not "
            "an approximation) across all 10 leave-one-family-out zero-day experiments "
            "(data/zeroday_<family>/), for native-schema input. PAC-X's contribution is "
            "provably constant for this input type (empty text_summary -> fixed 0.5 "
            "confidence / 0.0 evidence / 0.35 trust), so the fused confidence variation "
            "shown here comes entirely from the GNN's model+centroid+evidence blend."
        ),
        "n_families": len(per_family),
        "families": sorted(per_family.keys()),
        "n_known_correct": n_known_correct,
        "n_known_incorrect": n_known_incorrect,
        "n_zeroday": n_zeroday,
        "per_family": per_family,
        "threshold_sweep": sweep,
        "current_hardcoded_threshold": baseline,
        "best_by_objective": best,
    }
    os.makedirs("data/agent2_tuning", exist_ok=True)
    with open("data/agent2_tuning/report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\n[+] Saved data/agent2_tuning/report.json")


if __name__ == "__main__":
    main()
