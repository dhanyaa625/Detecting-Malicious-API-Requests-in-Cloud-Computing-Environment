"""
Measures the real cost/benefit of FAILING CLOSED on zero-day malware.

The problem this quantifies
---------------------------
A zero-day is, by definition, an attack. But the pipeline's final label is the
argmax of a classifier that has never seen the family, so a novel sample lands
on whichever known class is nearest -- overwhelmingly "benign". The result is a
confident-looking "Benign" verdict on real malware: the system fails OPEN.

Crucially, the system already KNOWS these samples are doubtful -- the fused
confidence sits below AGENT2_THRESHOLD and Agent 2 fires. The information is
there; the verdict just ignores it. On the 56-sample novel-family set, Agent 2
flagged 56/56 while the label caught only 17/56.

The policy under test
---------------------
    if fused_confidence < threshold and label == "Benign":
        label = "Suspicious"        # escalate rather than clear

This is strictly a LABELLING policy on top of the existing pipeline. It changes
no weights and retrains nothing, so it cannot regress the model.

What is measured (all via app.run_gnn_inference + app.fuse_decisions -- the
real live functions, monkey-patched onto each zero-day run's own model, exactly
as scripts/tune_agent2_threshold.py does):

  benign_fp_rate : real BENIGN test samples wrongly escalated  (the cost)
  known_mal_rate : real MALICIOUS known-family samples caught  (before/after)
  zeroday_rate   : held-out NOVEL-family samples caught        (the benefit)

Run:  python scripts/evaluate_zeroday_escalation.py
Out:  data/agent2_tuning/escalation_report.json
"""
import os
import sys
import json

import torch
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.tune_agent2_threshold import (  # noqa: E402
    build_centroids,
    compute_trust_scores,
    CONST_PACX_RESULT,
)
from scripts.build_zeroday_experiment import (  # noqa: E402
    row_to_graph,
    stratified_split_3way,
    compute_norm_stats,
    apply_normalize,
)
from agentic_pacx.gnn_classification import MalwareGAT  # noqa: E402
import app  # noqa: E402

DATA_CSV = "data/cleaned_data.csv"
OUT_PATH = "data/agent2_tuning/escalation_report.json"
SWEEP = [0.55, 0.60, 0.64, 0.68, 0.72, 0.76, 0.80]


def collect_one_family(fam, df, device):
    """Real fused (confidence, label, truth) triples for one zero-day run."""
    report_path = f"data/zeroday_{fam}/zeroday_report.json"
    model_path = f"data/zeroday_{fam}/model.pt"
    if not (os.path.exists(report_path) and os.path.exists(model_path)):
        print(f"[!] Skipping {fam}: no saved zero-day run found.")
        return None

    report = json.load(open(report_path))
    model_acc = report["known_class_test_accuracy"] / 100.0

    known_df = df[df["label"] != fam].reset_index(drop=True)
    heldout_df = df[df["label"] == fam].reset_index(drop=True)
    class_names = sorted(known_df["label"].unique().tolist())
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}
    benign_idx = label_to_idx["benign"]

    known_graphs = [row_to_graph(r, label_to_idx[r["label"]]) for _, r in known_df.iterrows()]
    heldout_graphs = [row_to_graph(r, None) for _, r in heldout_df.iterrows()]

    labels = [int(g.y.item()) for g in known_graphs]
    train_idx, _, test_idx = stratified_split_3way(labels, seed=42)
    norm_stats = compute_norm_stats(known_graphs, train_idx)
    known_graphs = apply_normalize(known_graphs, norm_stats)
    heldout_graphs = apply_normalize(heldout_graphs, norm_stats)

    train_dataset = [known_graphs[i] for i in train_idx]
    test_dataset = [known_graphs[i] for i in test_idx]

    model = MalwareGAT(num_node_features=260, num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # Same monkey-patch as the threshold tuner: the REAL live fusion functions,
    # pointed at this run's own model/centroids/labels.
    app.model = model
    app.CLASS_LABELS = class_names
    app.BENIGN_CLASS_INDEX = benign_idx
    app.MODEL_OVERALL_ACCURACY = model_acc
    app.CLASS_CENTROIDS = build_centroids(train_dataset)

    def fused(graph):
        graph = graph.to(device)
        gnn_result = app.run_gnn_inference(graph, completeness=1.0, signal_summary={})
        trust = compute_trust_scores(CONST_PACX_RESULT, gnn_result)
        f = app.fuse_decisions(CONST_PACX_RESULT, gnn_result, trust_scores=trust)
        return float(f["confidence"]), f["final_label"]

    benign, known_mal, zeroday = [], [], []
    for g in test_dataset:
        conf, label = fused(g)
        (benign if int(g.y.item()) == benign_idx else known_mal).append((conf, label))
    for g in heldout_graphs:
        zeroday.append(fused(g))

    print(f"[+] {fam:12s}: {len(benign):4d} benign, {len(known_mal):4d} known-malicious, "
          f"{len(zeroday):4d} zero-day")
    return {"benign": benign, "known_mal": known_mal, "zeroday": zeroday}


def caught(pairs, threshold, escalate):
    """Fraction flagged non-Benign under the given policy."""
    if not pairs:
        return 0.0
    n = 0
    for conf, label in pairs:
        flagged = label != "Benign"
        if escalate and label == "Benign" and conf < threshold:
            flagged = True
        n += flagged
    return n / len(pairs)


def main():
    device = torch.device("cpu")
    df = pd.read_csv(DATA_CSV, low_memory=False)
    families = sorted(
        d.replace("data/zeroday_", "").rstrip("/")
        for d in [p.replace("\\", "/") for p in
                  [os.path.join("data", x) for x in os.listdir("data")]]
        if "zeroday_" in d and os.path.isdir(d)
    )

    benign, known_mal, zeroday = [], [], []
    for fam in families:
        got = collect_one_family(fam, df, device)
        if not got:
            continue
        benign += got["benign"]
        known_mal += got["known_mal"]
        zeroday += got["zeroday"]

    print(f"\n[=] Pooled: {len(benign)} benign, {len(known_mal)} known-malicious, "
          f"{len(zeroday)} zero-day\n")

    rows = []
    print(f"{'thresh':>7} | {'benign FP':>10} | {'known-mal':>10} | {'zero-day':>9}")
    print("-" * 46)
    base_b = caught(benign, 0, False)
    base_k = caught(known_mal, 0, False)
    base_z = caught(zeroday, 0, False)
    print(f"{'OFF':>7} | {base_b*100:9.2f}% | {base_k*100:9.2f}% | {base_z*100:8.2f}%")
    for t in SWEEP:
        b = caught(benign, t, True)
        k = caught(known_mal, t, True)
        z = caught(zeroday, t, True)
        rows.append({
            "threshold": t,
            "benign_false_positive_rate": round(b, 4),
            "known_malicious_catch_rate": round(k, 4),
            "zeroday_catch_rate": round(z, 4),
        })
        print(f"{t:7.2f} | {b*100:9.2f}% | {k*100:9.2f}% | {z*100:8.2f}%")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    json.dump({
        "note": (
            "Real measurement of the fail-closed escalation policy (low-confidence "
            "'Benign' -> 'Suspicious') using app.run_gnn_inference + app.fuse_decisions "
            "verbatim across all 10 leave-one-family-out zero-day runs. 'OFF' is today's "
            "behaviour. benign_false_positive_rate is the cost; zeroday_catch_rate is the "
            "benefit."
        ),
        "n_benign": len(benign),
        "n_known_malicious": len(known_mal),
        "n_zeroday": len(zeroday),
        "baseline_no_escalation": {
            "benign_false_positive_rate": round(base_b, 4),
            "known_malicious_catch_rate": round(base_k, 4),
            "zeroday_catch_rate": round(base_z, 4),
        },
        "sweep": rows,
    }, open(OUT_PATH, "w"), indent=2)
    print(f"\n[+] Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
