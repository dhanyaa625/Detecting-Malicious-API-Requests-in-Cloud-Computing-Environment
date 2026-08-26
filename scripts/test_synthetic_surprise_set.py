"""
One-off verification script: tests the real live pipeline against all three
serializations of the same 500-sample synthetic "blind" PE static-analysis
benchmark the user generated (263 benign / 237 malicious, 201 flagged
zero_day) -- pe_dataset_surprise.txt (plain API list), .son (a JSON array,
despite the extension), and .csv (structured numeric PE-header fields +
semicolon-packed imported_apis column). Each format is fed through the exact
input_type a real user would pick for it (api/json/csv) via the same
real code path (core/pacx_analyzer.py + app.run_gnn_inference +
app.fuse_decisions -- unmodified functions, not reimplemented), skipping
only Agent 1's LLM narrative generation (pure descriptive text, doesn't
affect the verdict -- computes trust_scores with the identical formula
instead, to avoid 1500 real LLM calls for an accuracy check that doesn't
need them).

Usage: python scripts/test_synthetic_surprise_set.py
"""
import csv as csv_module
import io
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app  # noqa: E402

TXT_PATH = r"C:\Users\dhany\Downloads\pe_dataset_surprise.txt"
JSON_PATH = r"C:\Users\dhany\Downloads\pe_dataset_surprise.son"
CSV_PATH = r"C:\Users\dhany\Downloads\pe_dataset_surprise.csv"


def parse_txt_samples(path):
    text = open(path, encoding="utf-8").read()
    blocks = re.split(r"-{10,}", text)
    samples = []
    for block in blocks:
        m_id = re.search(r"\[(S\d+)\]\s+(\S+)", block)
        if not m_id:
            continue
        label = re.search(r"label:\s*(\w+)", block).group(1)
        zero_day = re.search(r"zero_day:\s*(\w+)", block).group(1) == "True"
        apis_m = re.search(r"imported_apis:\s*(.+)", block)
        apis = [a.strip() for a in apis_m.group(1).split(",")] if apis_m else []
        samples.append({"id": m_id.group(1), "label": label, "zero_day": zero_day,
                         "content": ", ".join(apis), "input_type": "api"})
    return samples


def parse_json_samples(path):
    records = json.load(open(path, encoding="utf-8"))
    samples = []
    for rec in records:
        samples.append({
            "id": rec["sample_id"], "label": rec["label"], "zero_day": bool(rec["is_zero_day"]),
            "content": json.dumps(rec), "input_type": "json",
        })
    return samples


def parse_csv_samples(path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv_module.DictReader(f)
        rows = list(reader)
    header = list(rows[0].keys()) if rows else []
    samples = []
    for row in rows:
        buf = io.StringIO()
        writer = csv_module.DictWriter(buf, fieldnames=header)
        writer.writeheader()
        writer.writerow(row)
        samples.append({
            "id": row["sample_id"], "label": row["label"], "zero_day": row["is_zero_day"] == "True",
            "content": buf.getvalue(), "input_type": "csv",
        })
    return samples


def compute_trust_scores(pacx_result, gnn_result):
    """Verbatim from agents/analyst_agent.py's ComparisonAgent.compare_results
    (reasoning-generation step skipped -- doesn't affect trust_scores)."""
    pacx_conf = pacx_result.get('confidence', 0)
    pacx_evidence = pacx_result.get('evidence_score', pacx_conf)
    pacx_trust = (pacx_conf * 0.7) + (pacx_evidence * 0.3)

    gnn_conf = gnn_result.get('confidence', 0)
    gnn_completeness = gnn_result.get('completeness', 1.0)
    gnn_evidence = gnn_result.get('evidence_score', gnn_conf)
    gnn_raw_conf = gnn_result.get('raw_confidence', gnn_conf)
    gnn_family_conf = gnn_result.get('family_confidence', 0.0)
    gnn_model_accuracy = gnn_result.get('model_accuracy', gnn_raw_conf)
    gnn_trust = (
        (gnn_raw_conf * 0.35) + (gnn_model_accuracy * 0.10) + (gnn_completeness * 0.15)
        + (gnn_evidence * 0.25) + (gnn_family_conf * 0.15)
    )
    return {"pacx": round(pacx_trust, 4), "gnn": round(gnn_trust, 4)}


def analyze_one(content, input_type="api"):
    parse_result = app.adapter.parse(content, input_type=input_type)
    graph = parse_result["graph"].to(app.device)
    completeness = parse_result.get("completeness", 1.0)
    signal_summary = parse_result.get("signal_summary", {})

    gnn_result = app.run_gnn_inference(graph, completeness, signal_summary)

    pacx_content = parse_result.get("text_summary", "") if input_type in ("pe_binary", "native_features") else content
    raw_pacx = app.pacx_engine.analyze(pacx_content, input_type=input_type)
    pacx_result = {
        "prediction": raw_pacx["prediction"],
        "confidence": raw_pacx["confidence"],
        "evidence_score": round(min(1.0,
            len(raw_pacx.get("detected_apis", [])) * 0.2
            + len(raw_pacx.get("detected_patterns", [])) * 0.15
            + (0.15 if raw_pacx.get("entropy", 0) >= 6.5 else 0.0)), 4),
    }

    trust_scores = compute_trust_scores(pacx_result, gnn_result)
    fusion = app.fuse_decisions(pacx_result, gnn_result, trust_scores=trust_scores)
    return {
        "pacx_label": pacx_result["prediction"],
        "gnn_label": gnn_result["prediction"],
        "fused_label": fusion["final_label"],
        "fused_confidence": fusion["confidence"],
        "gnn_family": gnn_result["predicted_family"],
    }


def acc(rows, key, truth_to_ours):
    if not rows:
        return None
    correct = sum(1 for r in rows if r[key] == truth_to_ours[r["true_label"]])
    return round(100 * correct / len(rows), 2)


def run_format(name, samples):
    truth_to_ours = {"benign": "Benign", "malicious": "Malicious"}
    results = []
    for i, s in enumerate(samples):
        with torch.no_grad():
            r = analyze_one(s["content"], input_type=s["input_type"])
        r.update({"id": s["id"], "true_label": s["label"], "zero_day": s["zero_day"]})
        results.append(r)
        if (i + 1) % 100 == 0:
            print(f"    ...{i+1}/{len(samples)}")

    zd = [r for r in results if r["zero_day"]]
    known = [r for r in results if not r["zero_day"]]
    summary = {
        "format": name,
        "n_total": len(results),
        "overall": {"pacx": acc(results, "pacx_label", truth_to_ours), "gnn": acc(results, "gnn_label", truth_to_ours), "fused": acc(results, "fused_label", truth_to_ours)},
        "zero_day_flagged": {"n": len(zd), "pacx": acc(zd, "pacx_label", truth_to_ours), "gnn": acc(zd, "gnn_label", truth_to_ours), "fused": acc(zd, "fused_label", truth_to_ours)},
        "non_zero_day": {"n": len(known), "pacx": acc(known, "pacx_label", truth_to_ours), "gnn": acc(known, "gnn_label", truth_to_ours), "fused": acc(known, "fused_label", truth_to_ours)},
    }
    print(f"\n=== {name} ===")
    print(json.dumps(summary, indent=2))
    return summary, results


def main():
    formats = [
        ("txt (api text)", parse_txt_samples(TXT_PATH)),
        ("son (json)", parse_json_samples(JSON_PATH)),
        ("csv", parse_csv_samples(CSV_PATH)),
    ]

    all_summaries = {}
    all_results = {}
    for name, samples in formats:
        print(f"\n[+] {name}: {len(samples)} samples "
              f"({sum(1 for s in samples if s['label']=='benign')} benign / "
              f"{sum(1 for s in samples if s['label']=='malicious')} malicious, "
              f"{sum(1 for s in samples if s['zero_day'])} flagged zero_day)")
        summary, results = run_format(name, samples)
        all_summaries[name] = summary
        all_results[name] = results

    os.makedirs("data/synthetic_surprise_test", exist_ok=True)
    with open("data/synthetic_surprise_test/report.json", "w") as f:
        json.dump({"summaries": all_summaries, "results": all_results}, f, indent=2)
    print("\n[+] Saved data/synthetic_surprise_test/report.json")


if __name__ == "__main__":
    main()
