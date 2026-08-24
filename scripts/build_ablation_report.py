"""
Ablation study: GNN-only vs PAC-X-only vs Fused (with trust arbitration),
evaluated on the same real held-out test split (data/test_indices.npy).

Honesty note: PACXHeuristicAnalyzer only has signal on raw text/API-call
logs. This project's dataset is PE-header CSV features, so PAC-X is run on
the CSV serialization of each row -- its structurally weaker modality (see
core/pacx_analyzer.py's input_type=="csv" confidence cap). This ablation
therefore measures whether fusion still helps (or at least doesn't hurt) on
the dataset's actual native format, not PAC-X's best-case input.

Usage: python scripts/build_ablation_report.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app
from agents.analyst_agent import ComparisonAgent


def binary_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred))) * 100
    return {
        "accuracy": round(accuracy, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
    }


def main():
    dataset = torch.load("data/pyg_dataset_norm.pt", weights_only=False)
    test_indices = np.load("data/test_indices.npy")
    df = pd.read_csv("data/cleaned_data.csv", low_memory=False)

    agent1 = ComparisonAgent()

    y_true, y_gnn, y_pacx, y_fused = [], [], [], []

    for n, idx in enumerate(test_indices):
        idx = int(idx)
        graph = dataset[idx]
        actual_label = int(graph.y.item())
        y_true.append(0 if actual_label == app.BENIGN_CLASS_INDEX else 1)

        csv_text = df.iloc[[idx]].to_csv(index=False)

        gnn_result = app.run_gnn_inference(graph.to(app.device), completeness=1.0, signal_summary={})
        raw_pacx = app.pacx_engine.analyze(csv_text, input_type="csv")
        pacx_result = {
            "prediction": raw_pacx["prediction"],
            "confidence": raw_pacx["confidence"],
            "findings": raw_pacx["findings"],
            "evidence_score": round(
                min(1.0, len(raw_pacx.get("detected_apis", [])) * 0.2 +
                    len(raw_pacx.get("detected_patterns", [])) * 0.15 +
                    (0.15 if raw_pacx.get("entropy", 0) >= 6.5 else 0.0)), 4
            ),
        }

        agent1_analysis = agent1.compare_results(pacx_result, gnn_result)
        fusion = app.fuse_decisions(
            pacx_result, gnn_result,
            trust_scores=agent1_analysis["agent1_decision"]["trust_scores"],
        )

        y_gnn.append(1 if gnn_result["prediction"] == "Malicious" else 0)
        y_pacx.append(1 if pacx_result["prediction"] == "Malicious" else 0)
        y_fused.append(1 if fusion["final_label"] == "Malicious" else 0)

        if (n + 1) % 100 == 0:
            print(f"[*] Processed {n + 1}/{len(test_indices)}")

    report = {
        "dataset_size": len(test_indices),
        "note": (
            "PAC-X evaluated on CSV serialization of each row (its weaker modality -- "
            "no raw text/API logs exist in this dataset). GNN evaluated on its native "
            "trained feature space. Fused uses the real decision-fusion + trust arbitration."
        ),
        "gnn_only": binary_metrics(y_true, y_gnn),
        "pacx_only": binary_metrics(y_true, y_pacx),
        "fused": binary_metrics(y_true, y_fused),
    }

    with open("ablation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
