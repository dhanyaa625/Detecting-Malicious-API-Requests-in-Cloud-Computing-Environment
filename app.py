import os
import sys
import json
import threading
import matplotlib
matplotlib.use('Agg') # Thread-safe backend for server environments

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import DataLoader
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from typing import Optional

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "agentic_pacx"))

from agentic_pacx.gnn_classification import MalwareGAT
from agents.analyst_agent import generate_ui_reports, trigger_agent_2_retraining
from core.utils.env_check import verify_environment
from core.utils.dataset_audit import audit_dataset
from core.utils.visualizer import generate_pictorial_graph
from core.input_adapter import UniversalInputAdapter
from core.pacx_analyzer import PACXHeuristicAnalyzer


def load_class_labels():
    csv_path = os.path.join("data", "cleaned_data.csv")
    if not os.path.exists(csv_path):
        return ["benign", "malicious"]

    try:
        df = pd.read_csv(csv_path, usecols=["label"], low_memory=False)
        labels = sorted(str(label) for label in df["label"].dropna().unique().tolist())
        return labels or ["benign", "malicious"]
    except Exception:
        return ["benign", "malicious"]


CLASS_LABELS = load_class_labels()
BENIGN_CLASS_INDEX = CLASS_LABELS.index("benign") if "benign" in CLASS_LABELS else 0


def get_graph_batch(graph):
    batch = getattr(graph, "batch", None)
    if batch is not None:
        return batch
    return torch.zeros(graph.x.size(0), dtype=torch.long, device=graph.x.device)


def label_name_for(index):
    if 0 <= index < len(CLASS_LABELS):
        return CLASS_LABELS[index]
    return f"class_{index}"


def load_class_centroids():
    dataset_path = 'data/pyg_dataset_norm.pt' if os.path.exists('data/pyg_dataset_norm.pt') else 'data/pyg_dataset.pt'
    if not os.path.exists(dataset_path):
        return {}

    try:
        dataset = torch.load(dataset_path, weights_only=False)

        # Centroids must only be built from training samples: including test-set
        # embeddings here would let a held-out sample's own vector contribute to
        # the centroid it's later compared against (train/test leakage).
        train_index_path = os.path.join("data", "train_indices.npy")
        if os.path.exists(train_index_path):
            train_indices = np.load(train_index_path)
            dataset = [dataset[int(idx)] for idx in train_indices]

        grouped = {}
        for sample in dataset:
            if sample.y is None:
                continue
            label = int(sample.y.item())
            grouped.setdefault(label, []).append(sample.x[:, :256].flatten().float())

        centroids = {}
        for label, vectors in grouped.items():
            stacked = torch.stack(vectors)
            centroid = stacked.mean(dim=0)
            centroid = centroid / (torch.norm(centroid) + 1e-8)
            centroids[label] = centroid
        return centroids
    except Exception:
        return {}


CLASS_CENTROIDS = load_class_centroids()


def compute_centroid_probabilities(graph):
    if not CLASS_CENTROIDS:
        return {}

    vector = graph.x[:, :256].flatten().float().detach().cpu()
    vector = vector / (torch.norm(vector) + 1e-8)

    similarities = {}
    for label, centroid in CLASS_CENTROIDS.items():
        similarities[label] = float(torch.dot(vector, centroid).item())

    sim_tensor = torch.tensor(list(similarities.values()), dtype=torch.float32)
    prob_tensor = torch.softmax(sim_tensor * 25.0, dim=0)

    return {
        label: float(prob_tensor[idx].item())
        for idx, label in enumerate(similarities.keys())
    }


def clamp(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(max_value, value))


def compute_model_certainty(probs_row):
    """
    Estimate certainty from both absolute confidence and class margin.
    """
    if probs_row.numel() == 0:
        return 0.0

    top_k = min(2, probs_row.numel())
    top_scores, _ = torch.topk(probs_row, k=top_k)
    top1 = float(top_scores[0].item())
    top2 = float(top_scores[1].item()) if top_k > 1 else 0.0
    margin = clamp(top1 - top2, 0.0, 1.0)
    return clamp((0.6 * top1) + (0.4 * margin), 0.0, 1.0)


def pacx_malicious_score(pacx_result):
    prediction = str(pacx_result.get("prediction", "Benign")).lower()
    confidence = clamp(float(pacx_result.get("confidence", 0.0)))
    return confidence if prediction == "malicious" else (1.0 - confidence)


def fuse_decisions(pacx_result, gnn_result, trust_scores=None):
    """
    Decision fusion layer combining PAC-X and GNN outputs.
    When both paths agree, blend their scores (reliability-weighted).
    When they disagree, defer to whichever path Agent 1 scored as more
    trustworthy for this sample instead of averaging the disagreement away.
    """
    pacx_score = pacx_malicious_score(pacx_result)
    gnn_score = clamp(float(gnn_result.get("binary_scores", {}).get("malicious", 0.0)))
    completeness = clamp(float(gnn_result.get("completeness", 1.0)))
    model_certainty = clamp(float(gnn_result.get("blend_weights", {}).get("model_certainty", 0.0)))

    pacx_pred = str(pacx_result.get("prediction", "Benign"))
    gnn_pred = str(gnn_result.get("prediction", "Benign"))
    agreement = pacx_pred == gnn_pred

    # Base fusion requested by design: 0.4 PAC-X + 0.6 GNN.
    gnn_weight = 0.6
    pacx_weight = 0.4

    # Reliability-aware tuning around the baseline.
    reliability_bonus = 0.2 * ((0.5 * completeness) + (0.5 * model_certainty))
    gnn_weight = clamp(gnn_weight + reliability_bonus, 0.5, 0.8)
    pacx_weight = 1.0 - gnn_weight

    if agreement or not trust_scores:
        final_score = clamp((pacx_weight * pacx_score) + (gnn_weight * gnn_score))
        final_label = "Malicious" if final_score >= 0.5 else "Benign"
        arbitration = "fused"
    else:
        winner = "gnn" if trust_scores.get("gnn", 0.0) >= trust_scores.get("pacx", 0.0) else "pacx"
        final_score = gnn_score if winner == "gnn" else pacx_score
        final_label = gnn_pred if winner == "gnn" else pacx_pred
        arbitration = f"arbitrated_by_trust:{winner}"

    return {
        "final_label": final_label,
        "final_score": round(final_score, 4),
        "confidence": round(final_score if final_label == "Malicious" else (1.0 - final_score), 4),
        "threshold": 0.5,
        "agreement": agreement,
        "arbitration": arbitration,
        "weights": {
            "pacx": round(pacx_weight, 4),
            "gnn": round(gnn_weight, 4),
        },
        "inputs": {
            "pacx_malicious_score": round(pacx_score, 4),
            "gnn_malicious_score": round(gnn_score, 4),
            "gnn_completeness": round(completeness, 4),
            "gnn_model_certainty": round(model_certainty, 4),
        },
    }


def summarize_binary_confusion(predictions):
    binary_true = []
    binary_pred = []

    for item in predictions:
        actual_label = item.get("actual_label")
        predicted_label = item.get("predicted_label")
        if actual_label is None or predicted_label is None:
            continue
        binary_true.append(0 if actual_label == BENIGN_CLASS_INDEX else 1)
        binary_pred.append(0 if predicted_label == BENIGN_CLASS_INDEX else 1)

    if not binary_true:
        return None

    tp = sum(1 for t, p in zip(binary_true, binary_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(binary_true, binary_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(binary_true, binary_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(binary_true, binary_pred) if t == 1 and p == 0)
    total = len(binary_true)

    accuracy = ((tp + tn) / total) * 100 if total else 0.0
    precision = (tp / (tp + fp)) * 100 if (tp + fp) else 0.0
    recall = (tp / (tp + fn)) * 100 if (tp + fn) else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "dataset_size": total,
        "positive_samples": sum(binary_true),
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1_score, 2),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def estimate_pacx_baseline_metrics(dataset_size, positive_samples):
    """
    PAC-X's reference performance from prior literature, projected onto this
    dataset's class balance so the confusion-matrix panel has consistent
    totals. These numbers are NOT measured on this dataset: PACXHeuristicAnalyzer
    works on raw text/API strings, while the stored dataset is already hashed
    into numeric feature vectors, so the heuristic can't be re-run against the
    held-out split today. Flagged explicitly via `is_measured` below.
    """
    total = max(int(dataset_size), 1)
    positives = min(max(int(positive_samples), 0), total)
    negatives = total - positives

    reference_accuracy = 0.821
    reference_recall = 0.758

    tp = min(positives, int(round(positives * reference_recall)))
    fn = positives - tp

    reference_correct = int(round(total * reference_accuracy))
    tn = min(negatives, max(0, reference_correct - tp))
    fp = negatives - tn

    accuracy = ((tp + tn) / total) * 100 if total else 0.0
    precision = (tp / (tp + fp)) * 100 if (tp + fp) else 0.0
    recall = (tp / (tp + fn)) * 100 if (tp + fn) else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "name": "PAC-X Heuristic Baseline",
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1_score, 2),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "is_measured": False,
        "note": "Reference values from prior PAC-X literature, not measured via live inference on this dataset.",
    }


def summarize_multiclass_confusion(predictions):
    """Real per-family confusion matrix and metrics (11-class), not the
    binary benign/malicious collapse used by summarize_binary_confusion."""
    y_true = []
    y_pred = []
    for item in predictions:
        actual_label = item.get("actual_label")
        predicted_label = item.get("predicted_label")
        if actual_label is None or predicted_label is None:
            continue
        y_true.append(int(actual_label))
        y_pred.append(int(predicted_label))

    if not y_true:
        return None

    labels = list(range(len(CLASS_LABELS)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred))) * 100

    per_class = [
        {
            "label": label_name_for(idx),
            "precision": round(float(precision[i]) * 100, 2),
            "recall": round(float(recall[i]) * 100, 2),
            "f1_score": round(float(f1[i]) * 100, 2),
            "support": int(support[i]),
        }
        for i, idx in enumerate(labels)
    ]

    return {
        "dataset_size": len(y_true),
        "accuracy": round(accuracy, 2),
        "macro_precision": round(float(np.mean(precision)) * 100, 2),
        "macro_recall": round(float(np.mean(recall)) * 100, 2),
        "macro_f1": round(float(np.mean(f1)) * 100, 2),
        "class_labels": CLASS_LABELS,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def build_dataset_report():
    gnn_metrics = None
    raw_predictions = None

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        all_predictions = data.get("trusted_predictions", []) + data.get("uncertain_predictions_zero_day", [])
        gnn_metrics = summarize_binary_confusion(all_predictions)
        if gnn_metrics:
            raw_predictions = all_predictions

    if not gnn_metrics:
        dataset_path = 'data/pyg_dataset_norm.pt' if os.path.exists('data/pyg_dataset_norm.pt') else 'data/pyg_dataset.pt'
        if not os.path.exists(dataset_path):
            return None

        dataset = torch.load(dataset_path, weights_only=False)
        test_index_path = os.path.join("data", "test_indices.npy")
        if os.path.exists(test_index_path):
            test_indices = np.load(test_index_path)
            evaluation_dataset = [dataset[int(idx)] for idx in test_indices]
        else:
            evaluation_dataset = dataset

        predictions = []
        loader = DataLoader(evaluation_dataset, batch_size=64, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.batch)
                predicted = torch.argmax(logits, dim=1)
                for pred, actual in zip(predicted, batch.y.view(-1)):
                    predictions.append({
                        "predicted_label": int(pred.item()),
                        "actual_label": int(actual.item()),
                    })

        gnn_metrics = summarize_binary_confusion(predictions)
        if not gnn_metrics:
            return None
        raw_predictions = predictions

    multiclass_metrics = summarize_multiclass_confusion(raw_predictions)

    return {
        "success": True,
        "dataset_size": gnn_metrics["dataset_size"],
        "gnn_model_multiclass": multiclass_metrics,
        "baseline_model": estimate_pacx_baseline_metrics(
            gnn_metrics["dataset_size"],
            gnn_metrics["positive_samples"],
        ),
        "gnn_model": {
            "name": "GNN (Agentic-PACX)",
            "accuracy": gnn_metrics["accuracy"],
            "precision": gnn_metrics["precision"],
            "recall": gnn_metrics["recall"],
            "f1_score": gnn_metrics["f1_score"],
            "confusion_matrix": gnn_metrics["confusion_matrix"],
        }
    }

# Initialize FastAPI
def run_gnn_inference(graph, completeness, signal_summary):
    gnn_result = {}
    with torch.no_grad():
        graph_batch = get_graph_batch(graph)
        with model_lock:
            try:
                output = model(graph.x, graph.edge_index, graph_batch, return_attention=True)
                if isinstance(output, tuple):
                    logits, attention_data = output
                    if len(attention_data) == 3:
                        att1, att2, att3 = attention_data
                    elif len(attention_data) == 2:
                        att1, att2 = attention_data
                    else:
                        att1, att2 = None, None
                else:
                    logits = output
                    att1, att2 = None, None
            except TypeError:
                logits = model(graph.x, graph.edge_index, graph_batch)
                att1, att2 = None, None
        
        probs = torch.softmax(logits, dim=1)
        gnn_confidence, gnn_prediction = torch.max(probs, dim=1)
        predicted_class_idx = gnn_prediction.item()
        predicted_family = label_name_for(predicted_class_idx)
        benign_prob = probs[0][BENIGN_CLASS_INDEX].item() if probs.shape[1] > BENIGN_CLASS_INDEX else 0.0
        model_malicious_prob = max(0.0, 1.0 - benign_prob)

        centroid_probs = compute_centroid_probabilities(graph)
        centroid_benign_prob = centroid_probs.get(BENIGN_CLASS_INDEX, 0.0)
        centroid_malicious_prob = max(0.0, 1.0 - centroid_benign_prob)

        suspicious_count = float(signal_summary.get("suspicious_count", 0))
        api_count = float(signal_summary.get("api_count", 0))
        network_count = float(signal_summary.get("network_count", 0))
        evidence_score = min(
            1.0,
            0.2 * suspicious_count + 0.08 * min(api_count, 5) + 0.08 * min(network_count, 4)
        )

        model_certainty = compute_model_certainty(probs[0])
        model_weight = clamp(0.30 + (0.35 * completeness) + (0.30 * model_certainty), 0.25, 0.88)
        centroid_weight = clamp(0.18 + (0.22 * (1.0 - model_certainty)) + (0.10 * completeness), 0.08, 0.42)
        evidence_weight = max(0.0, 1.0 - model_weight - centroid_weight)
        
        total_weight = model_weight + centroid_weight + evidence_weight
        if total_weight <= 0:
            model_weight, centroid_weight, evidence_weight = 0.6, 0.25, 0.15
        else:
            model_weight /= total_weight
            centroid_weight /= total_weight
            evidence_weight /= total_weight
            
        malicious_prob = (
            model_malicious_prob * model_weight +
            centroid_malicious_prob * centroid_weight +
            evidence_score * evidence_weight
        )
        
        malicious_prob = clamp(malicious_prob, 0.0, 1.0)
        benign_prob_dynamic = max(0.0, 1.0 - malicious_prob)
        diagnosis_threshold = 0.5
        diagnosis = "Malicious" if malicious_prob >= diagnosis_threshold else "Benign"
        raw_binary_confidence = malicious_prob if diagnosis == "Malicious" else benign_prob_dynamic
        confidence_penalty = max(0.5, completeness)
        binary_confidence = raw_binary_confidence * confidence_penalty
        
        top_k = min(3, probs.shape[1])
        top_scores, top_indices = torch.topk(probs[0], k=top_k)
        top_classes = [
            {
                "class_idx": int(idx.item()),
                "label": label_name_for(int(idx.item())),
                "confidence": round(float(score.item()), 4),
            }
            for score, idx in zip(top_scores, top_indices)
        ]

        centroid_top_classes = [
            {
                "class_idx": int(label),
                "label": label_name_for(int(label)),
                "confidence": round(float(score), 4),
            }
            for label, score in sorted(centroid_probs.items(), key=lambda item: item[1], reverse=True)[:top_k]
        ]

        if diagnosis == "Malicious" and predicted_class_idx == BENIGN_CLASS_INDEX:
            non_benign_candidates = [
                item for item in centroid_top_classes
                if item["class_idx"] != BENIGN_CLASS_INDEX
            ]
            if non_benign_candidates:
                predicted_family = non_benign_candidates[0]["label"]
                gnn_confidence = torch.tensor(non_benign_candidates[0]["confidence"], device=graph.x.device)
        
        node_attention = [0.0, 0.0, 0.0, 0.0]
        if att1 is not None:
            try:
                edge_index, weights = att1
                num_edges = min(len(weights), edge_index.shape[1])
                for i in range(num_edges):
                    target_node = edge_index[1][i].item()
                    if target_node < 4:
                        node_attention[target_node] += weights[i].mean().item()
            except Exception as e:
                print(f"[!] Attention extraction issue: {e}")
        
        total = sum(node_attention) if sum(node_attention) > 0 else 1
        radar_data = [round(v/total, 2) for v in node_attention]
        
        return {
            "prediction": diagnosis,
            "confidence": round(binary_confidence, 4),
            "completeness": round(completeness, 4),
            "attention_weights": radar_data,
            "predicted_class_idx": predicted_class_idx,
            "predicted_family": predicted_family,
            "family_confidence": round(gnn_confidence.item(), 4),
            "raw_confidence": round(raw_binary_confidence, 4),
            "binary_scores": {
                "benign": round(benign_prob_dynamic, 4),
                "malicious": round(malicious_prob, 4),
            },
            "model_scores": {
                "benign": round(benign_prob, 4),
                "malicious": round(model_malicious_prob, 4),
            },
            "centroid_scores": {
                "benign": round(centroid_benign_prob, 4),
                "malicious": round(centroid_malicious_prob, 4),
            },
            "blend_weights": {
                "model": round(model_weight, 4),
                "centroid": round(centroid_weight, 4),
                "evidence": round(evidence_weight, 4),
                "model_certainty": round(model_certainty, 4),
            },
            "evidence_score": round(evidence_score, 4),
            "signal_summary": signal_summary,
            "top_classes": top_classes,
            "centroid_top_classes": centroid_top_classes,
            "raw_attention": (att1, att2) if att1 is not None else []
        }

app = FastAPI(
    title="Neural Model Evaluator API",
    description="Professional GNN vs PAC-X Malware Analysis Backend",
    version="2.0.0"
)

# Add CORS Middleware
ALLOWED_ORIGINS = os.getenv(
    "CORS_ALLOWED_ORIGINS",
    "http://127.0.0.1:8000,http://localhost:8000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # override via CORS_ALLOWED_ORIGINS env var for other hosts
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize GNN Model (Aligned with 11-class dataset)
# Guards concurrent access to `model` across inference reads and the
# Agent-2 reload-after-retrain write, so a reload can't interleave with
# an in-flight forward pass.
model_lock = threading.Lock()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Using device: {device}")
model = MalwareGAT(num_node_features=260, hidden_channels=64, num_classes=11, num_heads=4)
try:
    if os.path.exists("model.pt"):
        state_dict = torch.load("model.pt", map_location=device)
        try:
            model.load_state_dict(state_dict)
            print("[+] GNN Model weights loaded successfully.")
        except Exception:
            model.load_state_dict(state_dict, strict=False)
            print("[*] GNN model partially loaded (compatible layers only).")
    else:
        print("[!] Warning: model.pt not found. Using untrained model.")
    model.eval().to(device)
except Exception as e:
    print(f"[-] Warning: Could not load GNN model: {e}")
    model.eval().to(device)

# Configuration
RESULTS_FILE = "gnn_outputs.json"
MAX_POOL_SIZE = 500  # cap on data/retrain_pool.pt so it doesn't grow unbounded
try:
    # Try normalized dataset first, then fall back to standard
    dataset_path = 'data/pyg_dataset_norm.pt' if os.path.exists('data/pyg_dataset_norm.pt') else 'data/pyg_dataset.pt'
    adapter = UniversalInputAdapter(raw_dataset_path=dataset_path)
    pacx_engine = PACXHeuristicAnalyzer()
    print(f"[+] Input adapter initialized with {dataset_path}")
    print("[+] PAC-X Heuristic Engine ready.")
except Exception as e:
    print(f"[-] Warning: Could not initialize engines: {e}")
    adapter = None
    pacx_engine = None

# Static Files & Templates
app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates")

@app.get("/api/health")
async def health_check():
    """System health check endpoint."""
    return {
        "status": "online",
        "device": str(device),
        "model_loaded": os.path.exists("model.pt"),
        "adapter_ready": adapter is not None
    }

@app.get("/")
async def read_index(request: Request):
    """Serves the Premium Model Evaluator Dashboard."""
    return templates.TemplateResponse(request, "index.html")

@app.get("/live_analysis.html")
async def read_live(request: Request):
    """Serves the Live Analysis Upload Dashboard."""
    return templates.TemplateResponse(request, "live_analysis.html")

@app.get("/metrics_report.html")
async def get_metrics_report(request: Request):
    """Serves the detailed model metrics report page."""
    return templates.TemplateResponse(request, "metrics_report.html")

@app.get("/visual_graphs.html")
async def get_visual_graphs(request: Request):
    """Serves the visual behavior analysis graphs page."""
    return templates.TemplateResponse(request, "visual_graphs.html")

@app.get("/api/results")
def get_results():
    """Returns the head-to-head performance reports for GNN and PAC-X."""
    try:
        data = generate_ui_reports(RESULTS_FILE)
        if not data:
            raise HTTPException(status_code=404, detail="Results file not found. Please run GNN engine first.")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _run_agent2_retrain_background():
    """
    Runs Agent-2 retraining off the request thread and reloads the updated
    weights under model_lock when done. Fire-and-forget: the live-analysis
    response that triggered this has already been returned to the client
    with the pre-retrain result, since retraining takes minutes.
    """
    print("\n" + "!" * 60)
    print("[AGENT 2] Background retraining started...")
    print("!" * 60 + "\n")
    retrain_status = trigger_agent_2_retraining()
    if not retrain_status.get("success"):
        print(f"[-] Background Agent-2 retraining failed: {retrain_status.get('log')}")
        return

    print("[+] Background Agent-2 retraining complete. Reloading weights...")
    if os.path.exists("model.pt"):
        try:
            state_dict = torch.load("model.pt", map_location=device)
            with model_lock:
                model.load_state_dict(state_dict)
                model.eval().to(device)
            print("[+] Reloaded new weights into active memory. Next analysis will use them.")
        except Exception as reload_err:
            print(f"[-] Reload model weights failed: {reload_err}")


@app.post("/api/analyze/live")
async def analyze_live(
    background_tasks: BackgroundTasks,
    text: Optional[str] = Form(None), 
    file: Optional[UploadFile] = File(None),
    input_type: Optional[str] = Form("api")
):
    """
    Dual-Path Malware Detection:
    1. PAC-X: Feature-based analysis (Prospect, Aspect, Context)
    2. GNN: Graph-based analysis with attention weights
    3. Agent 1: Compares both and generates dynamic reasoning
    4. Agent 2: Auto-retrains if confidence < 60% (zero-day detected)
    """
    content = ""
    if file:
        content = (await file.read()).decode("utf-8")
    elif text:
        content = text
    else:
        raise HTTPException(status_code=400, detail="No input provided.")

    try:
        # Check if adapter is ready
        if adapter is None:
            raise HTTPException(status_code=503, detail="Input adapter not initialized. Check dataset files.")
        
        # STEP 1: Parse Input into Graph with type hint
        try:
            parse_result = adapter.parse(content, input_type=input_type)
            graph = parse_result["graph"].to(device)
            completeness = parse_result.get("completeness", 1.0)
            signal_summary = parse_result.get("signal_summary", {})
            nodes_filled_detail = parse_result.get("nodes_filled_detail", {})
        except Exception as e:
            print(f"[-] Adapter error: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to parse input: {str(e)}")
        
        # STEP 2: GNN PATH - Run GNN Inference
        try:
            gnn_result = run_gnn_inference(graph, completeness, signal_summary)
        except Exception as e:
            print(f"[-] GNN inference error: {e}")
            raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
        
        # STEP 3: PAC-X PATH - Feature-based analysis (Prospect, Aspect, Context)
        if pacx_engine is not None:
            raw_pacx = pacx_engine.analyze(content, input_type=input_type)
            pacx_result = {
                "prediction": raw_pacx["prediction"],
                "confidence": raw_pacx["confidence"],
                "findings": raw_pacx["findings"],
                "entropy": raw_pacx["entropy"],
                "breakdown": raw_pacx.get("breakdown", {}),
                "detected_apis": raw_pacx.get("detected_apis", []),
                "detected_patterns": raw_pacx.get("detected_patterns", []),
                "evidence_score": round(
                    min(
                        1.0,
                        len(raw_pacx.get("detected_apis", [])) * 0.2 +
                        len(raw_pacx.get("detected_patterns", [])) * 0.15 +
                        (0.15 if raw_pacx.get("entropy", 0) >= 6.5 else 0.0)
                    ),
                    4
                ),
            }
        else:
            pacx_result = {
                "prediction": "Unknown",
                "confidence": 0.0,
                "findings": ["PAC-X Engine not initialized"],
                "breakdown": {},
                "detected_apis": [],
                "detected_patterns": [],
                "evidence_score": 0.0,
            }
        
        # STEP 4: AGENT 1 - Compare paths first so fusion can arbitrate disagreements
        from agents.analyst_agent import ComparisonAgent
        agent1 = ComparisonAgent()
        agent1_analysis = agent1.compare_results(pacx_result, gnn_result)

        # STEP 5: Decision Fusion Layer (uses Agent 1's trust scores on disagreement)
        fusion_result = fuse_decisions(
            pacx_result, gnn_result,
            trust_scores=agent1_analysis["agent1_decision"]["trust_scores"],
        )

        # STEP 6: AGENT 2 - Retraining Trigger if confidence < 60%
        agent2_triggered = False
        retraining_executed = False
        fusion_confidence = float(fusion_result.get("confidence", 0.0))
        should_retrain = fusion_confidence < 0.6
        
        if should_retrain:
            agent2_triggered = True
            pool_path = "data/retrain_pool.pt"
            new_sample = graph.to('cpu')
            
            # Pseudo-label from current best class so Agent-2 retraining has supervised targets.
            pseudo_label = gnn_result["predicted_class_idx"]
            if fusion_result.get("final_label") == "Benign":
                pseudo_label = BENIGN_CLASS_INDEX
            new_sample.y = torch.tensor([int(pseudo_label)], dtype=torch.long)
            
            pool = []
            if os.path.exists(pool_path):
                try: 
                    pool = torch.load(pool_path, weights_only=False)
                except: 
                    pool = []
            pool.append(new_sample)
            pool = pool[-MAX_POOL_SIZE:]
            torch.save(pool, pool_path)

            print("\n" + "!"*60)
            print("[ALERT] AGENT 2 ACTIVATED: ZERO-DAY DETECTED (CONFIDENCE < 60%)")
            print(f"[*] Low confidence sample added to retrain pool")
            print(f"[*] Total samples in pool: {len(pool)}")
            print("[*] Queuing GNN retraining as a background task...")
            print("!"*60 + "\n")

            # Retraining takes minutes -- queue it instead of blocking this
            # response. This request's result reflects the current (pre-retrain)
            # weights; the next analysis after retraining completes will pick
            # up the updated model automatically via model_lock.
            background_tasks.add_task(_run_agent2_retrain_background)

        # STEP 7: GNN Analysis Report
        prediction_package = {
            "predicted_label": 1 if gnn_result["prediction"] == "Malicious" else 0,
            "confidence": gnn_result["confidence"],
            "actual_label": None,
            "attention_weights": gnn_result["raw_attention"],
        }
        from agents.analyst_agent import AnalystAgent
        gnn_analyst = AnalystAgent()
        try:
            gnn_report = gnn_analyst.analyze_prediction(prediction_package)
        except Exception as e:
            print(f"[!] GNN report generation error (non-fatal): {e}")
            gnn_report = {"rationale": "Analysis report unavailable.", "recommendation": ""}
        
        # STEP 8: Return comprehensive dual-path results
        return {
            "success": True,
            "completeness": completeness,
            "decision_fusion": fusion_result,
            "agent1_decision": agent1_analysis["agent1_decision"],
            "metrics_comparison": agent1_analysis["metrics_comparison"],
            "agent1_recommendation": agent1_analysis["recommendation"],
            "path_pacx": {
                "diagnosis": pacx_result["prediction"],
                "confidence": pacx_result["confidence"],
                "findings": pacx_result["findings"],
                "entropy": pacx_result["entropy"],
                "breakdown": pacx_result["breakdown"],
                "detected_apis": pacx_result["detected_apis"],
                "detected_patterns": pacx_result["detected_patterns"],
                "evidence_score": pacx_result["evidence_score"],
                "method": "Prospect-Aspect-Context Analysis"
            },
            "path_gnn": {
                "diagnosis": gnn_result["prediction"],
                "confidence": gnn_result["confidence"],
                "completeness": gnn_result["completeness"],
                "attention_weights": gnn_result["attention_weights"],
                "predicted_family": gnn_result["predicted_family"],
                "predicted_class_idx": gnn_result["predicted_class_idx"],
                "family_confidence": gnn_result["family_confidence"],
                "raw_confidence": gnn_result["raw_confidence"],
                "binary_scores": gnn_result["binary_scores"],
                "model_scores": gnn_result["model_scores"],
                "centroid_scores": gnn_result["centroid_scores"],
                "blend_weights": gnn_result["blend_weights"],
                "evidence_score": gnn_result["evidence_score"],
                "signal_summary": gnn_result["signal_summary"],
                "top_classes": gnn_result["top_classes"],
                "centroid_top_classes": gnn_result["centroid_top_classes"],
                "method": "Graph Neural Network with Attention"
            },
            "gnn_report": gnn_report,
            "agent2": {
                "triggered": agent2_triggered,
                "retraining_executed": retraining_executed,
                "retraining_queued": agent2_triggered,
                "basis": "decision_fusion.confidence",
                "trigger_threshold": 0.6,
                "trigger_value": round(fusion_confidence, 4),
                "reason": (
                    "Low fused confidence (< 60%) -- sample added to retrain pool, "
                    "Agent-2 retraining queued in the background (this result uses current weights)"
                ) if agent2_triggered else "Fused confidence is stable"
            },
            "radar_data": gnn_result["attention_weights"],
            "nodes_filled_detail": nodes_filled_detail
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[-] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/retrain")
async def retrain_model():
    """Triggers the Agent 2 Autonomous Retraining Protocol."""
    result = trigger_agent_2_retraining()
    return result

@app.get("/api/audit/env")
async def audit_env():
    """Validates the deep learning environment and hardware acceleration."""
    return {"report": verify_environment()}

@app.get("/api/audit/dataset")
async def audit_data():
    """Audits the structural alignment between Graph and Tabular databases."""
    return {"results": audit_dataset()}

@app.get("/api/audit/visualize")
async def audit_viz():
    """Generates a pictorial graph sample for forensic inspection."""
    path = generate_pictorial_graph()
    return {"img_path": path}

@app.get("/api/report")
async def get_detailed_report():
    """Returns dataset-backed metrics instead of confidence-derived placeholders."""
    report = build_dataset_report()
    if not report:
        raise HTTPException(status_code=404, detail="Report data not found. Please generate gnn_outputs.json first.")
    return report

@app.get("/api/ablation")
async def get_ablation_report():
    """Returns the GNN-only vs PAC-X-only vs Fused ablation study, if it's been generated."""
    ablation_path = "ablation_report.json"
    if not os.path.exists(ablation_path):
        raise HTTPException(status_code=404, detail="No ablation report found. Run scripts/build_ablation_report.py first.")
    with open(ablation_path, "r", encoding="utf-8") as handle:
        return json.load(handle)

@app.get("/api/training_history")
async def get_training_history():
    """Returns real per-epoch training curve data from the last training run."""
    history_path = "training_history.json"
    if not os.path.exists(history_path):
        raise HTTPException(status_code=404, detail="No training history found. Run agentic_pacx/gnn_classification.py first.")
    with open(history_path, "r", encoding="utf-8") as handle:
        return json.load(handle)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
