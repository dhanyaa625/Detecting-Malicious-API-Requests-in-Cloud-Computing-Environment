# Detecting Malicious API Requests in Cloud Computing Environment

**Agentic-PACX** is a hybrid threat-detection framework that fuses a rule-based heuristic engine (PAC-X) with a structural Graph Attention Network (GNN), arbitrated by a trust-based Decision Fusion Layer and governed by two cognitive LLM agents — one for forensic explanation, one for gated self-healing retraining.

🔗 **Live demo:** [detecting-malicious-api-requests-in.onrender.com](https://detecting-malicious-api-requests-in.onrender.com/live_analysis.html) (free-tier hosting — sleeps after ~15 min idle, first request after that takes ~30-60s to wake up)

---

## 🌟 System Architecture

```
                     +---------------------------------------+
                     |          Uploaded Payload             |
                     +---------------------------------------+
                                         |
                                         v
                         +-------------------------------+
                         |    Universal Input Adapter    |
                         +-------------------------------+
                                  /            \
                                 v              v
                     +--------------------+    +-------------------------------+
                     |  PATH 1: PAC-X     |    |         PATH 2: S_GNN         |
                     |  Heuristic Engine  |    | GAT + Centroid Classifier +   |
                     |                    |    | Heuristic Evidence Score,     |
                     |                    |    | blended by graph completeness |
                     +--------------------+    +-------------------------------+
                                 \              /
                                  v            v
                        +-------------------------------+
                        |   Agent 1: Trust Scoring       |
                        +-------------------------------+
                                        |
                                        v
                        +-------------------------------+
                        |    Decision Fusion Layer       |
                        +-------------------------------+
                                   /         \
                    (Confidence >= 68%)   (Confidence < 68%)
                                /               \
                               v                 v
                 +-------------------+     +-----------------------------+
                 | Output + Forensic |     | Relabeled "Suspicious" +    |
                 | XAI Report        |     | pooled for AGENT 2 retrain  |
                 +-------------------+     +-----------------------------+
```

---

## 📂 Project Structure

```
[Project Root]
├── agentic_pacx/   <-- MalwareGAT (GNN) model + training script
├── agents/         <-- Agent 1 (forensic reasoning) & Agent 2 (retrain trigger)
├── core/           <-- Input adapter, PAC-X engine, graph construction, dataset splits
├── data/           <-- PyG datasets, split indices, norm stats, retrain pool
├── scripts/        <-- Dataset build, ablation, zero-day, and demo-sample utilities
├── paper/          <-- IEEE paper source (agentic_pacx.tex)
├── web/            <-- FastAPI-served frontend (templates + static assets)
├── model.pt        <-- Trained GNN weights
└── app.py          <-- FastAPI entry point (API + UI)
```

* **`core/input_adapter.py`** — normalizes raw text, JSON, CSV, or a PE binary into the model's 4-node graph format.
* **`core/pacx_analyzer.py`** — Prospect-Aspect-Context heuristic scoring.
* **`agentic_pacx/gnn_classification.py`** — MalwareGAT architecture, training, and `--retrain` (Agent 2's fine-tuning path).
* **`agents/analyst_agent.py`** — Agent 1 (trust comparison + forensic XAI reports) and Agent 2's trigger orchestration.
* **Dataset provenance:** `data/cleaned_data.csv` (5,138 rows, 11 classes: Benign + GandCrab, Emotet, Gamarue, Hotbar, DarkKomet, Delf, Domaiq, DLHelper, DriverPack, GameHack) matches `final_data.csv` from [McGill-DMaS/PACX](https://github.com/McGill-DMaS/PACX) (Saqib, Fung & Charland, *PAC-X: Fuzzy Explainable AI for Multiclass Malware Detection*, IEEE Trans. Fuzzy Syst., 2026).

---

## 🛠️ Technology Stack

* **Frontend:** HTML5/CSS3, vanilla JS, Plotly.js (topological node graphs, sunburst forensic charts)
* **Backend:** FastAPI (ASGI), Jinja2 templates
* **ML:** PyTorch + PyTorch Geometric — MalwareGAT is a 3-layer GAT stack (4 → 2 → 1 attention heads, 256 → 128 → 64-dim, BatchNorm + 25% dropout per layer), global mean pooling, 11-class output

---

## 🧠 How Detection Works

**Universal Input Adapter** accepts four input shapes and normalizes each into the same 4-node graph (PE Header, Section Entropy, API Import/Export, Network/String artifacts): raw API-call text/logs, JSON telemetry, CSV records (native schema or a public Kaggle PE-header schema), or a raw PE binary. Malformed input degrades gracefully to a lower-completeness graph rather than failing outright.

**Decision Fusion Layer** blends the two pathways' scores when they agree (dynamically weighted by graph completeness and model certainty, GNN weight clamped to `[0.50, 0.80]`), or arbitrates by trust score when they disagree — Agent 1 computes those trust scores first, since fusion's disagreement branch depends on them.

**S_GNN is not the bare GAT output.** Internally, the GNN pathway score blends three signals: the raw MalwareGAT softmax, a nearest-centroid classifier over the same graph embedding, and a content-based heuristic evidence score (attack-API/behavior keyword hits, not raw call volume) — weighted by graph completeness and model certainty. This matters most for the text/JSON input paths, which only ever populate 2 of the 4 node categories (API + network, never header/entropy): a confidently-wrong GAT/centroid prediction on that kind of partial graph can no longer dominate the fused verdict outright. Full weight formulas in `app.py`'s `run_gnn_inference`.

**Fail-closed zero-day escalation:** a fused verdict of "Benign" with confidence below the gate (`AGENT2_THRESHOLD = 0.68`) is relabeled "Suspicious" instead of cleared — a labelling rule only, no weights change. Measured over 12,081 pooled leave-one-family-out predictions: zero-day recall rises 78.36% → 85.45%, known-malware detection 99.98% → 100.00%, at a benign false-positive cost of 0.57% → 3.52%. Full derivation and the negative-result novelty-detection experiment are in `paper/agentic_pacx.tex`.

**Agent 2 (self-healing):** pooling a low-confidence sample into `data/retrain_pool.pt` is automatic; retraining is not — it requires an explicit `POST /api/retrain` trigger, and is refused if the pool has collapsed onto one pseudo-label or fewer than 2 classes, or discarded post-hoc if held-out accuracy regresses more than 3 points.

---

## 📈 Measured Performance

On the 1,029-sample held-out test split (never used for model selection):

| Metric | Score |
|---|---|
| Multiclass accuracy (11 classes) | **99.61%** |
| Multiclass macro F1 / precision / recall | **99.17%** / **98.49%** / **99.89%** |
| Binary accuracy | **99.71%** |
| Binary precision / recall | **99.56%** / **99.78%** |

Regenerate with:
```bash
python scripts/build_graph_dataset.py && python -m core.dataset_manager && python agentic_pacx/gnn_classification.py && python scripts/build_ablation_report.py
```

---

## 🚀 Quickstart

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. (Optional) Enable real LLM reasoning
Agent 1 works out of the box with static templated reasoning if no key is configured.
```bash
cp .env.example .env
# edit .env with a Groq API key
```

### 3. Run
```bash
python app.py
```
Open **http://localhost:8000/live_analysis.html**.

### 4. Try it
Upload any file from `demo_samples/` (true labels in `demo_samples/manifest.csv`), or paste `demo_samples/api_call_log_malicious.txt` into the text mode. See `/metrics_report.html` for the full confusion matrix and training curve, `/visual_graphs.html` for the topological graph view.

### 5. Health checks
`GET /api/health`, `/api/audit/env`, `/api/audit/dataset`, `/api/report`, `/api/ablation`, `/api/training_history`.
