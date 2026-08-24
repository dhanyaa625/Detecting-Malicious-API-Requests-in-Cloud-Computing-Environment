# Detecting Malicious API Requests in Cloud Computing Environment

**Agentic-PACX** is a hybrid, next-generation threat intelligence and explainability ecosystem. It seamlessly fuses rule-based heuristics with deep structural Graph Attention Networks (GNN), dynamically orchestrated by cooperative cognitive LLM agents to deliver automated forensic reasoning and self-healing transfer-learning cycles.

---
Project Link: https://drive.google.com/file/d/1UvdVN8DfP-RtieoE0nyamg6S6Lf8Sj_q/view?usp=drive_link
## 🌟 System Architecture & Dynamic Flow

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
                                 /              \
                                v                v
                     +--------------------+    +--------------------+
                     |  PATH 1: PAC-X     |    |   PATH 2: GNN      |
                     |  Heuristic Engine  |    |  Graph Attention   |
                     +--------------------+    +--------------------+
                                 \              /
                                  \            /
                                   v          v
                        +-------------------------------+
                        |     Decision Fusion Layer     |
                        +-------------------------------+
                                        |
                                        v
                        +-------------------------------+
                        |    Agent 1: Forensic LLM      |
                        +-------------------------------+
                                   /         \
                                  /           \
                 (Confidence >= 60%)         (Confidence < 60%)
                                /               \
                               v                 v
                 +-------------------+     +-------------------------+
                 | System Safe /     |     |   AGENT 2 ACTIVATED     |
                 | Threat Isolated   |     |   Zero-Day Retrain Pool |
                 +-------------------+     +-------------------------+
                                                     |
                                                     v
                                           +-------------------------+
                                           | Transfer Learning Loop  |
                                           |  Updates model.pt weights|
                                           +-------------------------+
                                                     |
                                                     v
                                           +-------------------------+
                                           | Active Memory Reload    |
                                           +-------------------------+
```

---

## 📂 Project Folder Directory Structure

```
[Project Root]
├── agentic_pacx/              <-- Core GNN neural network implementation and training script.
├── agents/                    <-- System Orchestration Agents (Agent 1 & Agent 2 trigger).
├── core/                      <-- Core utility algorithms (Adapter, PAC-X, Graphs, Visualizer).
│   └── utils/                 <-- Environmental check, dataset audit, and forensic image generator.
├── data/                      <-- Local databases: pyg datasets, split indices, norm stats, retrain pools.
├── graph_output/              <-- Forensics image storage for dynamic behavior charts.
├── prospect/                  <-- Threat intelligence consistency research and MMD scores.
├── scripts/                   <-- Dataset build/ablation/demo-sample utilities (see below).
├── paper/                     <-- The IEEE paper source (agentic_pacx.tex).
├── web/                       <-- Complete Frontend code (templates and static stylesheets/scripts).
│   ├── templates/             <-- Server-served HTML dashboard layouts.
│   └── static/                <-- Stylesheets, icons, and dynamic Plotly JS UI logic.
├── model.pt                   <-- Active compiled PyTorch weights file for the GNN.
├── app.py                     <-- Main Unified FastAPI entry point (combines backend API & UI templates).
├── run_frontend.py            <-- Backup static simple HTTP server utility.

```

### Folder Roles & Responsibilities
* **`web/` (The Front-End Codebase):** Manages the entire visual dashboard layer. 
  - `web/templates/` holds the served layouts: `index.html` (main evaluation dashboard), `live_analysis.html` (real-time upload and fusion scan page), `metrics_report.html` (ROC/confusion matrix comparisons), and `visual_graphs.html` (interactive topological node graphs).
  - `web/static/` holds `css/style.css` (custom dark-mode design system) and `js/script.js` (UI logic, form submissions, and Plotly dynamic rendering).
* **`agentic_pacx/` (GNN Core Algorithms):** Houses the GNN mathematical model definition (`MalwareGAT` - a deeper 3-layer GAT architecture with batch normalization and dropout) and `gnn_classification.py` which executes GNN training and Phase 3 transfer-learning retraining loops.
* **`agents/` (The Orchestration Agents):** Contains `analyst_agent.py`. It holds the logic for **Agent 1** (forensic LLM threat reasoning) and the starting trigger for **Agent 2** (`trigger_agent_2_retraining()`).
* **`core/` (Core Logic Layers):**
  - `core/input_adapter.py`: Standardizes live raw-text/JSON/CSV input, normalizes vector features against `data/norm_stats.pt` (train-split statistics), and handles format conversions. This is an approximate bridge from unstructured logs into the model's structured feature space, not a byte-for-byte reconstruction of the training data's PE-analysis features.
  - `core/pacx_analyzer.py`: Performs Prospect-Aspect-Context heuristic evaluations.
  - `core/graph_constructor.py`: Shared node/edge constants and the `hash_feature_vector`/`pad_numeric_features` helpers.
  - `core/dataset_manager.py`: Stratified train/val/test split (70/10/20) and train-split-only feature normalization.
  - `core/utils/visualizer.py`: Generates the forensic behavior graphs saved to disk.
* **`scripts/` (Dataset & Reporting Utilities):**
  - `build_graph_dataset.py`: Builds the 4-node graph dataset from `data/cleaned_data.csv` -- every node (Header, Entropy, API, Network) is a real numeric feature vector pulled directly from that node's CSV columns, not a hash of stringified values.
  - `build_ablation_report.py`: Runs the GNN-only / PAC-X-only / Fused ablation on the real held-out test split, producing `ablation_report.json`.
  - `build_demo_samples.py`: Pulls one real, held-out CSV row per class into `demo_samples/` for live-demo testing, with a `manifest.csv` of true labels.
  - `build_pool_from_uncertain.py`: Rebuilds `data/retrain_pool.pt` from the genuinely low-confidence predictions in `gnn_outputs.json`, with real ground-truth labels.
* **`data/` (Local Datasets):** Contains GNN datasets (`pyg_dataset.pt`, `pyg_dataset_norm.pt`), NumPy evaluation splits (`train_indices.npy` / `val_indices.npy` / `test_indices.npy`), normalization stats (`norm_stats.pt`), clean labels (`cleaned_data.csv`), and active retraining pool files (`retrain_pool.pt`). Regenerate the first four with `python scripts/build_graph_dataset.py && python -m core.dataset_manager`.
* **`prospect/` (MMD Robustness & Consistency):** Validates feature robustness and structural consistency across diverse malware variants using Maximum Mean Discrepancy (MMD) scores (`consistency.py`, `robustness.py`, and score sheets).

---

## 🛠️ Technology Stack Breakdown

### Frontend:
* **HTML5 & CSS3:** Responsive structural grid layout built on a Catppuccin-inspired dark-mode slate theme, styled with premium glassmorphism tokens, hover micro-animations, and status cards.
* **Vanilla JavaScript:** Event-driven async architecture managing dynamic API requests, parsing responses, and refreshing panels without page reloads.
* **Plotly.js:** Powers dynamic forensic visuals in the dashboard:
  - **Topological Node Graphs:** Interactively displays node-connection strengths between the 4 core nodes (Header, Entropy, API, Network).
  - **Behavioral Sunburst Charts:** Hierarchically maps complex dynamic indicators in the "Deep Dive" panel for easy forensics.

### Backend:
* **FastAPI (ASGI Core):** Asynchronous Python web framework serving endpoints, health checks, environment diagnostics, and Jinja2-rendered templates.
* **PyTorch & PyTorch Geometric (PyG):** Defines and runs the neural engine:
  - **MalwareGAT:** A 3-layer **Graph Attention Network (GAT)** stack (`gat1`, `gat2`, `gat3`) with hidden dimension 64 throughout. Attention heads step down per layer -- 4 heads (concat, → 256-dim) in `gat1`, 2 heads (concat, → 128-dim) in `gat2`, 1 head (no concat, → 64-dim) in `gat3` -- with Batch Normalization and Dropout (25%) after each. It convolutes structural semantic associations and runs global mean pooling to output predictions across 11 classes.

---

## 🎯 Universal Input Adapter & Drop-down Selector

The frontend features a drop-down to specify format structures, but the backend is designed to be **extremely robust against user misconfigurations**:

### Supported Dropdown Formats
1. **Simple Text check (API dropdown):**
   * **Accepts:** Raw comma-separated commands, API traces, or command logs.
   * **Example File:** [demo_samples/api_call_log_malicious.txt](demo_samples/api_call_log_malicious.txt) (also see `api_call_log_benign.txt`).
2. **Structured JSON (JSON dropdown):**
   * **Accepts:** JSON telemetry records.
   * **Parser Logic:** Maps `"api"`, `"api_calls"`, `"network"`, and `"ip"` keys directly to corresponding API and Network nodes.
   * No bundled example file ships yet -- see `core/tests/demo_adapter_inputs.py` for a JSON payload you can paste in directly.
3. **CSV Log File (CSV dropdown):**
   * **Accepts:** Tabular log tables.
   * **Parser Logic:** Searches for column headings containing `"api"`, `"network"`, or `"ip"` to extract values.
   * **Example Files:** [demo_samples/](demo_samples/) -- one held-out CSV row per malware family (e.g. `gandcrab_3814.csv`, `benign_1.csv`), with true labels listed in `demo_samples/manifest.csv`. Regenerate with `python scripts/build_demo_samples.py`.

### 🛡️ Corner-Point & Fail-Safe Mechanisms
* **Drop-Down Mismatch Recovery:** If you upload a `.csv` log but keep the drop-down selected as `"JSON"`, the adapter catches the parsing error, degrades to global regex extraction, and successfully builds a valid 4-node GNN graph anyway.
* **Format Corruption Fallback:** If a JSON or CSV file is malformed, the adapter intercepts the exception and seamlessly processes the payload as standard raw text.
* **Obfuscation Stripping:** Runs `canonicalize_api()` to remove common API wrappers (e.g. stripping kernel indicators like `Nt`, `Zw` and Windows ABI flags like `Ex`, `A`, `W`) to map zero-days directly back to the GNN's known behavioral mappings.

---

## 🧠 Model Fusion Math & Orchestration Agents

### 1. **The Decision Fusion Layer**
Blends heuristic rule outputs with structural GNN predictions:
$$\text{Final Score} = (\text{PAC-X Weight} \times \text{PAC-X Score}) + (\text{GNN Weight} \times \text{GNN Score})$$
* **Base Settings:** 0.60 GNN Weight + 0.40 PAC-X Weight.
* **Dynamic Calibration:** Weights adjust dynamically using GNN structural completeness and prediction certainty:
  $$\text{Reliability Bonus} = 0.20 \times \left( (0.50 \times \text{Completeness}) + (0.50 \times \text{Certainty}) \right)$$
* GNN Weight is clamped between `[0.50, 0.80]`, ensuring balanced consensus at all times.

### 2. **Agent 1 (Comparative Showdowns)**
Compares both pathways and generates natural, expert-level forensic reports explaining the diagnosis (using Gemini 2.0 Flash / Flash-Lite; falls back to static templated reasoning if no `GEMINI_API_KEY` is set or the API call fails). It evaluates Model Trust Scores based on completeness, evidence, and model confidence:
* **PAC-X Trust:** $(0.7 \times \text{PAC-X Confidence}) + (0.3 \times \text{PAC-X Evidence})$
* **GNN Trust:** $(0.35 \times \text{GNN Raw Confidence}) + (0.10 \times \text{Acc}) + (0.15 \times \text{Completeness}) + (0.25 \times \text{Evidence}) + (0.15 \times \text{Family Confidence})$
* **Acc** is the model's own measured overall test accuracy (loaded from `training_history.json`'s `final_test_metrics`), a fixed quality prior for the currently-loaded model version -- not this sample's own confidence, which already contributes via the 0.35 term above.

### 3. **Agent 2 (Autonomous Retraining & Self-Healing)**
To maintain high speed and prevent redundant code blocks, **Agent 2 has no standalone file**. It is divided between:
* **Trigger Orchestrator:** Located inside `agents/analyst_agent.py` (`trigger_agent_2_retraining()`).
* **Learning Brain:** Located inside `agentic_pacx/gnn_classification.py` (`--retrain`).

* **Trigger Condition:** Agent 2 triggers **ONLY** when the final fused confidence falls below **60% (< 0.60)**.
* **Retrain Flow:** Appends the zero-day sample with a pseudo-label to `data/retrain_pool.pt`, runs a 10-epoch transfer learning loop to fine-tune `model.pt` (via a `BackgroundTasks` job, so it doesn't block the server), reloads the weights in active memory under a lock, and corrects future diagnoses automatically.
* **Guardrails:** Refuses to fine-tune on a pool smaller than 20 samples or spanning fewer than 2 classes, and discards the update (keeping the prior `model.pt`) if held-out test accuracy regresses by more than 3 points after fine-tuning. These exist because an unguarded retrain on a tiny, single-class pool measurably regresses accuracy -- reproduced directly: a deliberately unguarded 2-sample, single-class retrain took held-out accuracy from 99.61% to 98.64% even with best-epoch selection, and as low as 80.08% mid-run.

---

## 📈 System Performance & Health Checks

Measured on the 1,029-sample held-out test split (never used for model selection -- see `core/dataset_manager.py`), after fixing the API/Network node feature construction (see Stage 1 of the audit below):

* **GNN Multiclass Accuracy (11 classes):** **99.61%** (Macro F1: **99.17%**, Macro Precision: **98.49%**, Macro Recall: **99.89%**)
* **GNN Binary Accuracy (benign/malicious):** **99.71%** (Precision: **99.56%**, Recall: **99.78%**)
* **PAC-X Heuristic Accuracy, measured on this dataset's CSV modality:** **56.27%** (0% precision/recall on the malicious class -- PAC-X's keyword/entropy heuristics have no signal on a numeric PE-header row; see `ablation_report.json`). PAC-X's designed strength is raw text/API-log input, which this dataset doesn't natively provide -- **82.1%** is a literature reference figure for that modality, not a result measured here.

Regenerate all of the above with `python scripts/build_graph_dataset.py && python -m core.dataset_manager && python agentic_pacx/gnn_classification.py && python scripts/build_ablation_report.py`.

### Professional Health Checks
* `/api/health` - Live FastAPI process status.
* `/api/audit/env` - Evaluates CUDA/CPU deep learning hardware acceleration.
* `/api/audit/dataset` - Verifies tabular and graph database structural alignment.

---

## 🚀 Quickstart Guide

### 1. Setup & Installation
```bash
# Clone the repository and install requirements
pip install -r requirements.txt
```

### 2. (Optional) Enable real LLM reasoning
Agent 1 works out of the box with static templated reasoning if no key is configured. To get real Gemini-generated forensic reports:
```bash
cp .env.example .env
# then edit .env and paste in a real key from https://aistudio.google.com/apikey
```
`agents/analyst_agent.py` loads `.env` automatically (via `python-dotenv`) on every run -- no need to export an environment variable by hand. `.env` is gitignored, so the key never gets committed.

### 3. Launch the Application
Run the FastAPI backend server (from the root folder):
```bash
python app.py
```
Open your browser at **[http://localhost:8000/live_analysis.html](http://localhost:8000/live_analysis.html)**.

### 4. Regenerate the dataset/model (optional -- already committed as trained artifacts)
```bash
python scripts/build_graph_dataset.py       # data/pyg_dataset.pt
python -m core.dataset_manager              # split + normalize -> pyg_dataset_norm.pt, norm_stats.pt
python agentic_pacx/gnn_classification.py   # trains model.pt, writes training_history.json + gnn_outputs.json
python scripts/build_ablation_report.py     # ablation_report.json
python scripts/build_demo_samples.py        # refreshes demo_samples/ against the current split
```

### 5. Test every feature end-to-end
* **Live analysis, text mode:** on `/live_analysis.html`, leave the dropdown on "API Call Log (Text)", paste the contents of `demo_samples/api_call_log_malicious.txt` (or `_benign.txt`), click **Analyze Request**. Check: PAC-X panel, GNN panel, the Agent 1 reasoning box, the new **GNN Structural Forensic Report** card, and the Decision Fusion panel all populate.
* **Live analysis, CSV mode:** switch the dropdown to "Full Feature Record (CSV)" and upload any file from `demo_samples/` (e.g. `gandcrab_3814.csv`) -- these are genuine held-out rows; `demo_samples/manifest.csv` lists the true label for each, so you can confirm the prediction matches.
* **Live analysis, JSON mode:** switch the dropdown to "Partial Request Data (JSON)" and paste a payload like `{"api_calls": ["ResumeThread", "kernel32!LoadLibraryA"], "ip": "192.168.1.1"}` (see `core/tests/demo_adapter_inputs.py` for more).
* **Agent 2 (self-healing):** click **"Trigger Agent 2 Retraining"**. It queues in the background and confirms immediately; refresh after ~30-60s to pick up new weights if the pool passed its guardrails (it needs >=20 samples across >=2 classes -- check the server console log for whether it ran or was skipped).
* **Metrics dashboard:** open `/metrics_report.html` for the full confusion matrix, per-class precision/recall table, and training curve.
* **Graph visualizations:** open `/visual_graphs.html` for the topological node-graph view.
* **Health/audit endpoints:** `GET /api/health`, `/api/audit/env`, `/api/audit/dataset`, `/api/report`, `/api/ablation`, `/api/training_history`.
* **Automated smoke test:** `python core/tests/test_adapter.py` exercises the adapter + model end-to-end without the UI.
