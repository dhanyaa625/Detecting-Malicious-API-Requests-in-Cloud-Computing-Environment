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
├── data/                      <-- Local databases: pyg datasets, training indices, retrain pools.
├── graph_output/              <-- Forensics image storage for dynamic behavior charts.
├── prospect/                  <-- Threat intelligence consistency research and MMD scores.
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
  - `core/input_adapter.py`: Standardizes input logs, normalizes vector features, and handles format conversions.
  - `core/pacx_analyzer.py`: Performs Prospect-Aspect-Context heuristic evaluations.
  - `core/graph_constructor.py`: Standardizes structural PE graph nodes and hashes feature lists.
  - `core/utils/visualizer.py`: Generates the forensic behavior graphs saved to disk.
* **`data/` (Local Datasets):** Contains GNN datasets (`pyg_dataset.pt`, `pyg_dataset_norm.pt`), NumPy evaluation splits (`test_indices.npy`), clean labels (`cleaned_data.csv`), and active retraining pool files (`retrain_pool.pt`).
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
  - **MalwareGAT:** A 3-layer **Graph Attention Network (GAT)** stack (`gat1`, `gat2`, `gat3`) with 4 attention heads, hidden dimensions of 64, Batch Normalization, and Dropout (25%). It convolutes structural semantic associations and runs global pooling to output predictions across 11 classes.

---

## 🎯 Universal Input Adapter & Drop-down Selector

The frontend features a drop-down to specify format structures, but the backend is designed to be **extremely robust against user misconfigurations**:

### Supported Dropdown Formats
1. **Simple Text check (API dropdown):**
   * **Accepts:** Raw comma-separated commands, API traces, or command logs.
   * **Example File:** [zero_day_sample.txt](zero_day_sample.txt) (contains PE signs and APIs).
2. **Structured JSON (JSON dropdown):**
   * **Accepts:** JSON telemetry records.
   * **Parser Logic:** Maps `"api"`, `"api_calls"`, `"network"`, and `"ip"` keys directly to corresponding API and Network nodes.
   * **Example File:** [sample_json.json](sample_json.json)
3. **CSV Log File (CSV dropdown):**
   * **Accepts:** Tabular log tables.
   * **Parser Logic:** Searches for column headings containing `"api"`, `"network"`, or `"ip"` to extract values.
   * **Example File:** [sample_csv.csv](sample_csv.csv)

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
Compares both pathways and generates natural, expert-level forensic reports explaining the diagnosis (using Gemini 2.0 / 1.5 Flash). It evaluates Model Trust Scores based on completeness, evidence, and model confidence:
* **PAC-X Trust:** $(0.7 \times \text{PAC-X Confidence}) + (0.3 \times \text{PAC-X Evidence})$
* **GNN Trust:** $(0.35 \times \text{GNN Raw Confidence}) + (0.10 \times \text{Acc}) + (0.15 \times \text{Completeness}) + (0.25 \times \text{Evidence}) + (0.15 \times \text{Family Confidence})$

### 3. **Agent 2 (Autonomous Retraining & Self-Healing)**
To maintain high speed and prevent redundant code blocks, **Agent 2 has no standalone file**. It is divided between:
* **Trigger Orchestrator:** Located inside `agents/analyst_agent.py` (`trigger_agent_2_retraining()`).
* **Learning Brain:** Located inside `agentic_pacx/gnn_classification.py` (`--retrain`).

* **Trigger Condition:** Agent 2 triggers **ONLY** when the final fused confidence falls below **60% (< 0.60)**.
* **Retrain Flow:** Appends the zero-day sample with a pseudo-label to `data/retrain_pool.pt`, runs a 10-epoch transfer learning loop to fine-tune `model.pt`, reloads the weights in active memory, and corrects the live diagnosis automatically.

---

## 📈 System Performance & Health Checks

* **GNN Classification Accuracy:** **95.55%** (Macro F1-Score: **0.9184**)
* **PAC-X Heuristic Accuracy:** **82.10%**

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

### 2. Launch the Application
Run the FastAPI backend server (from the root folder):
```bash
python app.py
```
Open your browser at **[http://localhost:8000/live_analysis.html](http://localhost:8000/live_analysis.html)**.

### 3. Simulate Zero-Day Attacks
* **Test Text Log:** Upload `zero_day_sample.txt` using the **Simple Text check** dropdown.
* **Test JSON Log:** Upload `sample_json.json` using the **Structured JSON** dropdown.
* **Test CSV Log:** Upload `sample_csv.csv` using the **CSV Log File** dropdown.

*To test Agent 2 manually, click **"FORCE MODEL EVOLUTION"** in the bottom left panel of the live upload page. The backend will instantly run fine-tuning and reload the optimized model weights within seconds!*
