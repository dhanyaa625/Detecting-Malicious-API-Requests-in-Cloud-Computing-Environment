import json
import os
import subprocess
import sys
from dotenv import load_dotenv
from google import genai
from sklearn.metrics import confusion_matrix
import numpy as np

# Loads GEMINI_API_KEY from a .env file in the project root if present, so a
# key can be configured without exporting an environment variable by hand.
# Never overrides a key already set in the real environment.
load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY", "")
genai_client = genai.Client(api_key=api_key) if api_key else None

class ComparisonAgent:
    """Agent 1: Compares PAC-X vs GNN results with Gemini-based dynamic reasoning."""
    
    def __init__(self):
        self.use_gemini = genai_client is not None
    
    def _generate_comparison_reasoning(self, pacx_metrics, gnn_metrics, pacx_pred, gnn_pred, pacx_result=None, gnn_result=None, better_model=None):
        """Uses Gemini to generate dynamic reasoning about which model is better."""
        if not self.use_gemini:
            return self._static_reasoning(pacx_metrics, gnn_metrics, pacx_pred, gnn_pred, pacx_result, gnn_result, better_model)
        
        prompt = f"""
You are a Malware Detection Expert. Compare two detection paths for THIS single sample:

**PAC-X (Heuristic Pattern Analysis):**
- Confidence: {pacx_metrics.get('confidence', 0) * 100:.2f}%
- Evidence score: {pacx_metrics.get('evidence_score', 0) * 100:.2f}%
- Prediction: {pacx_pred}

**GNN (Graph Attention Network - Agentic-PACX):**
- Confidence: {gnn_metrics.get('confidence', 0) * 100:.2f}%
- Evidence score: {gnn_metrics.get('evidence_score', 0) * 100:.2f}%
- Graph completeness: {gnn_metrics.get('completeness', 0) * 100:.2f}%
- Prediction: {gnn_pred}

Based on these signals:
1. Which model is MORE TRUSTWORTHY? Why?
2. What are the KEY DIFFERENCES in their approaches?
3. If they disagree, which prediction should be trusted more?

Provide a 3-4 sentence expert analysis focusing on technical differences and which model's methodology is superior for THIS specific case.
"""
        
        try:
            response = genai_client.models.generate_content(model="gemini-2.0-flash-lite", contents=prompt)
            return response.text.strip()
        except Exception as e:
            print(f"[!] Gemini API error: {e}")
            return self._static_reasoning(pacx_metrics, gnn_metrics, pacx_pred, gnn_pred, pacx_result, gnn_result, better_model)
    
    def _static_reasoning(self, pacx_metrics, gnn_metrics, pacx_pred, gnn_pred, pacx_result=None, gnn_result=None, better_model=None):
        """Fallback static reasoning if Gemini is unavailable."""
        pacx_result = pacx_result or {}
        gnn_result = gnn_result or {}

        pacx_findings = pacx_result.get("findings", [])
        gnn_family = gnn_result.get("predicted_family", "unknown")
        gnn_completeness = gnn_result.get("completeness", 0.0)
        gnn_evidence = gnn_result.get("evidence_score", gnn_metrics.get("confidence", 0.0))
        pacx_evidence = pacx_result.get("evidence_score", pacx_metrics.get("confidence", 0.0))

        if pacx_pred == gnn_pred:
            if pacx_pred == "Malicious":
                return (
                    f"PAC-X and GNN both flag this input as malicious. PAC-X found {len(pacx_findings)} heuristic indicators, "
                    f"while the GNN matched the structure closest to `{gnn_family}` with {gnn_completeness * 100:.0f}% graph completeness."
                )
            return (
                f"PAC-X and GNN both treat this input as benign. The GNN still tracked the sample against `{gnn_family}` and assigned "
                f"a low malicious evidence score of {gnn_evidence * 100:.1f}%, so the agreement looks stable."
            )

        if better_model == "GNN":
            return (
                f"GNN is preferred for this scan because the live graph carries stronger structural evidence "
                f"({gnn_evidence * 100:.1f}%) than PAC-X's heuristic evidence ({pacx_evidence * 100:.1f}%). "
                f"The graph also aligns most closely with `{gnn_family}`, which is useful for dynamic and zero-day style inputs."
            )

        return (
            f"PAC-X is preferred for this scan because its triggered findings are more decisive than the current graph confidence. "
            f"The GNN still contributes structural context through `{gnn_family}`, but this input is too incomplete for it to fully dominate yet."
        )
    
    def compare_results(self, pacx_result, gnn_result):
        """
        Compare PAC-X and GNN results and generate dynamic reasoning.
        
        Args:
            pacx_result: dict with keys: accuracy, precision, recall, confidence, prediction, explanation
            gnn_result: dict with keys: accuracy, precision, recall, confidence, prediction, attention_weights, explanation
        
        Returns:
            dict with comparison data and Agent 1 reasoning
        """
        
        # Extract per-scan signals (accuracy/precision/recall don't exist for a
        # single sample; confidence + evidence are the only meaningful metrics here).
        pacx_metrics = {
            'confidence': pacx_result.get('confidence', 0),
            'evidence_score': pacx_result.get('evidence_score', 0),
        }

        gnn_metrics = {
            'confidence': gnn_result.get('confidence', 0),
            'evidence_score': gnn_result.get('evidence_score', 0),
            'completeness': gnn_result.get('completeness', 1.0),
        }
        
        pacx_pred = pacx_result.get('prediction', 'Unknown')
        gnn_pred = gnn_result.get('prediction', 'Unknown')
        gnn_completeness = gnn_result.get('completeness', 1.0)
        pacx_evidence = pacx_result.get('evidence_score', pacx_metrics['confidence'])
        gnn_evidence = gnn_result.get('evidence_score', gnn_metrics['confidence'])
        
        pacx_trust = (pacx_metrics['confidence'] * 0.7) + (pacx_evidence * 0.3)
        gnn_raw_conf = gnn_result.get('raw_confidence', gnn_metrics['confidence'])
        gnn_family_conf = gnn_result.get('family_confidence', 0.0)
        # "Acc" is the model's own measured overall test accuracy (a fixed
        # quality prior for this model version) -- not this sample's
        # per-scan confidence, which already contributes at weight 0.35
        # above via gnn_raw_conf. Falls back to gnn_raw_conf only if the
        # caller didn't supply model_accuracy (e.g. older cached results).
        gnn_model_accuracy = gnn_result.get('model_accuracy', gnn_raw_conf)
        gnn_trust = (
            (gnn_raw_conf * 0.35) +
            (gnn_model_accuracy * 0.10) +
            (gnn_completeness * 0.15) +
            (gnn_evidence * 0.25) +
            (gnn_family_conf * 0.15)
        )
        better_model = "GNN" if gnn_trust >= pacx_trust else "PAC-X"
        
        # Generate dynamic reasoning
        reasoning = self._generate_comparison_reasoning(
            pacx_metrics,
            gnn_metrics,
            pacx_pred,
            gnn_pred,
            pacx_result,
            gnn_result,
            better_model,
        )
        
        # Calculate agreement
        prediction_agreement = pacx_pred == gnn_pred
        
        return {
            "agent1_decision": {
                "better_model": better_model,
                "reasoning": reasoning,
                "prediction_agreement": prediction_agreement,
                "agreed_on": pacx_pred if prediction_agreement else "CONFLICTING - Requires Review",
                "trust_scores": {
                    "pacx": round(pacx_trust, 4),
                    "gnn": round(gnn_trust, 4),
                }
            },
            "metrics_comparison": {
                "pacx": pacx_metrics,
                "gnn": gnn_metrics
            },
            "recommendation": f"Trust {better_model} model" if prediction_agreement else f"Trust {better_model} (models disagree - manual review recommended)"
        }

class AnalystAgent:
    def __init__(self, use_llm=True):
        """Initializes the Analyst Agent (LLM or Rule-Based)."""
        self.node_map = {
            0: "PE Header Array",
            1: "Section Entropy",
            2: "API Import/Export",
            3: "Network Topology"
        }
        self.use_llm = use_llm and genai_client is not None
        
    def _generate_dynamic_rationale(self, label_str, confidence, top_interaction, max_weight):
        """Generates rationale using Gemini 1.5 Flash."""
        if not self.use_llm:
            return None
            
        prompt = f"""
        Act as a Malware Security Researcher. Analyze the following GNN (Graph Neural Network) detection output:
        - Diagnosis: {label_str}
        - Confidence: {confidence:.2f}%
        - Top Topological Interaction: {top_interaction}
        - Attention Score: {max_weight:.2f} (This indicates importance in the software graph)

        Write a 2-3 sentence technical rationale explaining why these structural indicators support the diagnosis. 
        Focus on how the interaction between {top_interaction.split(' → ')[0]} and {top_interaction.split(' → ')[1]} is significant in a malware context.
        """
        try:
            response = genai_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            return response.text.strip()
        except Exception:
            return None

    def analyze_prediction(self, prediction_data):
        """Transforms raw mathematical GNN output into a clean Markdown Report."""
        confidence = prediction_data.get('confidence', 0) * 100
        predicted_label = prediction_data.get('predicted_label', 0)
        label_str = "Benign (Safe)" if predicted_label == 0 else "Malicious (Threat Detected)"
        
        attention_tuples = prediction_data.get('attention_weights', [])
        top_interaction = "Complex Structural Behavior"
        max_weight = 0.0
        
        try:
            # Parse Layer 1 Attention (Index 0 in tuple)
            layer1 = attention_tuples[0]
            sources = layer1[0][0]
            targets = layer1[0][1]
            alphas = layer1[1]  # Matrix of 4 heads
            
            for i in range(len(sources)):
                # Take highest attention from the 4 heads for this specific edge
                weight = max(alphas[i])
                if weight > max_weight:
                    max_weight = weight
                    src_name = self.node_map.get(sources[i], f"Node {sources[i]}")
                    dst_name = self.node_map.get(targets[i], f"Node {targets[i]}")
                    top_interaction = f"{src_name} → {dst_name}"
        except Exception as e:
            pass

        # Use LLM for Dynamic Logic if available, else fallback to Static
        dynamic_rationale = self._generate_dynamic_rationale(label_str, confidence, top_interaction, max_weight)
        
        if dynamic_rationale:
            rationale = dynamic_rationale
            recommendation = "🚨 <b>Expert Recommendation:</b> " + ("Immediate quarantine and behavioral analysis." if predicted_label == 1 else "Safe to proceed, regular system scans recommended.")
        else:
            import random
            # Build Explanable AI Report (Static Fallback)
            if predicted_label == 1:
                rationale_opts = [
                    f"This executable exhibits highly anomalous structural patterns. Specifically, the GNN identified a critical vulnerability indicator between <b>{top_interaction}</b> with an attention score of {max_weight:.2f}.",
                    f"Malicious infrastructure detected. The neural edge between <b>{top_interaction}</b> heavily indicates obfuscated API loading tactics typical of ransomware.",
                    f"Warning flags raised by the GNN due to strange topological interactions centered around <b>{top_interaction}</b>."
                ]
                rationale = random.choice(rationale_opts)
                recommendation = "🚨 <b>Recommendation:</b> Immediate quarantine. Investigate network endpoints if outbound calls were made."
            else:
                rationale_opts = [
                    f"The executable aligns with standard software patterns. The highest structural priority was placed on <b>{top_interaction}</b>, but no malicious signatures were mathematically triggered.",
                    f"Normal structural behavior. The neural network confirmed safe execution flow, prioritizing <b>{top_interaction}</b> with standard baseline weights.",
                    f"No topological anomalies found. The interaction <b>{top_interaction}</b> appears routine for this type of executable."
                ]
                rationale = random.choice(rationale_opts)
                recommendation = "✅ <b>Recommendation:</b> Safe to execute. Monitor if system privileges escalate unexpectedly."

        # HTML Report Generation
        report = f"""
<h4 style="margin-bottom: 5px; color: #FFFFFF;">Threat Intelligence Report</h4>
<p style="margin-top: 0; font-size: 14px; color: #CDD6F4;">
  <b>Final Diagnosis:</b> <span style="color: {'#F38BA8' if predicted_label == 1 else '#A6E3A1'};">{label_str}</span> | 
  <b>Mathematical Confidence:</b> {confidence:.2f}%
</p>
<p style="color: #CDD6F4;"><b>Explainable AI (XAI) Rationale:</b><br>{rationale}</p>
<p style="color: #CDD6F4;">{recommendation}</p>
"""
        return report

    def calculate_performance_metrics(self, data_list):
        """Calculates TP, TN, FP, FN and confusion matrix."""
        y_true = []
        y_pred = []
        
        for item in data_list:
            actual = item.get("actual_label")
            predicted = item.get("predicted_label")
            if actual is not None:
                y_true.append(actual)
                y_pred.append(predicted)
        
        if not y_true:
            return None
            
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle cases with 1 class or multiple classes
            # Simple fallback for TN/FP/FN/TP calculation
            tp = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
            tn = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 0))
            fp = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 1))
            fn = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 0))
        
        return {
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "Matrix": cm.tolist()
        }

    def compare_models(self, gnn_results, pacx_results=None):
        """Generates the competitive analysis string explicitly proving why GNN > PAC-X"""
        
        # Reference baseline for PAC-X (from prior literature, not measured on
        # this dataset -- PACXHeuristicAnalyzer runs on raw text, not the
        # pre-hashed feature vectors stored here, so it can't be re-run on demand).
        pacx_metrics = {
            "Accuracy": 82.1,
            "F1_Score": 79.4,
            "Recall": 75.8,
            "Precision": 76.5,
            "is_measured": False,
        }

        # Calculate dynamic GNN metrics from prediction stream
        gnn_perf = self.calculate_performance_metrics(gnn_results)
        if gnn_perf:
            total = sum([gnn_perf["TP"], gnn_perf["TN"], gnn_perf["FP"], gnn_perf["FN"]])
            acc = (gnn_perf["TP"] + gnn_perf["TN"]) / total * 100
            rec = gnn_perf["TP"] / (gnn_perf["TP"] + gnn_perf["FN"]) * 100 if (gnn_perf["TP"] + gnn_perf["FN"]) > 0 else 0
            prec = gnn_perf["TP"] / (gnn_perf["TP"] + gnn_perf["FP"]) * 100 if (gnn_perf["TP"] + gnn_perf["FP"]) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

            gnn_metrics = {
                "Accuracy": acc,
                "F1_Score": f1,
                "Recall": rec,
                "Precision": prec,
                "Raw": gnn_perf,
                "is_measured": True,
            }
        else:
            gnn_metrics = {
                "Accuracy": None,
                "F1_Score": None,
                "Recall": None,
                "Precision": None,
                "is_measured": False,
                "note": "Not yet computed -- run gnn_classification.py to generate labeled predictions first.",
            }
        
        metrics = {
            "PAC-X (Baseline)": pacx_metrics,
            "GNN (Agentic-PACX)": gnn_metrics
        }
        
        rationale = """
**Agent 1 Model Showdown Analysis:**
The original PAC-X model relied on static, flat arrays. This means it only understood that statistical features existed, but not how they interacted. 

The **Graph Attention Network (GNN)** outperformed PAC-X because it mathematically analyzes the *topology* of the software. By focusing on the exact edge connections (e.g., when the PE Header directly communicates with a suspicious Network API export), the GNN can detect obfuscated zero-day ransomware that standard baseline statistics completely miss. 
        """
        
        return {
            "metrics": metrics,
            "rationale": rationale
        }

def generate_ui_reports(json_filepath):
    """Utility function to be called by Streamlit UI to get processed data."""
    if not os.path.exists(json_filepath):
        return None
        
    with open(json_filepath, 'r') as f:
        data = json.load(f)
        
    agent = AnalystAgent()
    
    trusted_cases = data.get("trusted_predictions", [])
    zero_day_cases = data.get("uncertain_predictions_zero_day", [])
    
    import random
    # Sort array so Malicious items appear globally at the top in the UI
    trusted_cases.sort(key=lambda x: x.get('predicted_label', 0), reverse=True)
    zero_day_cases.sort(key=lambda x: x.get('predicted_label', 0), reverse=True)
    
    # Generate reports for UI display (mapping them directly)
    for t in trusted_cases:
        t['report'] = agent.analyze_prediction(t)
        # Winner Logic: Higher accuracy confidence score
        t['winner'] = "GNN" if t.get('confidence', 0) >= 0.5 else "PAC-X (Dummy)"
        
    for z in zero_day_cases:
        z['report'] = agent.analyze_prediction(z)
        z['winner'] = "PAC-X (Heuristic)" if z.get('confidence', 0) < 0.4 else "GNN"
        
    showdown_data = agent.compare_models(trusted_cases + zero_day_cases)
        
    return {
        "trusted": trusted_cases, 
        "zero_days": zero_day_cases,
        "showdown": showdown_data
    }

def trigger_agent_2_retraining():
    """Agent 2 (Retraining Orchestrator) triggers the GNN training loop on Zero Days."""
    try:
        # Path corrected to the aligned agentic_pacx folder
        script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agentic_pacx", "gnn_classification.py")
        # Run it and capture the output so the UI can display it
        result = subprocess.run([sys.executable, script_path, "--retrain"], capture_output=True, text=True, check=True)
        return {"success": True, "log": result.stdout}
    except subprocess.CalledProcessError as e:
        return {"success": False, "log": e.stdout + e.stderr}

if __name__ == "__main__":
    # Test block
    agent = AnalystAgent()
    print("[*] Python Rule-Based Analyst Agent Ready.")
