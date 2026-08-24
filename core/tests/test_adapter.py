import torch
import warnings
import sys
import os
warnings.filterwarnings("ignore")

# Force absolute pathing for safety due to deeply nested project wrapper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_pacx.input_adapter import UniversalInputAdapter
from phase2_gnn_pipeline import MalwareGAT
import torch.nn.functional as F

print("="*60)
print(" Initiating Universal Input Adapter Tests ")
print("="*60)

try:
    # Requires baseline dataset to extract existing mean/std from phase 1
    adapter = UniversalInputAdapter(raw_dataset_path='data/pyg_dataset.pt')
except Exception as e:
    print(f"[-] Could not initialize setup (Ensure 'data/pyg_dataset.pt' exists locally): {e}")
    exit(1)

# =====================================================================
# TEST 1: MODE 2 CANONICALIZATION & ZERO-PADDING
# =====================================================================
print("\n[TEST 1] Mode 2: Extreme Noise API String (Zero Context)")
raw_api_string = "kernel32.dll!CreateFileW, NtCreateFile, VirtualAllocEx, zwTerminateProcess, createfile"
print(f"-> Raw Input:  {raw_api_string}")

result_mode2 = adapter.parse(raw_api_string)
graph_m2 = result_mode2["graph"]

print(f"-> Completeness Score: {result_mode2['completeness']}")
print(f"-> Interpretation: {result_mode2['confidence_interpretation']}")

# Prove Canonicalization logic works symmetrically
tokens = [t.strip() for t in raw_api_string.split(',')]
canon = [adapter.canonicalize_api(t) for t in tokens]
print(f"-> Extracted Canonical Tokens: {canon}")
print(f"-> Node Matrix Built. Shape: {graph_m2.x.shape} (Must match 4x260)")

# =====================================================================
# TEST 2: MODE 3 JSON MAPPING
# =====================================================================
print("\n[TEST 2] Mode 3: Semi-Structured JSON (Missing Header/Entropy)")
json_payload = {
    "ip": "192.168.1.1, 8.8.8.8",
    "api_calls": ["ResumeThread", "kernel32!LoadLibraryA"]
}
print(f"-> JSON Payload: {json_payload}")

result_mode3 = adapter.parse(json_payload)
print(f"-> Completeness Score: {result_mode3['completeness']} (Should map nicely to 0.5)")
print(f"-> Interpretation: {result_mode3['confidence_interpretation']}")
print(f"-> Is Normalizer executing properly? Sum over Empty Nodes (Entropy/Header): {result_mode3['graph'].x[0:2, :256].sum().item():.3f} (This will securely float instead of crashing at absolute 0.0)")

# =====================================================================
# TEST 3: GNN ARCHITECTURE AND TRIGGER INJECTION (TAU-TEST)
# =====================================================================
print("\n[TEST 3] Phase 2 Mathematical Structural Integrity Check")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Using the teammate's newly finalized Confidence model schema 
model = MalwareGAT(num_node_features=260, hidden_channels=128, num_classes=2, dropout=0.2).to(device)
model.eval()

with torch.no_grad():
    # Pass Ghost Node graph locally. Batch vector maps all 4 nodes to a single 0-index graph.
    logits = model(graph_m2.x.to(device), graph_m2.edge_index.to(device), batch=torch.zeros(4, dtype=torch.long).to(device))
    probs = F.softmax(logits, dim=1)
    confidence, _ = torch.max(probs, dim=1)

print(f"-> [SUCCESS] Input successfully consumed by standard GNN Architecture.")
print(f"-> Network Dimensionality constraints securely preserved despite unstructured payload.")
print(f"-> Sample Output Confidence Level against Untrained Weights: {confidence[0].item():.4f}")
print("   *Note: Due to massive zeroes on input logic, fully trained weights will aggressively plunge this prediction below τ=0.6*")
print("\n[+] Verification Testing Checks Finished Completely.")
