"""
Smoke test for the Universal Input Adapter + live GNN inference path.
Run from the project root: python core/tests/test_adapter.py
(needs data/norm_stats.pt and data/pyg_dataset_norm.pt -- run
`python -m core.dataset_manager` first if they're missing.)
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# core/tests/test_adapter.py -> core/tests -> core -> project root (3 levels up).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import torch
import torch.nn.functional as F

from core.input_adapter import UniversalInputAdapter
from agentic_pacx.gnn_classification import MalwareGAT

print("=" * 60)
print(" Initiating Universal Input Adapter Tests ")
print("=" * 60)

try:
    adapter = UniversalInputAdapter()
except Exception as e:
    print(f"[-] Could not initialize adapter (run `python -m core.dataset_manager` first): {e}")
    sys.exit(1)

# =====================================================================
# TEST 1: MODE 2 CANONICALIZATION & ZERO-PADDING
# =====================================================================
print("\n[TEST 1] Mode 2: Extreme Noise API String (Zero Context)")
raw_api_string = "kernel32.dll!CreateFileW, NtCreateFile, VirtualAllocEx, ZwTerminateProcess, createfile"
print(f"-> Raw Input:  {raw_api_string}")

result_mode2 = adapter.parse(raw_api_string)
graph_m2 = result_mode2["graph"]

print(f"-> Completeness Score: {result_mode2['completeness']}")
print(f"-> Interpretation: {result_mode2['confidence_interpretation']}")

tokens = [t.strip() for t in raw_api_string.split(',')]
canon = [adapter.canonicalize_api(t) for t in tokens]
print(f"-> Extracted Canonical Tokens: {canon}")
print(f"-> Node Matrix Built. Shape: {tuple(graph_m2.x.shape)} (Must match 4x260)")
assert tuple(graph_m2.x.shape) == (4, 260), "Node matrix shape mismatch!"

# =====================================================================
# TEST 2: MODE 3 JSON MAPPING
# =====================================================================
print("\n[TEST 2] Mode 3: Semi-Structured JSON (Missing Header/Entropy)")
json_payload = {
    "ip": "192.168.1.1, 8.8.8.8",
    "api_calls": ["ResumeThread", "kernel32!LoadLibraryA"]
}
print(f"-> JSON Payload: {json_payload}")

result_mode3 = adapter.parse(json_payload, input_type="json")
print(f"-> Completeness Score: {result_mode3['completeness']} (api + network filled -> 0.5)")
print(f"-> Interpretation: {result_mode3['confidence_interpretation']}")
print(
    f"-> Normalizer ran on the empty Header/Entropy nodes without crashing: "
    f"sum={result_mode3['graph'].x[0:2, :256].sum().item():.3f}"
)

# =====================================================================
# TEST 3: GNN ARCHITECTURE AND TRIGGER INJECTION (TAU-TEST)
# =====================================================================
print("\n[TEST 3] Structural Integrity Check Against the Live Model Architecture")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MalwareGAT(num_node_features=260, num_classes=11).to(device)
model.eval()

with torch.no_grad():
    logits = model(
        graph_m2.x.to(device), graph_m2.edge_index.to(device),
        batch=torch.zeros(4, dtype=torch.long).to(device),
    )
    probs = F.softmax(logits, dim=1)
    confidence, _ = torch.max(probs, dim=1)

print("-> [SUCCESS] Input successfully consumed by the live MalwareGAT architecture.")
print(f"-> Sample output confidence against untrained weights: {confidence[0].item():.4f}")
print("\n[+] Verification Testing Checks Finished Completely.")
