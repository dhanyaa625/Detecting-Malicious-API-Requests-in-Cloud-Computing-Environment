"""
==============================================================================
  Environment Verification Script for Member 2 (GNN Architect)
==============================================================================

Purpose: Verify that all required dependencies are installed and working
         before running the full GNN training pipeline.

Usage: python verify_gnn_environment.py

"""

import sys
import warnings
warnings.filterwarnings("ignore")

def verify_environment():
    """Verify that all required dependencies are installed and working. Returns report list."""
    report = []
    
    report.append("[1/4] Checking Python version...")
    version = sys.version_info
    report.append(f"Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        report.append("❌ ERROR: Python 3.8+ required")
        return report
    
    report.append("[2/4] Checking required packages...")
    required = [
        ("torch", "PyTorch"),
        ("torch_geometric", "PyTorch Geometric"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-Learn"),
        ("requests", "LLM Provider HTTP client (Groq/Ollama)"),
        ("fastapi", "FastAPI Framework")
    ]
    
    for module, display_name in required:
        try:
            __import__(module)
            report.append(f"✅ {display_name} OK")
        except ImportError:
            report.append(f"❌ {display_name} FAILED")
    
    report.append("[3/4] Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            report.append(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            report.append("⚠️ No GPU detected - using CPU")
    except Exception as e:
        report.append(f"⚠️ GPU check error: {str(e)}")
    
    report.append("[4/4] Testing PyTorch Geometric...")
    try:
        from torch_geometric.data import Data
        x = torch.randn(5, 3) 
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        report.append(f"✅ PyG Test Passed. Nodes: {data.num_nodes}")
    except Exception as e:
        report.append(f"❌ PyG test failed: {str(e)}")
        
    return report

if __name__ == "__main__":
    for line in verify_environment():
        print(line)
