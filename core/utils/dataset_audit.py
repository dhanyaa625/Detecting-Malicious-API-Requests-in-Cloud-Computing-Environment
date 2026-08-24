import torch
import numpy as np
import pandas as pd
from typing import List
from torch_geometric.data import Data

def audit_dataset():
    """Audit the structural integrity and label alignment between CSV and Graphs."""
    report = []
    
    # Paths
    graph_path = "data/pyg_dataset_norm.pt"
    csv_path = "data/cleaned_data.csv"
    train_path = "data/train_indices.npy"
    test_path = "data/test_indices.npy"
    
    try:
        report.append("[1/3] Loading Dataset structures...")
        graphs = torch.load(graph_path, weights_only=False)
        df = pd.read_csv(csv_path, low_memory=False)
        train_indices = np.load(train_path)
        test_indices = np.load(test_path)
        
        report.append(f"✅ Loaded {len(graphs)} graphs and {len(df)} tabular rows.")
        
        report.append("[2/3] Verifying Index Splits...")
        report.append(f"Training Samples: {len(train_indices)}")
        report.append(f"Testing Samples: {len(test_indices)}")
        
        if len(graphs) == len(df):
            report.append("✅ 1:1 Database Alignment Verified.")
        else:
            report.append("❌ Database Mismatch Detected!")
            
        report.append("[3/3] Cross-Checking Label Synchronization...")
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        encoded = le.fit_transform(df['label'])
        
        mismatches = 0
        for i in range(min(len(graphs), len(df))):
            tabular_enc = encoded[i]
            gnn_label = graphs[i].y.item() if graphs[i].y is not None else -1
            if tabular_enc != gnn_label:
                mismatches += 1
                
        if mismatches == 0:
            report.append("✅ Perfect Label Sync! GNN and PAC-X are perfectly aligned.")
        else:
            report.append(f"⚠️ Warning: {mismatches} labels misaligned between models.")
            
    except Exception as e:
        report.append(f"❌ Audit Failed: {str(e)}")
        
    return report

if __name__ == "__main__":
    for line in audit_dataset():
        print(line)
