import torch
import pandas as pd
from typing import List
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

def inject_encoded_labels():
    print("="*60)
    print(" Injecting Synchronized Labels to PyG Dataset ")
    print("="*60)
    
    csv_path = "data/cleaned_data.csv"
    graph_path = "data/pyg_dataset_norm.pt"
    
    print("[*] Loading CSV to map LabelEncoder...")
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Mirror PAC-X training.py exactly
    le = LabelEncoder()
    df['encoded_label'] = le.fit_transform(df['label'])
    
    encoded_classes = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"[*] Encoded Classes Mapping: {encoded_classes}")
    
    print("[*] Loading Graph Dataset...")
    graphs: List[Data] = torch.load(graph_path, weights_only=False)
    
    if len(graphs) != len(df):
        print("[!] FATAL ERROR: Row count mismatch.")
        return
        
    print(f"[*] Injecting labels into {len(graphs)} graphs...")
    for i in range(len(graphs)):
        lbl = df.iloc[i]['encoded_label']
        # PAC-X uses one-hot encoding internally in NN, but PyG typically just uses class indices for CrossEntropy
        graphs[i].y = torch.tensor([lbl], dtype=torch.long)
        
    print("[*] Resaving synchronized graph dataset...")
    torch.save(graphs, graph_path)
    print("[✓] Injection Complete. GNN will learn mathematically aligned labels.")
    
if __name__ == "__main__":
    inject_encoded_labels()
