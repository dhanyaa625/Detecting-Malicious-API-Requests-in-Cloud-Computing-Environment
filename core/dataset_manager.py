import torch
import pandas as pd
import numpy as np
from typing import List
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import LabelEncoder

class DatasetManager:
    """Professional manager for Graph Dataset preparation and auditing."""
    
    def __init__(self, raw_graph_path="data/pyg_dataset.pt", 
                 output_path="data/pyg_dataset_norm.pt", 
                 csv_path="data/cleaned_data.csv"):
        self.raw_graph_path = raw_graph_path
        self.output_path = output_path
        self.csv_path = csv_path

    def process_complete_pipeline(self):
        """Runs injection, normalization, and split in one professional flow."""
        print("--- Initiating Professional Data Pipeline ---")
        try:
            graphs = self.inject_labels()
            normalized_graphs = self.normalize(graphs)
            self.save(normalized_graphs)
            return {"success": True, "message": "Pipeline completed successfully."}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def inject_labels(self) -> List[Data]:
        """Synchronizes CSV labels with the Graph structures."""
        print("[*] Loading CSV and Graph data for synchronization...")
        df = pd.read_csv(self.csv_path, low_memory=False)
        graphs = torch.load(self.raw_graph_path, weights_only=False)
        
        if len(graphs) != len(df):
            raise ValueError(f"Mismatch: {len(graphs)} graphs vs {len(df)} CSV rows.")
            
        le = LabelEncoder()
        df['encoded_label'] = le.fit_transform(df['label'])
        
        for i in range(len(graphs)):
            lbl = df.iloc[i]['encoded_label']
            graphs[i].y = torch.tensor([lbl], dtype=torch.long)
            
        print(f"[✓] Injected {len(graphs)} labels.")
        return graphs

    def normalize(self, dataset: List[Data]) -> List[Data]:
        """Standardizes features per node type to Mean=0, Std=1."""
        print("[*] Normalizing topological features...")
        dim_features = dataset[0].x.shape[1] - 4 # Assuming last 4 are node-type one-hots
        
        # Standardize per node type
        for node_idx in range(4):
            node_feats = torch.stack([g.x[node_idx, :dim_features] for g in dataset])
            mean = node_feats.mean(dim=0, keepdim=True)
            std = node_feats.std(dim=0, keepdim=True)
            std[std == 0] = 1.0 # Avoid div-by-zero
            
            norm_feats = (node_feats - mean) / std
            
            for i in range(len(dataset)):
                dataset[i].x[node_idx, :dim_features] = norm_feats[i]
                
        print("[✓] Normalization complete.")
        return dataset

    def save(self, dataset: List[Data]):
        torch.save(dataset, self.output_path)
        print(f"[✓] Professional dataset saved to {self.output_path}")

if __name__ == "__main__":
    manager = DatasetManager()
    manager.process_complete_pipeline()
