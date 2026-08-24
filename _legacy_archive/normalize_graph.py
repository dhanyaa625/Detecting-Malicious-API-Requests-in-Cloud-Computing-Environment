import torch
import numpy as np
from typing import List
from torch_geometric.data import Data

def normalize_dataset(dataset_path: str, output_path: str):
    print("="*60)
    print(" Feature Scaling Verification & Normalization ")
    print("="*60)
    
    # Load the graph
    dataset: List[Data] = torch.load(dataset_path, weights_only=False)
    if len(dataset) == 0:
        print("[!] Dataset is empty.")
        return
        
    num_graphs = len(dataset)
    dim_features = dataset[0].x.shape[1] - 4 # 256
    
    # Extract raw features per node type (ignoring one-hot)
    # Shape: [num_graphs, dim_features]
    header_feats = torch.stack([g.x[0, :dim_features] for g in dataset])
    entropy_feats = torch.stack([g.x[1, :dim_features] for g in dataset])
    api_feats = torch.stack([g.x[2, :dim_features] for g in dataset])
    network_feats = torch.stack([g.x[3, :dim_features] for g in dataset])
    
    def print_stats(name, tensor):
        # Flatten to compute global mean/std/min/max
        flat = tensor.flatten()
        mean = flat.mean().item()
        std = flat.std().item()
        c_min = flat.min().item()
        c_max = flat.max().item()
        
        # Check if values > 5
        large_vals = (flat.abs() > 5.0).sum().item()
        
        print(f"--- {name} ---")
        print(f"Mean: {mean:7.4f} | Std: {std:7.4f} | Min: {c_min:7.4f} | Max: {c_max:7.4f}")
        print(f"Values > |5| : {large_vals}")
        return mean, std
        
    print("\n[ BEFORE NORMALIZATION ]")
    print_stats("Header Node", header_feats)
    print_stats("Entropy Node", entropy_feats)
    print_stats("API Node", api_feats)
    print_stats("Network Node", network_feats)
    
    # ── Normalize ──
    print("\n[*] Normalizing features to (mean=0, std=1)...")
    
    # Applying Standardization per node type, per feature
    def standardize(tensor):
        mean = tensor.mean(dim=0, keepdim=True)
        std = tensor.std(dim=0, keepdim=True)
        # Avoid division by zero for constant features
        std[std == 0] = 1.0
        return (tensor - mean) / std

    header_norm = standardize(header_feats)
    entropy_norm = standardize(entropy_feats)
    api_norm = standardize(api_feats)
    network_norm = standardize(network_feats)
    
    print("\n[ AFTER NORMALIZATION ]")
    print_stats("Header Node", header_norm)
    print_stats("Entropy Node", entropy_norm)
    print_stats("API Node", api_norm)
    print_stats("Network Node", network_norm)
    
    # ── Reconstruct Graphs ──
    print("\n[*] Reconstructing dataset...")
    for i, g in enumerate(dataset):
        # Build new X
        new_x = torch.zeros_like(g.x)
        # Copy normalized features
        new_x[0, :dim_features] = header_norm[i]
        new_x[1, :dim_features] = entropy_norm[i]
        new_x[2, :dim_features] = api_norm[i]
        new_x[3, :dim_features] = network_norm[i]
        
        # Copy one-hots
        new_x[:, -4:] = g.x[:, -4:]
        
        g.x = new_x
        
    # Validation checks
    nan_inf = any((torch.isnan(g.x).any() or torch.isinf(g.x).any()) for g in dataset)
    structs = all((g.num_nodes == 4 and g.num_edges == 8) for g in dataset)
    dims = all((g.x.shape[1] == dim_features + 4) for g in dataset)
    
    print("\n[ VALIDATION ]")
    print(f"No NaN or Inf values      : {not nan_inf}")
    print(f"Consistent Structure (4,8): {structs}")
    print(f"Consistent Dims (260)     : {dims}")
    
    # Save
    torch.save(dataset, output_path)
    print(f"\n[✓] Normalized dataset saved to: {output_path}")

if __name__ == "__main__":
    normalize_dataset("data/pyg_dataset.pt", "data/pyg_dataset_norm.pt")
