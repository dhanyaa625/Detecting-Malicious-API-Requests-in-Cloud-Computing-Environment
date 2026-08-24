import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def generate_unified_split(input_csv: str, cleaned_csv: str, train_out: str, test_out: str, seed: int = 42):
    print("="*60)
    print(" Generating Unified Train/Test Split ")
    print("="*60)
    
    print(f"[*] Loading raw dataset: {input_csv}")
    dataset = pd.read_csv(input_csv, low_memory=False)
    initial_shape = dataset.shape
    
    # 1. Clean dataset natively matching PAC-X logic
    dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
    dataset.drop_duplicates(inplace=True)
    
    # Label formatting
    dataset['label'] = dataset['label'].astype(str).str.replace(r'^benign_.+', 'benign', regex=True)
    
    # Filter classes with few samples
    value_counts = dataset['label'].value_counts()
    to_remove = value_counts[value_counts <= 50].index
    print(f"[*] Removing rare classes based on <= 50 counts (Found: {len(to_remove)})")
    
    dataset = dataset[~dataset['label'].isin(to_remove)]
    
    # Ensure index is reset so row indices map precisely 0 to N-1
    dataset.reset_index(drop=True, inplace=True)
    print(f"[*] Cleaned Dataset Shape: {dataset.shape} (Removed {initial_shape[0] - dataset.shape[0]} rows)")
    
    # 2. Save cleaned dataset
    dataset.to_csv(cleaned_csv, index=False)
    print(f"[✓] Saved cleaned dataset to: {cleaned_csv}")
    
    # 3. Create stratified split indices
    indices = np.arange(len(dataset))
    labels = dataset['label']
    
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=0.2, 
        random_state=seed, 
        stratify=labels
    )
    
    # 4. Save indices
    np.save(train_out, train_indices)
    np.save(test_out, test_indices)
    
    print(f"[✓] Saved Train Indices: {len(train_indices)} samples ({train_out})")
    print(f"[✓] Saved Test Indices : {len(test_indices)} samples ({test_out})")
    print("="*60)
    print(" Pipeline is officially aligned for PAC-X and GNN.")
    print("="*60)

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    generate_unified_split(
        input_csv="data/final_data.csv",
        cleaned_csv="data/cleaned_data.csv",
        train_out="data/train_indices.npy",
        test_out="data/test_indices.npy",
        seed=42
    )
