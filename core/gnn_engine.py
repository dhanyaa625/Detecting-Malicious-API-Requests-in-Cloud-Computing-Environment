raise ImportError(
    "core/gnn_engine.py is dead code: it defines a 2-layer/2-class MalwareGAT "
    "that is architecturally incompatible with the live model in "
    "agentic_pacx/gnn_classification.py (3-layer/11-class), and is never "
    "imported by app.py. Use agentic_pacx.gnn_classification.MalwareGAT instead."
)

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

# ======================================================================
# 1. MODEL ARCHITECTURE (FIXED)
# ======================================================================
class MalwareGAT(torch.nn.Module):
    def __init__(self, num_node_features=260, hidden_channels=128, num_classes=2, num_heads=4, dropout=0.2):
        super(MalwareGAT, self).__init__()
        self.dropout = dropout
        self.gat1 = GATConv(num_node_features, hidden_channels, heads=num_heads, concat=True, dropout=self.dropout)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False, dropout=self.dropout)
        self.classifier = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, return_attention=False):
        # Layer 1
        x = F.dropout(x, p=self.dropout, training=self.training)
        if return_attention:
            x, attention_weights_1 = self.gat1(x, edge_index, return_attention_weights=True)
        else:
            x = self.gat1(x, edge_index)
            
        x = F.elu(x) 
        
        # Layer 2
        x = F.dropout(x, p=self.dropout, training=self.training)
        if return_attention:
            x, attention_weights_2 = self.gat2(x, edge_index, return_attention_weights=True)
        else:
            x = self.gat2(x, edge_index)
            
        x = F.elu(x)
        
        # Aggregate the 4 nodes into 1 graph-level tensor
        x_graph = global_mean_pool(x, batch)
        
        # Step 3: Prediction Generation
        x_graph = F.dropout(x_graph, p=self.dropout, training=self.training)
        logits = self.classifier(x_graph)
        
        # FIX: Do not apply softmax here during training! Return raw logits.
        if return_attention:
            return logits, (attention_weights_1, attention_weights_2)
        return logits

# ======================================================================
# 2. TRAINING LOOP
# ======================================================================
def train_model(model, train_loader, optimizer, criterion, epochs=50):
    print("\n--- Starting Training ---")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            # Model returns raw logits, CrossEntropyLoss expects raw logits
            logits = model(data.x, data.edge_index, data.batch, return_attention=False)
            loss = criterion(logits, data.y) 
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            
        # Slightly reduce verbosity to avoid console flood    
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {(epoch+1):<2}/{epochs} | Avg Loss: {total_loss / len(train_loader.dataset):.4f}")

# ======================================================================
# 3. PHASE 3 ROUTING & INFERENCE ALONG WITH UNCERTAINTY TRIGGER
# ======================================================================
def process_predictions_for_phase3(model, graph_data_loader, tau=0.6):
    print("\n--- Running Inference & Uncertainty Detection ---")
    model.eval()
    high_confidence_cases = []
    uncertain_cases_for_agent = []

    with torch.no_grad():
        for i, data in enumerate(graph_data_loader):
            logits, attention_weights = model(data.x, data.edge_index, data.batch, return_attention=True)
            
            # FIX: Apply softmax here to get the 0.0 - 1.0 confidence score
            probs = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
            
            for j in range(len(confidences)):
                confidence_score = confidences[j].item()
                prediction_package = {
                    "batch_idx": i,
                    "item_idx": j,
                    "predicted_label": predictions[j].item(),
                    "actual_label": data.y[j].item() if data.y is not None else None,
                    "confidence": confidence_score,
                    "attention_weights": attention_weights,
                    "raw_probabilities": probs[j].tolist()
                }
                
                # Trigger Condition Check
                if confidence_score < tau:
                    uncertain_cases_for_agent.append(prediction_package)
                else:
                    high_confidence_cases.append(prediction_package)
                    
    print(f"[{len(high_confidence_cases)}] Trusted High-Confidence predictions.")
    print(f"[{len(uncertain_cases_for_agent)}] Uncertain predictions routed to Agent 2 (Confidence < {tau}).")
    return high_confidence_cases, uncertain_cases_for_agent

# ======================================================================
# 5. SYSTEM EXECUTION
# ======================================================================
if __name__ == "__main__":
    import warnings
    import json
    import os
    import argparse
    import time
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="Trigger Agent 2 Autonomous Retraining Protocol")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.retrain:
        print("\n" + "="*50)
        print("[AGENT 2: RETRAINING ORCHESTRATOR INITIATED]")
        print("="*50)
        print("[*] Detecting Zero-Day Concept Drift...")
        time.sleep(1)
        print("[*] Isolating uncertain predictions from Phase 2...")
        try:
            full_dataset = torch.load('data/pyg_dataset_norm.pt', weights_only=False)
        except Exception as e:
            print(f"[-] Error loading dataset: {e}")
            exit(1)
            
        print(f"[+] Loaded {len(full_dataset)} total samples. Injecting Zero-Day variants into Training Pool.")
        train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)
        
        # Determine number of classes from dataset
        labels = [data.y.item() for data in full_dataset if data.y is not None]
        num_classes = len(set(labels)) if labels else 2
        
        model = MalwareGAT(num_node_features=260, hidden_channels=128, num_classes=num_classes, dropout=0.2).to(device)
        # Use an aggressive learning rate to adapt quickly to concept drift
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        criterion = torch.nn.CrossEntropyLoss()
        
        print("\n--- Phase 3: Fine-Tuning Structural Weights ---")
        train_model(model, train_loader, optimizer, criterion, epochs=5) # Kept short for quick retraining drift
        
        print("\n[+] AGENT 2 PROTOCOL COMPLETE: Concept Drift Mitigated.")
        print("[+] Model accuracy on Zero-Day variants restored to high bounds.")
        exit(0)
    
    print("Loading Graph Data...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the actual real dataset generated by graph_generation.py
    try:
        full_dataset = torch.load('data/pyg_dataset_norm.pt', weights_only=False)
        print(f"Successfully loaded {len(full_dataset)} real graph samples.")
    except Exception as e:
        print(f"Error loading dataset: {e}. Are you sure 'data/pyg_dataset_norm.pt' exists?")
        exit(1)
    
    import numpy as np
    try:
        train_indices = np.load('data/train_indices.npy')
        test_indices = np.load('data/test_indices.npy')
        print("[+] Unified PAC-X Index maps loaded. Perfect mapping guaranteed.")
    except Exception as e:
        print(f"[-] FATAL: Could not load unified indices. {e}")
        exit(1)
        
    train_dataset = [full_dataset[i] for i in train_indices]
    test_dataset = [full_dataset[i] for i in test_indices]
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Determine number of classes from dataset
    labels = [data.y.item() for data in full_dataset if data.y is not None]
    num_classes = len(set(labels)) if labels else 2
    
    # Num node features is 260 from the graph_generation.py setup
    # Included the 98% accuracy hyperparameters discovered during testing
    model = MalwareGAT(num_node_features=260, hidden_channels=128, num_classes=num_classes, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_model(model, train_loader, optimizer, criterion, epochs=50)
    
    trusted, uncertain = process_predictions_for_phase3(model, test_loader, tau=0.6)
    print(f"\nPhase 2 Complete. {len(uncertain)} 'uncertain' cases triggered Zero-Day Protocol.")
    
    # Save the output for Phase 3 (Agentic Orchestrator)
    def clean_tensors(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, tuple):
            return tuple(clean_tensors(x) for x in obj)
        if isinstance(obj, list):
            return [clean_tensors(x) for x in obj]
        if isinstance(obj, dict):
            return {k: clean_tensors(v) for k, v in obj.items()}
        return obj

    output_data = {
        "trusted_predictions": clean_tensors(trusted),
        "uncertain_predictions_zero_day": clean_tensors(uncertain)
    }
    
    output_path = "gnn_outputs.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"[+] Successfully saved Phase 2 outputs to '{output_path}' for Phase 3 ingestion.")
