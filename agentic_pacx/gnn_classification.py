import argparse
import copy
import json
import os
import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.nn import Linear
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

# ======================================================================
# 1. MODEL ARCHITECTURE (FIXED)
# ======================================================================
class MalwareGAT(torch.nn.Module):
    def __init__(self, num_node_features=260, hidden_channels=64, num_classes=2, num_heads=4):
        super(MalwareGAT, self).__init__()
        self.gat1 = GATConv(num_node_features, hidden_channels, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=2, concat=True)
        self.gat3 = GATConv(hidden_channels * 2, hidden_channels, heads=1, concat=False)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels * num_heads)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels * 2)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.classifier = Linear(hidden_channels, num_classes)
        self.dropout = torch.nn.Dropout(p=0.25)

    def forward(self, x, edge_index, batch, return_attention=False):
        # Layer 1
        if return_attention:
            x, attention_weights_1 = self.gat1(x, edge_index, return_attention_weights=True)
        else:
            x = self.gat1(x, edge_index)
            
        x = F.elu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        
        # Layer 2
        if return_attention:
            x, attention_weights_2 = self.gat2(x, edge_index, return_attention_weights=True)
        else:
            x = self.gat2(x, edge_index)
            
        x = F.elu(x)
        x = self.bn2(x)
        x = self.dropout(x)

        # Layer 3
        if return_attention:
            x, attention_weights_3 = self.gat3(x, edge_index, return_attention_weights=True)
        else:
            x = self.gat3(x, edge_index)

        x = F.elu(x)
        x = self.bn3(x)
        x = self.dropout(x)
        
        # Aggregate the 4 nodes into 1 graph-level tensor
        x_graph = global_mean_pool(x, batch)
        
        # Step 3: Prediction Generation
        logits = self.classifier(x_graph)
        
        # FIX: Do not apply softmax here during training! Return raw logits.
        if return_attention:
            return logits, (attention_weights_1, attention_weights_2, attention_weights_3)
        return logits

# ======================================================================
# 2. TRAINING LOOP
# ======================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_split(dataset, train_ratio=0.8, seed=42):
    grouped_indices = {}
    for idx, sample in enumerate(dataset):
        if sample.y is None:
            continue
        label = int(sample.y.item())
        grouped_indices.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    train_indices = []
    test_indices = []
    for label_indices in grouped_indices.values():
        rng.shuffle(label_indices)
        split_idx = max(1, int(len(label_indices) * train_ratio))
        split_idx = min(split_idx, len(label_indices) - 1) if len(label_indices) > 1 else 1
        train_indices.extend(label_indices[:split_idx])
        test_indices.extend(label_indices[split_idx:] if len(label_indices) > 1 else label_indices[:1])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    train_dataset = [dataset[i] for i in train_indices]
    test_dataset = [dataset[i] for i in test_indices]
    return train_dataset, test_dataset, np.array(test_indices, dtype=np.int64)


def build_weighted_sampler(dataset, num_classes=11):
    labels = [int(sample.y.item()) for sample in dataset if sample.y is not None]
    class_counts = np.zeros(num_classes, dtype=np.int64)
    for label in labels:
        if 0 <= label < num_classes:
            class_counts[label] += 1
    class_weights = 1.0 / np.clip(class_counts, 1, None)
    sample_weights = [class_weights[int(sample.y.item())] for sample in dataset if sample.y is not None]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler, class_weights


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch, return_attention=False)
            loss = criterion(logits, data.y)
            total_loss += loss.item() * data.num_graphs
            pred = torch.argmax(logits, dim=1)
            y_true.extend(data.y.detach().cpu().tolist())
            y_pred.extend(pred.detach().cpu().tolist())

    if not y_true:
        return {"loss": 0.0, "accuracy": 0.0, "f1_macro": 0.0}

    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    avg_loss = total_loss / max(1, len(data_loader.dataset))
    return {"loss": avg_loss, "accuracy": accuracy, "f1_macro": f1_macro}


def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, device, epochs=35, patience=8):
    print("\n--- Starting Training ---")
    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index, data.batch, return_attention=False)
            loss = criterion(logits, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running_loss += loss.item() * data.num_graphs

        train_loss = running_loss / max(1, len(train_loader.dataset))
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy'] * 100:.2f}% | "
            f"Val F1(macro): {val_metrics['f1_macro']:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_accuracy": round(val_metrics["accuracy"] * 100, 2),
            "val_f1_macro": round(val_metrics["f1_macro"] * 100, 2),
        })

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"[+] Early stopping triggered at epoch {epoch + 1}.")
                break

    model.load_state_dict(best_state)
    return model, history

# ======================================================================
# 3. PHASE 3 ROUTING & INFERENCE ALONG WITH UNCERTAINTY TRIGGER
# ======================================================================
def process_predictions_for_phase3(model, graph_data_loader, tau=0.6):
    print("\n--- Running Inference & Uncertainty Detection ---")
    model.eval()
    high_confidence_cases = []
    uncertain_cases_for_agent = []

    device = next(model.parameters()).device
    with torch.no_grad():
        for i, data in enumerate(graph_data_loader):
            data = data.to(device)
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
    warnings.filterwarnings("ignore")
    set_seed(42)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="Trigger Agent 2 Autonomous Retraining Protocol")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.retrain:
        print("\n" + "="*50)
        print("[AGENT 2: RETRAINING ORCHESTRATOR INITIATED]")
        print("="*50)
        
        pool_path = "data/retrain_pool.pt"
        if not os.path.exists(pool_path):
            print("[*] No custom retraining pool found. Recalibrating on baseline dataset...")
            try:
                retrain_dataset = torch.load("data/pyg_dataset_norm.pt", weights_only=False)
            except Exception:
                exit(1)
        else:
            print(f"[*] Loading {pool_path} for Zero-Day structural adaptation...")
            retrain_dataset = torch.load(pool_path, weights_only=False)

        retrain_dataset = [sample for sample in retrain_dataset if getattr(sample, "y", None) is not None]
        if not retrain_dataset:
            print("[*] Retrain pool has no labeled samples. Falling back to baseline dataset.")
            retrain_dataset = torch.load("data/pyg_dataset_norm.pt", weights_only=False)

        # Guardrails: refuse to fine-tune on a pool too small or too
        # one-sided to produce a meaningful update -- this is exactly what
        # let a 2-sample, single-class pool silently overwrite a good model.
        MIN_RETRAIN_SAMPLES = 20
        MIN_RETRAIN_CLASSES = 2
        pool_labels = set(int(s.y.item()) for s in retrain_dataset)
        if len(retrain_dataset) < MIN_RETRAIN_SAMPLES or len(pool_labels) < MIN_RETRAIN_CLASSES:
            print(
                f"[-] Retrain pool too small/uniform to fine-tune safely "
                f"({len(retrain_dataset)} samples, {len(pool_labels)} classes; "
                f"need >= {MIN_RETRAIN_SAMPLES} samples and >= {MIN_RETRAIN_CLASSES} classes). "
                f"Skipping retrain -- model.pt left unchanged."
            )
            exit(1)

        print(f"[+] Total labeled samples for fine-tuning: {len(retrain_dataset)}")
        sampler, class_weights = build_weighted_sampler(retrain_dataset)
        train_loader = DataLoader(retrain_dataset, batch_size=16, sampler=sampler)

        # Validate against the REAL held-out test split, not the pool itself --
        # validating on the same data being trained on trivially hits 100%
        # and can't catch a fine-tune that's actually hurting generalization.
        full_dataset = torch.load("data/pyg_dataset_norm.pt", weights_only=False)
        test_index_path = os.path.join("data", "test_indices.npy")
        if os.path.exists(test_index_path):
            real_test_indices = np.load(test_index_path)
            real_test_dataset = [full_dataset[int(i)] for i in real_test_indices]
        else:
            real_test_dataset = full_dataset
        val_loader = DataLoader(real_test_dataset, batch_size=64, shuffle=False)

        model = MalwareGAT(num_node_features=260, num_classes=11).to(device)
        pre_retrain_state = None
        if os.path.exists("model.pt"):
            state_dict = torch.load("model.pt", map_location=device)
            pre_retrain_state = state_dict
            try:
                model.load_state_dict(state_dict)
                print("[+] Loaded existing weights for transfer learning.")
            except Exception:
                # Allow smooth architecture evolution while keeping compatible layers.
                model.load_state_dict(state_dict, strict=False)
                print("[*] Loaded partial compatible weights (architecture updated).")

        pre_retrain_metrics = evaluate_model(model, val_loader, torch.nn.CrossEntropyLoss(), device)
        print(f"[*] Pre-retrain accuracy on real held-out test set: {pre_retrain_metrics['accuracy'] * 100:.2f}%")

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.6, patience=2)
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

        print("\n--- Phase 3: Fine-Tuning Structural Weights ---")
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            epochs=10,
            patience=4,
        )

        post_retrain_metrics = evaluate_model(model, val_loader, torch.nn.CrossEntropyLoss(), device)
        print(f"[*] Post-retrain accuracy on real held-out test set: {post_retrain_metrics['accuracy'] * 100:.2f}%")

        # Safety net: don't let a bad fine-tune silently replace a good model.
        ACCURACY_DROP_TOLERANCE = 0.03
        if pre_retrain_state is not None and post_retrain_metrics["accuracy"] < pre_retrain_metrics["accuracy"] - ACCURACY_DROP_TOLERANCE:
            print(
                f"[-] Retrained model is worse on the real test set "
                f"({post_retrain_metrics['accuracy']*100:.2f}% vs {pre_retrain_metrics['accuracy']*100:.2f}% before). "
                f"Discarding this update -- model.pt left unchanged."
            )
            exit(1)

        torch.save(model.state_dict(), "model.pt")
        with open("training_history.json", "w") as f:
            json.dump({"run_type": "agent2_retrain", "history": history}, f, indent=2)

        # gnn_outputs.json was generated by the last full training run and is
        # now stale relative to these updated weights -- remove it so
        # /api/report falls back to live re-evaluation against the new model
        # instead of silently serving pre-retrain cached numbers.
        if os.path.exists("gnn_outputs.json"):
            os.remove("gnn_outputs.json")

        print("\n[+] AGENT 2 PROTOCOL COMPLETE: Model weights updated with new pattern knowledge.")
        exit(0)
    
    print("Loading Graph Data...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the actual real dataset generated by graph_generation.py
    try:
        full_dataset = torch.load("data/pyg_dataset_norm.pt", weights_only=False)
        print(f"Successfully loaded {len(full_dataset)} real graph samples.")
    except Exception as e:
        print(f"Error loading dataset: {e}. Are you sure 'data/pyg_dataset_norm.pt' exists?")
        exit(1)
    
    train_dataset, test_dataset, test_indices = stratified_split(full_dataset, train_ratio=0.8, seed=42)
    train_indices = np.array(sorted(set(range(len(full_dataset))) - set(test_indices.tolist())), dtype=np.int64)
    os.makedirs("data", exist_ok=True)
    np.save(os.path.join("data", "test_indices.npy"), test_indices)
    np.save(os.path.join("data", "train_indices.npy"), train_indices)

    sampler, class_weights = build_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Determine number of classes from dataset
    labels = [data.y.item() for data in full_dataset if data.y is not None]
    num_classes = len(set(labels)) if labels else 2
    
    # Num node features is 260 from the graph_generation.py setup
    model = MalwareGAT(num_node_features=260, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        epochs=35,
        patience=8,
    )

    with open("training_history.json", "w") as f:
        json.dump({"run_type": "full_train", "history": history}, f, indent=2)

    final_metrics = evaluate_model(model, test_loader, criterion, device)
    print(
        f"[+] Final Test Metrics | "
        f"Accuracy: {final_metrics['accuracy'] * 100:.2f}% | "
        f"F1(macro): {final_metrics['f1_macro']:.4f}"
    )
    
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
    
    # Save the trained model weights for live inference
    torch.save(model.state_dict(), "model.pt")
    print("[+] Model weights saved locally to 'model.pt'")
