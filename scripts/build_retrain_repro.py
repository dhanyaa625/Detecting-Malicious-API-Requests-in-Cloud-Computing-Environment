"""
Controlled reproduction of the catastrophic-forgetting failure mode that
motivated Agent 2's retrain guardrails (agentic_pacx/gnn_classification.py,
`MIN_RETRAIN_SAMPLES` / `MIN_RETRAIN_CLASSES` / `ACCURACY_DROP_TOLERANCE`).

It deliberately bypasses those guardrails and fine-tunes a copy of the live
model on a tiny, single-class pool (2 Benign samples), using the exact same
training call (`train_model`, `evaluate_model`, same optimizer/scheduler
settings, same real held-out test split) as the production `--retrain` path
in agentic_pacx/gnn_classification.py -- just without the guardrail checks.

This script only reads model.pt and never writes it: it fine-tunes a
deep-copied model instance and saves results to
data/retrain_repro/retrain_repro_report.json. The live model.pt,
gnn_outputs.json, and training_history.json are never touched.

Usage: python scripts/build_retrain_repro.py
"""
import copy
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_pacx.gnn_classification import (
    MalwareGAT,
    build_weighted_sampler,
    evaluate_model,
    set_seed,
    train_model,
)
from torch_geometric.data import DataLoader


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = torch.load("data/pyg_dataset_norm.pt", weights_only=False)
    train_indices = np.load("data/train_indices.npy")
    test_indices = np.load("data/test_indices.npy")

    class_labels = json.load(open("data/class_labels.json"))
    benign_idx = class_labels.index("benign")

    # Deliberately tiny, single-class pool: 2 Benign samples pulled from the
    # TRAIN split only (never the held-out test split, to avoid leakage into
    # the accuracy numbers this script reports).
    benign_train_samples = [
        full_dataset[int(i)] for i in train_indices
        if int(full_dataset[int(i)].y.item()) == benign_idx
    ]
    if len(benign_train_samples) < 2:
        print("[-] Not enough Benign training samples to build the 2-sample pool.")
        sys.exit(1)
    tiny_pool = benign_train_samples[:2]

    sampler, class_weights = build_weighted_sampler(tiny_pool, num_classes=len(class_labels))
    train_loader = DataLoader(tiny_pool, batch_size=16, sampler=sampler)

    real_test_dataset = [full_dataset[int(i)] for i in test_indices]
    val_loader = DataLoader(real_test_dataset, batch_size=64, shuffle=False)

    model = MalwareGAT(num_node_features=260, num_classes=len(class_labels)).to(device)
    if not os.path.exists("model.pt"):
        print("[-] model.pt not found -- train the base model first.")
        sys.exit(1)
    state_dict = torch.load("model.pt", map_location=device)
    model.load_state_dict(state_dict)

    pre_metrics = evaluate_model(model, val_loader, torch.nn.CrossEntropyLoss(), device)
    print(f"[*] Pre-retrain accuracy on real held-out test set: {pre_metrics['accuracy'] * 100:.2f}%")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.6, patience=2)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    print("[!] GUARDRAILS DELIBERATELY BYPASSED for this repro "
          f"(pool: {len(tiny_pool)} samples, 1 class) -- production code would refuse this pool.")
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

    post_metrics = evaluate_model(model, val_loader, torch.nn.CrossEntropyLoss(), device)
    print(f"[*] Post-retrain (best-epoch) accuracy on real held-out test set: {post_metrics['accuracy'] * 100:.2f}%")

    mid_run_accuracies = [h["val_accuracy"] for h in history]
    worst_epoch = min(history, key=lambda h: h["val_accuracy"])
    print(f"[*] Worst mid-run epoch: epoch {worst_epoch['epoch']}, {worst_epoch['val_accuracy']:.2f}%")

    report = {
        "run_type": "unguarded_retrain_repro",
        "note": (
            "Guardrails (MIN_RETRAIN_SAMPLES=20, MIN_RETRAIN_CLASSES=2, "
            "ACCURACY_DROP_TOLERANCE=0.03) deliberately bypassed to reproduce "
            "the catastrophic-forgetting failure mode they exist to prevent. "
            "Fine-tunes a deep copy of the live model -- model.pt is never "
            "overwritten by this script."
        ),
        "pool_size": len(tiny_pool),
        "pool_classes": 1,
        "pre_retrain_accuracy": round(pre_metrics["accuracy"] * 100, 2),
        "post_retrain_best_epoch_accuracy": round(post_metrics["accuracy"] * 100, 2),
        "worst_mid_run_epoch": worst_epoch["epoch"],
        "worst_mid_run_accuracy": worst_epoch["val_accuracy"],
        "per_epoch_val_accuracy": mid_run_accuracies,
        "full_epoch_history": history,
    }

    os.makedirs("data/retrain_repro", exist_ok=True)
    report_path = "data/retrain_repro/retrain_repro_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[+] Saved report to {report_path}")


if __name__ == "__main__":
    main()
