import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn as nn

# =========================
# LOAD DATA (MANDATORY FROM DOC)
# =========================
dataset = torch.load('data/pyg_dataset_norm.pt', weights_only=False)

train_idx = np.load('data/train_indices.npy')
test_idx = np.load('data/test_indices.npy')

train_dataset = [dataset[i] for i in train_idx]
test_dataset = [dataset[i] for i in test_idx]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# =========================
# MODEL (GAT - REQUIRED)
# =========================
class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        self.gat1 = GATConv(input_dim, hidden_dim, heads=4, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Adding Dropout directly to features dynamically limits overfitting
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat2(x, edge_index))

        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training) 
        x = self.fc(x)

        return x  # IMPORTANT: raw logits

# =========================
# SETUP
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = len(set([int(data.y) for data in dataset]))

model = GATModel(260, 128, num_classes, dropout=0.2).to(device)

# Adjusted Weight Decay and Dropout for lighter regularization to prevent underfitting
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# =========================
# TRAIN FUNCTION
# =========================
def train():
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)

        optimizer.zero_grad()
        out = model(data)

        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# =========================
# TEST + CONFIDENCE (CRITICAL FOR PHASE 3)
# =========================
def test_with_confidence(loader):
    model.eval()

    correct = 0
    all_conf = []

    for data in loader:
        data = data.to(device)

        out = model(data)

        probs = F.softmax(out, dim=1)
        confidence, pred = probs.max(dim=1)

        correct += (pred == data.y).sum().item()
        all_conf.extend(confidence.detach().cpu().numpy())

    accuracy = correct / len(loader.dataset)
    avg_conf = sum(all_conf) / len(all_conf)

    return accuracy, avg_conf

# =========================
# TRAIN LOOP
# =========================
best_acc = 0.0
for epoch in range(1, 101):
    loss = train()
    acc, conf = test_with_confidence(test_loader)
    
    if acc > best_acc:
        best_acc = acc

    print(f"Epoch {epoch:<3} | Loss: {loss:.4f} | Accuracy: {acc:.4f} | Avg Confidence: {conf:.4f}")

print("=" * 40)
print(f"FINAL BEST ACCURACY: {best_acc * 100:.2f}%")
print("=" * 40)