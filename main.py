import kagglehub
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model.LSTM import LSTM


# Download data
path = kagglehub.dataset_download("faresls/fd001-prepared-data")
df = pd.read_csv(os.path.join(path, "train_FD001_prepared.csv"))

# Labels
h = 30
df["label"] = (df["RUL"] <= h).astype(int)
sensor_cols = [col for col in df.columns if col not in ["unit_nr", "cycle", "RUL", "label"]]

# Sliding window
w = 30

X, y, groups = [], [], []
for unit, group in df.groupby("unit_nr"):
    data = group[sensor_cols].values
    labels = group["label"].values
    for i in range(len(data) - w):
        X.append(data[i:i+w])
        y.append(labels[i+w])
        groups.append(unit)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)
groups = np.array(groups)

# Split into train and test datasets
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Make dataloaders
class EngineDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y  = torch.tensor(y)
    def __len__(self): 
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
train_loader = DataLoader(EngineDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader  = DataLoader(EngineDataset(X_test,  y_test),  batch_size=64, shuffle=False)

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = LSTM(n_features=len(sensor_cols)).to(device)

# Weighted loss
pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
epochs = 20
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} — Loss: {total_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
all_probs, all_labels = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y_batch.numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# Alert threshold
threshold = 0.5
preds = (all_probs >= threshold).astype(int)

print("\n── Classification Report ──")
print(classification_report(all_labels, preds, target_names=["Normal", "Incident"]))
print(f"ROC-AUC: {roc_auc_score(all_labels, all_probs):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, preds))