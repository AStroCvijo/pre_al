import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from data.dataset import prepare_data
from model.LSTM import LSTM
from utils.args import parse_args


args = parse_args()

train_loader, test_loader, y_train, n_features = prepare_data(
    h=args.rul_threshold,
    w=args.window,
    test_size=args.test_size,
    batch_size=args.batch_size,
    random_state=args.seed,
)

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = LSTM(
    n_features=n_features,
    hidden_size=args.hidden_size,
    num_layers=args.num_layers,
    dropout=args.dropout,
).to(device)

# Weighted loss
pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Training
epochs = args.epochs
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
threshold = args.threshold
preds = (all_probs >= threshold).astype(int)

print(f"\n── Classification Report (threshold={threshold}) ──")
print(classification_report(all_labels, preds, target_names=["Normal", "Incident"]))
print(f"ROC-AUC: {roc_auc_score(all_labels, all_probs):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, preds))

# Threshold sweep — precision/recall trade-off for alerting
from sklearn.metrics import precision_score, recall_score, f1_score

thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
print("\n── Threshold Sweep ──")
print(f"{'Threshold':>10} | {'Precision':>9} | {'Recall':>6} | {'F1':>6}")
print("-" * 44)
best_f1, best_thresh = 0, 0.5
for t in thresholds:
    p = (all_probs >= t).astype(int)
    pr = precision_score(all_labels, p, zero_division=0)
    re = recall_score(all_labels, p, zero_division=0)
    f1 = f1_score(all_labels, p, zero_division=0)
    print(f"{t:>10.2f} | {pr:>9.4f} | {re:>6.4f} | {f1:>6.4f}")
    if f1 > best_f1:
        best_f1, best_thresh = f1, t
