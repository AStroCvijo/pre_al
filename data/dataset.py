import kagglehub
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit


class EngineDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def prepare_data(h=30, w=30, test_size=0.2, batch_size=64, random_state=42):
    path = kagglehub.dataset_download("faresls/fd001-prepared-data")
    df = pd.read_csv(os.path.join(path, "train_FD001_prepared.csv"))

    df["label"] = (df["RUL"] <= h).astype(int)
    sensor_cols = [col for col in df.columns if col not in ["unit_nr", "cycle", "RUL", "label"]]

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

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_loader = DataLoader(EngineDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(EngineDataset(X_test,  y_test),  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, y_train, len(sensor_cols)
