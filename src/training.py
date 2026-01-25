import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.model import BuiltUpNet

def train_one_model(
    X_train_t, y_train_t, X_val_t, y_val_t,
    lr=1e-3, hidden_dim=64, dropout=0.1,
    epochs=10, batch_size=64
):
    model = BuiltUpNet(input_dim=X_train_t.shape[1], hidden_dim=hidden_dim, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True
    )

    losses = []
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(train_loader))

    # validation metrics
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t).numpy().flatten()
        val_probs = 1 / (1 + np.exp(-val_logits))
        val_pred = (val_probs >= 0.5).astype(int)

    y_val_np = y_val_t.numpy().flatten().astype(int)

    acc = accuracy_score(y_val_np, val_pred)
    f1 = f1_score(y_val_np, val_pred)
    cm = confusion_matrix(y_val_np, val_pred)

    # val loss
    with torch.no_grad():
        val_logits_t = model(X_val_t)
        val_loss = float(loss_fn(val_logits_t, y_val_t).item())

    return model, losses, val_loss, acc, f1, cm
