
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score


# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="B4 - Train (STAC)", page_icon="🧠", layout="wide")
st.title("🧠 Track B — Train Deep Model (STAC Urban Classification)")


# ============================================================
# Check prepared dataset
# ============================================================
if "B_stac_df" not in st.session_state:
    st.error("❌ No prepared dataset found.")
    st.info("➡️ Please run **B3 (Prep STAC)** first and click **Save dataset for training**.")
    st.stop()

df = st.session_state["B_stac_df"].copy()

if "B_features" not in st.session_state or "B_target" not in st.session_state:
    st.error("❌ Missing features/target selection in session_state.")
    st.info("➡️ Please re-run B3 and save dataset again.")
    st.stop()

features = st.session_state["B_features"]
target = st.session_state["B_target"]


# ============================================================
# Explanation
# ============================================================
st.markdown("""
## ✅ Goal
We train a Deep Learning model to predict:

**Built-up (urban) vs Not built-up**

from STAC-derived features such as:
- NDVI
- NDBI
- optional row/col

This is the **ML step** that improves over simple thresholds.
""")


# ============================================================
# Data + Split
# ============================================================
st.subheader("📦 Train/Validation split")

X = df[features].values.astype(np.float32)
y = df[target].values.astype(np.float32).reshape(-1, 1)

c1, c2, c3 = st.columns(3)
with c1:
    test_size = st.slider("Validation split", 0.1, 0.4, 0.2, 0.05)
with c2:
    seed = st.number_input("Random seed", 0, 9999, 42)
with c3:
    scale_features = st.checkbox("Simple feature scaling (recommended)", value=True)

# Simple scaling (fast + explainable)
# NDVI and NDBI are typically in [-1, 1], row/col are large => scale row/col
X_scaled = X.copy()
if scale_features:
    for i, col in enumerate(features):
        if col in ["row", "col"]:
            max_val = np.max(X_scaled[:, i])
            if max_val > 0:
                X_scaled[:, i] = X_scaled[:, i] / max_val  # normalize to [0,1]

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled,
    y,
    test_size=test_size,
    random_state=int(seed),
    stratify=y
)

# Convert to torch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)

st.success(f"✅ Train size: {len(X_train_t)} | Val size: {len(X_val_t)}")


# ============================================================
# Model
# ============================================================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # logits output
        )

    def forward(self, x):
        return self.net(x)


def evaluate_model(model, X_val_t, y_val_t, threshold=0.5):
    model.eval()
    with torch.no_grad():
        logits = model(X_val_t).cpu().numpy().flatten()
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= threshold).astype(int)

    y_true = y_val_t.cpu().numpy().flatten().astype(int)

    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    cm = confusion_matrix(y_true, preds)

    return acc, f1, prec, rec, cm


def train_one_config(lr, hidden_dim, dropout, epochs, batch_size):
    model = MLP(input_dim=X_train_t.shape[1], hidden_dim=hidden_dim, dropout=dropout)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True
    )

    train_losses = []
    val_f1s = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / len(loader)
        train_losses.append(avg_loss)

        # validation tracking each epoch (for visualization)
        acc, f1, prec, rec, cm = evaluate_model(model, X_val_t, y_val_t)
        val_f1s.append(f1)

    acc, f1, prec, rec, cm = evaluate_model(model, X_val_t, y_val_t)
    return model, train_losses, val_f1s, acc, f1, prec, rec, cm


# ============================================================
# Hyperparameter tuning UI
# ============================================================
st.subheader("🧪 Hyperparameter tuning (grid search)")

st.markdown("""
We test multiple combinations of:

- **Learning rate**
- **Hidden dimension**
- **Dropout**

Then we pick the best configuration using **F1 score** (good for imbalanced classes).
""")

c1, c2, c3, c4 = st.columns(4)
with c1:
    lr_list = st.multiselect("Learning rates", [1e-4, 3e-4, 1e-3, 3e-3], default=[3e-4, 1e-3])
with c2:
    hidden_list = st.multiselect("Hidden dim", [32, 64, 128, 256], default=[64, 128])
with c3:
    drop_list = st.multiselect("Dropout", [0.0, 0.1, 0.2, 0.3], default=[0.0, 0.1])
with c4:
    batch_size = st.selectbox("Batch size", [32, 64, 128], index=1)

epochs = st.slider("Epochs per experiment", 3, 40, 10)


# ============================================================
# Run tuning
# ============================================================
if st.button("🚀 Run tuning"):
    total = len(lr_list) * len(hidden_list) * len(drop_list)
    if total == 0:
        st.error("❌ Please select values for lr / hidden_dim / dropout.")
        st.stop()

    st.info(f"Running {total} experiments...")

    progress = st.progress(0)
    status = st.empty()
    chart_placeholder = st.empty()

    best_f1 = -1
    best_model = None
    best_cfg = None
    best_cm = None
    best_history = None

    history_rows = []

    run_id = 0
    t0 = time.perf_counter()

    for lr in lr_list:
        for hd in hidden_list:
            for dr in drop_list:
                run_id += 1
                status.write(f"🧪 Experiment {run_id}/{total} → lr={lr}, hidden={hd}, dropout={dr}")

                model, train_losses, val_f1s, acc, f1, prec, rec, cm = train_one_config(
                    lr=lr,
                    hidden_dim=hd,
                    dropout=dr,
                    epochs=epochs,
                    batch_size=batch_size
                )

                history_rows.append({
                    "lr": lr,
                    "hidden_dim": hd,
                    "dropout": dr,
                    "acc": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1
                })

                # live chart: best val f1 over epochs
                if (best_history is None) or (f1 > best_f1):
                    best_f1 = f1
                    best_model = model
                    best_cfg = {"lr": lr, "hidden_dim": hd, "dropout": dr}
                    best_cm = cm
                    best_history = {
                        "train_losses": train_losses,
                        "val_f1s": val_f1s
                    }

                    # show live plot of best experiment so far
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.plot(train_losses, label="train loss")
                    ax.set_title("Best experiment so far: Training loss")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.legend()
                    chart_placeholder.pyplot(fig)

                progress.progress(run_id / total)

    elapsed = time.perf_counter() - t0
    status.success(f"✅ Tuning finished in {elapsed:.2f} seconds")

    hist_df = pd.DataFrame(history_rows).sort_values("f1", ascending=False)

    st.subheader("📋 Experiment history (sorted by F1)")
    st.dataframe(hist_df, use_container_width=True)

    st.subheader("🏆 Best model selected")
    st.write("Best config:", best_cfg)
    st.write(f"Best F1 score: **{best_f1:.4f}**")

    # Confusion matrix
    st.subheader("🧾 Confusion Matrix (best model)")
    st.write(best_cm)
    st.caption("Rows = true labels, Columns = predicted labels")

    # Save best model and metadata in session state
    st.session_state["B_best_model"] = best_model
    st.session_state["B_best_cfg"] = best_cfg
    st.session_state["B_best_history"] = best_history
    st.session_state["B_features"] = features  # store in case needed later
    st.session_state["B_target"] = target

    st.success("✅ Best model saved to session_state: B_best_model")
    st.info("➡️ Next: go to **B5 Export ONNX**")
