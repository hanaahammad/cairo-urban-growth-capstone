import os
import time
import streamlit as st
import numpy as np
import pandas as pd

import torch
from sklearn.model_selection import train_test_split

from src.training import train_one_model
from src.viz import plot_loss

st.set_page_config(page_title="Train DL Model", page_icon="🧠", layout="wide")
st.title("🧠 Train Deep Learning Model + Hyperparameter Tuning")

if "patch_df" not in st.session_state:
    st.error("❌ No dataset found. Please go to Page 3 first.")
    st.stop()

df = st.session_state["patch_df"].copy()

st.subheader("📄 Dataset preview")
st.dataframe(df.head())

st.markdown("""
## 🎯 What we do here

We train a small neural network classifier to detect:

- 1 = built-up
- 0 = non-built-up

And we do **hyperparameter tuning** like in the course:

✅ try many configs  
✅ keep the best based on validation loss  
""")

feature_cols = [c for c in df.columns if c not in ["label"]]

X = df[feature_cols].values.astype(np.float32)
y = df["label"].values.astype(np.float32).reshape(-1, 1)

test_size = st.slider("Validation size", 0.1, 0.4, 0.2, 0.05)
seed = st.number_input("Random seed", 0, 9999, 42)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=test_size, random_state=int(seed), stratify=y
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)

st.info(f"Train size: {len(X_train)} | Val size: {len(X_val)}")

st.subheader("🧪 Hyperparameter tuning (Grid Search)")

colA, colB, colC = st.columns(3)

with colA:
    lr_list = st.multiselect(
        "Learning rates",
        options=[1e-4, 3e-4, 1e-3, 3e-3],
        default=[3e-4, 1e-3],
    )
with colB:
    hidden_list = st.multiselect(
        "Hidden sizes",
        options=[32, 64, 128, 256],
        default=[64, 128],
    )
with colC:
    drop_list = st.multiselect(
        "Dropout",
        options=[0.0, 0.1, 0.2, 0.3],
        default=[0.0, 0.1],
    )

epochs = st.slider("Epochs per experiment", 3, 30, 8)
batch_size = st.selectbox("Batch size", [32, 64, 128], index=1)

with st.expander("🧾 Show tuning code"):
    st.code("""
for lr in lr_list:
  for hidden_dim in hidden_list:
    for dropout in drop_list:
      model, losses, val_loss, acc, f1, cm = train_one_model(...)
      log results
      keep best model
""", language="python")

run = st.button("🚀 Run tuning")

if run:
    if len(lr_list) == 0 or len(hidden_list) == 0 or len(drop_list) == 0:
        st.error("❌ Choose at least one lr, hidden_dim and dropout value.")
        st.stop()

    total_runs = len(lr_list) * len(hidden_list) * len(drop_list)
    st.info(f"Total experiments: {total_runs}")

    progress = st.progress(0)
    status = st.empty()

    best_val_loss = float("inf")
    best_model = None
    best_row = None
    best_losses = None

    results = []
    run_id = 0
    t0 = time.time()

    for lr in lr_list:
        for hidden_dim in hidden_list:
            for dropout in drop_list:
                run_id += 1
                status.write(f"Experiment {run_id}/{total_runs} → lr={lr}, hidden={hidden_dim}, dropout={dropout}")

                model, losses, val_loss, acc, f1, cm = train_one_model(
                    X_train_t, y_train_t, X_val_t, y_val_t,
                    lr=lr,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    epochs=epochs,
                    batch_size=batch_size
                )

                results.append({
                    "lr": lr,
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "val_loss": val_loss,
                    "val_acc": acc,
                    "val_f1": f1
                })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model
                    best_row = results[-1]
                    best_losses = losses

                progress.progress(run_id / total_runs)

    elapsed = time.time() - t0
    st.success(f"✅ Tuning done in {elapsed:.1f} sec")

    results_df = pd.DataFrame(results).sort_values("val_loss", ascending=True)

    st.session_state["tuning_results"] = results_df
    st.session_state["best_model"] = best_model
    st.session_state["best_config"] = best_row

    st.subheader("📋 Experiment results")
    st.dataframe(results_df)

    st.subheader("🏆 Best config")
    st.write(best_row)
    st.write(f"Best val_loss = **{best_val_loss:.5f}**")

    st.subheader("📉 Best training loss curve")
    st.pyplot(plot_loss(best_losses, "Best model training loss"))

    # Save model
    if st.button("💾 Save best model to models/best_model.pt"):
        os.makedirs("models", exist_ok=True)
        torch.save(best_model.state_dict(), "models/best_model.pt")
        st.success("Saved: models/best_model.pt ✅")
