import os
import numpy as np
import streamlit as st

st.set_page_config(page_title="B6 - Deploy & Test (ONNX)", page_icon="🚀", layout="wide")
st.title("🚀 Track B — Deploy & Test the Model (ONNX)")


st.markdown("""
## ✅ What does “deployment” mean here?

In this capstone, deployment means:

✅ We exported the best model to **ONNX** (B5)  
✅ We run inference using **onnxruntime** inside Streamlit Cloud  
✅ Reviewers can test it directly in the browser

So instead of deploying a separate API, we provide a **fully working online demo**.

---

## 🧠 Two prediction methods in this app

### ✅ 1) Baseline rule (threshold method)
Built-up pixel if:

- **NDBI > threshold**
- **NDVI < threshold**

### ✅ 2) Deep Learning model (ONNX)
The model learns patterns from your training data and returns:

- probability of being built-up
- predicted class (0/1)

✅ This is often more stable than manual thresholds.
""")

st.divider()

# ============================================================
# Model file check
# ============================================================
st.subheader("📦 Load ONNX model")

MODEL_PATH = "models/urban_growth_best_model.onnx"
META_PATH = "models/model_metadata.txt"

if not os.path.exists(MODEL_PATH):
    st.error("❌ ONNX model file not found.")
    st.write(f"Expected path: `{MODEL_PATH}`")
    st.info("➡️ Train in B4, export in B5, then come back here.")
    st.stop()

st.success(f"✅ ONNX model found: `{MODEL_PATH}`")

# ============================================================
# Load ONNX runtime
# ============================================================
try:
    import onnxruntime as ort
except Exception as e:
    st.error("❌ onnxruntime is missing.")
    st.info("Add `onnxruntime` to requirements.txt")
    st.exception(e)
    st.stop()

sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# ============================================================
# Feature order
# ============================================================
st.subheader("🧾 Features used by the model")

features = None
if os.path.exists(META_PATH):
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = f.read()
    st.text(meta)

# Use session_state if available (best)
if "B_features" in st.session_state:
    features = st.session_state["B_features"]

if features is None:
    features = ["ndvi", "ndbi", "row", "col"]  # fallback
    st.warning("⚠️ Feature list not found in session_state, using default order:")
    st.write(features)
else:
    st.success("✅ Feature order loaded from session_state:")
    st.write(features)

st.caption("⚠️ Feature order matters! It must match training in B3/B4.")

st.divider()

# ============================================================
# Test pixel input
# ============================================================
st.subheader("🧪 Test a sample pixel")

st.markdown("""
### ✅ How to interpret NDVI / NDBI quickly

- **NDVI**:
  - high (≈ 0.4 to 0.8) → vegetation
  - low (≈ 0.0 to 0.2) → urban/desert/roads

- **NDBI**:
  - higher → built-up surfaces
  - lower → vegetation / water / non-urban
""")

col1, col2, col3, col4 = st.columns(4)

with col1:
    ndvi = st.slider("NDVI", -0.2, 0.9, 0.10, 0.01)
with col2:
    ndbi = st.slider("NDBI", -0.2, 0.5, 0.20, 0.01)
with col3:
    row = st.slider("row (scaled 0→1)", 0.0, 1.0, 0.50, 0.01)
with col4:
    col = st.slider("col (scaled 0→1)", 0.0, 1.0, 0.50, 0.01)

# Baseline thresholds
st.subheader("📏 Baseline thresholds (rule-based)")
t1, t2 = st.columns(2)
with t1:
    ndbi_thr = st.slider("Baseline NDBI threshold", -0.2, 0.5, 0.10, 0.01)
with t2:
    ndvi_thr = st.slider("Baseline NDVI threshold", -0.2, 0.9, 0.30, 0.01)

baseline_pred = int((ndbi > ndbi_thr) and (ndvi < ndvi_thr))

st.write(f"✅ Baseline rule prediction: **{baseline_pred}** (1 = built-up, 0 = non built-up)")

st.divider()

# ============================================================
# Run ONNX model prediction
# ============================================================
st.subheader("🤖 Deep Learning prediction (ONNX)")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Build input vector based on feature order
feature_map = {"ndvi": ndvi, "ndbi": ndbi, "row": row, "col": col}

x_vec = np.array([[feature_map[f] for f in features]], dtype=np.float32)

if st.button("🚀 Predict with ONNX model"):
    out = sess.run(None, {"X": x_vec})
    logits = float(out[0].flatten()[0])
    prob = float(sigmoid(logits))
    pred = int(prob >= 0.5)

    st.success("✅ Prediction done!")

    m1, m2, m3 = st.columns(3)
    m1.metric("Logit", f"{logits:.4f}")
    m2.metric("Built-up probability", f"{prob:.4f}")
    m3.metric("Predicted class", f"{pred} (1 = built-up)")

    st.markdown("""
### ✅ Interpretation
- Probability close to **1.0** → model strongly believes it is **built-up**
- Probability close to **0.0** → model strongly believes it is **not built-up**

✅ Compare:
- Baseline prediction (threshold rule)
- Deep model prediction (learned rule)
""")

st.divider()

# ============================================================
# Explain deployment platforms
# ============================================================
st.subheader("🌍 Where is this model deployed?")

st.markdown("""
### ✅ Streamlit Cloud (this current demo)
This Streamlit app is deployed on Streamlit Cloud.

Reviewers can:
- open the app
- run B1 (or load cache)
- train in B4
- export ONNX in B5
- test predictions here in B6

✅ This is the easiest deployment for a capstone.

---

 ✅ Other free cloud option 

✅ **Render.com**  
or  
✅ **Railway.app**###

Both can deploy a FastAPI API (like `api.py`).

If you want a clean API deployment:
- create an API repo
- upload ONNX file
- deploy with Render/Railway
""")

st.info("✅ Next: you can keep Streamlit Cloud as your main deployment, and mention Render/Railway as optional.")
