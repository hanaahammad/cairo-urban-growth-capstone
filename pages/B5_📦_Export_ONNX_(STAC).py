import os
import numpy as np
import streamlit as st
import torch


st.set_page_config(page_title="B5 - Export ONNX", page_icon="📦", layout="wide")
st.title("📦 Track B — Export Best Model to ONNX")


st.markdown("""
## ✅ Why export to ONNX?

ONNX is a portable model format that allows you to run inference:
- outside Python
- on different platforms
- using **onnxruntime** (fast)

In this project:
- we train a PyTorch model (B4)
- then export it to ONNX here
""")


# ============================================================
# Checks
# ============================================================
if "B_best_model" not in st.session_state:
    st.error("❌ No trained model found in session_state.")
    st.info("➡️ Please go to **B4** and run tuning first.")
    st.stop()

if "B_features" not in st.session_state:
    st.error("❌ Missing feature list in session_state.")
    st.info("➡️ Please go to **B3** and save the dataset, then train in **B4**.")
    st.stop()

model = st.session_state["B_best_model"]
features = st.session_state["B_features"]

st.success("✅ Best model found!")
st.write("Feature columns:", features)


# ============================================================
# Export settings
# ============================================================
st.subheader("⚙️ Export settings")

os.makedirs("models", exist_ok=True)

default_name = "urban_growth_best_model.onnx"
onnx_name = st.text_input("ONNX filename", default_name)
onnx_path = os.path.join("models", onnx_name)

opset_version = st.slider("ONNX opset version", 11, 18, 13)
do_constant_folding = st.checkbox("Constant folding (recommended)", value=True)

st.caption("✅ Tip: Opset 13 is usually safe and widely supported.")


# ============================================================
# Export button
# ============================================================
st.subheader("📤 Export")

if st.button("📦 Export to ONNX"):
    try:
        model.eval()

        # Dummy input: [batch, features]
        dummy = torch.randn(1, len(features), dtype=torch.float32)

        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            input_names=["X"],
            output_names=["logits"],
            dynamic_axes={
                "X": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )

        st.success(f"✅ Exported ONNX model to: `{onnx_path}`")

        # Save metadata helpful for deployment
        meta_path = os.path.join("models", "model_metadata.txt")
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write("Model metadata\n")
            f.write("====================\n")
            f.write(f"Features: {features}\n")
            f.write(f"ONNX file: {onnx_path}\n")
            if "B_best_cfg" in st.session_state:
                f.write(f"Best config: {st.session_state['B_best_cfg']}\n")

        st.info("✅ Also saved: `models/model_metadata.txt`")

    except Exception as e:
        st.error(f"❌ Export failed: {e}")


# ============================================================
# Quick ONNX Runtime test
# ============================================================
st.subheader("🧪 Quick inference test (ONNX Runtime)")

st.markdown("""
This test requires:

- `onnxruntime`

If it works, your ONNX model is valid ✅
""")

try:
    import onnxruntime as ort  # noqa
    onnxruntime_available = True
except Exception:
    onnxruntime_available = False

if not onnxruntime_available:
    st.warning("onnxruntime is not installed. Add it to requirements.txt: `onnxruntime`")
else:
    if os.path.exists(onnx_path):
        if st.button("▶️ Run ONNX test inference"):
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            x = np.random.randn(5, len(features)).astype(np.float32)
            out = sess.run(None, {"X": x})
            logits = out[0].flatten()

            st.write("✅ Output logits sample:", logits[:5])
            st.write("Shape:", out[0].shape)
    else:
        st.info("ℹ️ Export the model first to enable the ONNX test.")
