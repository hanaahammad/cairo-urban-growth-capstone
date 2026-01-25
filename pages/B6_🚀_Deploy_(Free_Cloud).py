import streamlit as st


st.set_page_config(page_title="B6 - Deploy (Free Cloud)", page_icon="🚀", layout="wide")
st.title("🚀 Track B — Deploy the ONNX Model (Free Cloud)")


st.markdown("""
## ✅ Goal of this page
This page explains (step-by-step) how to deploy your trained model **for free**.

We deploy the **ONNX model**, which is:
- portable
- lightweight
- fast with `onnxruntime`

---

## ✅ Recommended free deployment options

### ⭐ Option 1 — Hugging Face Spaces (FastAPI)
✅ Best for deployment / API  
✅ Free tier available  
✅ Easy GitHub integration

### ⭐ Option 2 — Streamlit Cloud
✅ Best for interactive demo  
✅ Very simple for capstone reviewers  
⚠️ Not an API by default (but good enough for a demo)

---

# ✅ What we deploy
After B5 you should have:

- `models/urban_growth_best_model.onnx`
- `models/model_metadata.txt`

These two files are enough for inference.
""")


st.divider()

st.subheader("✅ Minimal requirements for deployment")

st.code(
    """streamlit
numpy
pandas
torch
onnxruntime
""",
    language="text",
)

st.info("💡 You do NOT need full training dependencies in the deployed API, only inference ones.")


st.divider()

st.subheader("🧠 How ONNX inference works (simple explanation)")

st.markdown("""
Your deployed service will do:

1) Load `urban_growth_best_model.onnx`
2) Receive input features (NDVI, NDBI, row, col)
3) Run the model → return logits
4) Convert logits → probability → class (0/1)

✅ Just like PyTorch, but faster and more portable.
""")


st.divider()

st.subheader("🚀 Example: FastAPI ONNX inference server (copy/paste)")

st.markdown("Create a file in your repo named: **`api.py`**")

st.code(
    r"""
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession("models/urban_growth_best_model.onnx", providers=["CPUExecutionProvider"])

app = FastAPI(title="Cairo Urban Growth ONNX API")

# Update the feature order to match your B3 selected features!
FEATURES = ["ndvi", "ndbi", "row", "col"]

class InputRow(BaseModel):
    ndvi: float
    ndbi: float
    row: float
    col: float

@app.post("/predict")
def predict(data: InputRow):
    x = np.array([[data.ndvi, data.ndbi, data.row, data.col]], dtype=np.float32)
    logits = session.run(None, {"X": x})[0].flatten()[0]
    prob = 1 / (1 + np.exp(-logits))
    pred = int(prob >= 0.5)

    return {"logit": float(logits), "prob": float(prob), "pred": pred}
""",
    language="python",
)

st.caption("✅ This creates a clean API endpoint: POST /predict")


st.divider()

st.subheader("🧪 How to test locally")

st.markdown("Run FastAPI locally:")

st.code(
    """bash
pip install fastapi uvicorn onnxruntime numpy
uvicorn api:app --host 0.0.0.0 --port 8000
""",
    language="bash",
)

st.markdown("Test with curl:")

st.code(
    """bash
curl -X POST "http://localhost:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"ndvi\":0.1,\"ndbi\":0.2,\"row\":0.5,\"col\":0.5}"
""",
    language="bash",
)


st.divider()

st.subheader("🌐 Deploy to Hugging Face Spaces (Free)")

st.markdown("""
### ✅ Steps
1) Create a Hugging Face account
2) Create a new **Space**
3) Choose **Docker** or **Gradio** (Docker is easiest for FastAPI)
4) Upload your repository OR connect GitHub

### ✅ Minimum files needed in the Space repo
- `api.py`
- `models/urban_growth_best_model.onnx`
- `requirements.txt`

✅ Once deployed, your API becomes public and your classmates can test it.
""")


st.divider()

st.subheader("✅ What to write in the README (short version)")

st.code(
    """
## Deployment (Free Cloud)

This project exports the best PyTorch model to ONNX (B5).

### Run the demo (Streamlit)
streamlit run app.py

### Export to ONNX
Open B5 and click "Export to ONNX".

### Deploy inference API (FastAPI + ONNX)
Run locally:
uvicorn api:app --host 0.0.0.0 --port 8000

Test:
POST /predict with input features
""",
    language="markdown",
)
