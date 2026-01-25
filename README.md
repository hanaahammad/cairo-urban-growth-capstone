# 🏙️ Cairo Urban Growth — Deep Learning Capstone (Track A + Track B)

This capstone project explores **urban growth and built-up area detection** using:
- ✅ classic remote sensing indices (NDVI / NDBI)
- ✅ deep learning classification (PyTorch)
- ✅ deployment-ready model export (ONNX)
- ✅ an interactive Streamlit application

The project is designed to be **easy to review and reproduce**, with code visible like a notebook and clear explanations in each step.

---

## 🎯 Problem Statement (Simple & Clear)

Cities expand over time, and urban growth impacts:
- infrastructure planning
- transportation needs
- housing and services
- environmental sustainability

In this project we answer:

✅ **Can we detect built-up (urban) areas using satellite-derived features?**  
✅ **Can we compare two time windows and highlight new growth?**  
✅ **Can we train a deep learning model to improve stability vs simple thresholds?**  

---

## 🔥 Two learning paths (A + B)

This Streamlit app offers **two paths** to solve the problem:

### ✅ Track A — Generated / Synthetic Dataset (Fast & Educational)
Track A is best to understand the machine learning pipeline quickly:
- Generates a clean dataset instantly
- Perfect for learning / debugging
- Fast training and evaluation

👉 Best for reviewers who want immediate results.

---

### ✅ Track B — Real Remote Sensing Data via STAC (More realistic)
Track B uses **real Sentinel-2 imagery** for Cairo via STAC and calculates:
- NDVI (vegetation index)
- NDBI (built-up index)
- Built-up mask (baseline threshold)
- Growth map (0 → 1 change)

👉 Best for showing real-world skills: geospatial + remote sensing + ML.

---

## 🧭 Application Navigation (Best Practices)

### ✅ Recommended order (do not skip)
To avoid confusion, always follow the order:

### ✅ Track A (Synthetic Data)
1. **A0 / A1** → Introduction  
2. **A2** → Generate synthetic dataset + EDA  
3. **A3** → Train model + tuning  
4. **A4** → Results + interpretation  
5. **A5** → Export ONNX  
6. **A6** → Deployment guide  

---

### ✅ Track B (Real STAC Data)
1. **B0** → Introduction  
2. **B1** → Load Cairo STAC + compute NDVI/NDBI + cache  
3. **B2** → Results + story (built-up + growth + lost)  
4. **B3** → Prepare tabular dataset for DL  
5. **B4** → Deep training + hyperparameter tuning  
6. **B5** → Export best model to ONNX  
7. **B6** → Deployment guide (free platforms)  

✅ **Important rule:**  
**B2/B3/B4 require B1 first** (or loading the cached results).

---

## ⏳ STAC Waiting Time + Cache (Important for Reviewers)

Track B loads real satellite imagery from a public STAC API.  
Depending on internet speed and STAC server latency, the first run may take:

⏳ **~1 to 5 minutes** (sometimes more)

### ✅ Fast Demo Mode (Recommended)
For smooth review, enable in **B1**:

✅ **Fast demo mode (downsample rasters)**  
✅ **Use cached results if available (faster)**

### ✅ Cache file
After the first successful computation, the app creates:

📌 `data/cairo_stac_cache.npz`

On future runs, simply click:

⚡ **Load from cache now** ✅  
➡️ results load instantly (no STAC delay)

---

## 🧪 Deep Learning Training (What happens in B4?)

Track B training is done with a simple deep neural network (MLP) using **PyTorch**.

### ✅ Why Deep Learning?
The baseline built-up method in B1 is a threshold rule:

- built-up = (NDBI > threshold) AND (NDVI < threshold)

This can be unstable because:
- haze / dust / illumination changes
- desert or bare soil may look like built-up
- thresholds do not generalize well

✅ Deep Learning learns a more stable decision boundary from data.

---

## 🔁 Hyperparameter Tuning (Like the course style)

In **B4**, we reproduce the tuning logic used in ML courses:

We loop over:
- learning rate (lr)
- hidden dimension
- dropout

and select the best model using **F1 score**.

Why F1?
✅ Urban pixels vs non-urban pixels can be imbalanced.

---

## 📦 Export to ONNX (B5)

After training, the best PyTorch model is exported to:

📌 `models/urban_growth_best_model.onnx`

✅ ONNX helps deployment because it is:
- portable
- fast to run with `onnxruntime`
- usable outside PyTorch

A metadata file is also saved:
📌 `models/model_metadata.txt`

---

## 🚀 Deployment (Free Options)

### ⭐ Option 1 — Streamlit Cloud (Recommended for capstone demo)
✅ easiest for sharing with reviewers  
✅ interactive UI  
✅ deploy directly from GitHub

---

### ⭐ Option 2 — Hugging Face Spaces (FastAPI deployment)
✅ best for API deployment  
✅ free tier  
✅ model served as an endpoint `/predict`

---

## ▶️ How to Run Locally

### ✅ 1) Create environment (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
### ✅ 2) Create environment (Linux / macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
```text
cairo-urban-growth-capstone/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── cairo_stac_cache.npz        # (created after B1 first run)
│
├── models/
│   ├── urban_growth_best_model.onnx
│   └── model_metadata.txt
│
├── pages/
│   ├── B0_📘_Intro_(STAC).py
│   ├── B1_🌍_Load_Cairo_STAC.py
│   ├── B2_📈_Urban_Growth_Results_(STAC).py
│   ├── B3_🧼_Prep_(STAC).py
│   ├── B4_🧠_Train_(STAC).py
│   ├── B5_📦_Export_ONNX_(STAC).py
│   └── B6_🚀_Deploy_(Free_Cloud).py
│
└── api.py                          # optional: FastAPI ONNX inference server
```
