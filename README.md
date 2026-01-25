# cairo-urban-growth-capstone
capstone2 for ML zoomcamp course

# 🏙️ Urban Growth Detection — Cairo (Capstone 2)

## 🚀 Live Application
(Deploy link here after Streamlit Cloud)

---

## 👋 Why this project?
This project started from my curiosity about how cities change over time.
Urban growth is a real-world phenomenon that affects infrastructure, mobility,
environment, and public services.

The goal of this capstone is to build a **complete end-to-end ML system**
using geospatial-style data and deep learning.

---

## 🎯 Problem Statement
We want to estimate **urban growth in Cairo** between two timestamps:

- **t0** → older city state
- **t1** → newer city state

Urban growth is detected as:
> areas that were non-built-up at t0 but become built-up at t1

---

## 🧾 Dataset
To keep the capstone reproducible and easy to evaluate, the app generates
a synthetic geospatial grid that mimics satellite built-up intensity.

The dataset includes:
- pixel position: row, col
- built-up mask at t0 and t1
- intensity values
- growth label

This setup allows us to demonstrate:
✅ loading data  
✅ EDA  
✅ preprocessing  
✅ deep learning training  
✅ hyperparameter tuning  
✅ evaluation  
✅ visualization and interpretation  

---

## 🧠 Deep Learning Approach
We train a small neural network classifier (PyTorch) on patch-level features:

Features extracted from local neighborhood (patch):
- mean / std intensity
- min / max intensity
- built-up neighbor ratio

Label:
- 1 = built-up
- 0 = non-built-up

---

## 🧪 Hyperparameter tuning (like the course)
We perform a grid search over:

- learning rate (lr)
- hidden dimension (hidden_dim)
- dropout

The Streamlit app shows:
- progress bar
- experiment history table
- best configuration selected by validation loss

---

## 📊 Outputs & Visualizations
The app provides:
- built-up map at t0
- built-up map at t1
- growth map (new built-up areas)
- growth statistics (%)

---

## ▶️ Run locally
```bash
pip install -r requirements.txt
streamlit run app.py

## 🧪 Environment Setup (Windows + Linux/Mac)

You can run this project using either:

✅ **Option A: venv (recommended, standard Python)**  
✅ **Option B: uv (faster installs)**

---

### ✅ Option A — Using `venv`

#### 🪟 Windows (PowerShell)

```powershell
cd C:\Workspaces\cairo-urban-growth-capstone
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py

#### Linux / 🍎 MacOS (bash)
cd cairo-urban-growth-capstone
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py


cairo-urban-growth-capstone/
├── app.py
├── requirements.txt
├── README.md
├── pages/
├── src/
└── data/

