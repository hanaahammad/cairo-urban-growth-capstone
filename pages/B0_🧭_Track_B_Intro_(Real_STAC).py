import streamlit as st

st.set_page_config(page_title="Track B Intro (STAC)", page_icon="🧭", layout="wide")

st.title("🧭 Track B — Real Urban Growth in Cairo (STAC + Sentinel-2)")

st.markdown("""
Welcome to **Track B** of this capstone project.

This track uses **real satellite imagery** (Sentinel-2) to study **urban expansion** in Cairo.

---

## 🎯 Problem Statement (Urban Growth)
Cities grow over time, and Cairo is a great example of fast urban expansion.

In this track, we want to answer:

✅ **Where did new built-up areas appear?**  
✅ **How much did the built-up footprint grow between two dates?**  
✅ **Can we create an explainable pipeline that anyone can reproduce?**

---

## 🛰️ Why Satellite Imagery?
Satellite images allow us to observe cities consistently through time.

We use:
- **Sentinel-2 Level 2A** (surface reflectance)
- A **STAC API** (SpatioTemporal Asset Catalog) to search and download data by:
  - location (Cairo bounding box)
  - time range (two time windows)
  - cloud cover filtering

---

## 📌 What is STAC?
**STAC** is a standard catalog format for satellite imagery.

Instead of downloading huge datasets manually, STAC lets us:
✅ search “what exists”  
✅ filter by quality (cloud cover)  
✅ select only what we need  

---

## 🌿 Indices we compute
We compute two classic remote sensing indices:

### NDVI — Vegetation Index
It helps detect vegetation areas:

\\[
NDVI = \\frac{NIR - RED}{NIR + RED}
\\]

### NDBI — Built-up Index
It helps detect built-up / urban surfaces:

\\[
NDBI = \\frac{SWIR - NIR}{SWIR + NIR}
\\]

---

## 🧠 Built-up Mask & Urban Growth Map
To get an interpretable “urban expansion” output:

✅ **Built-up mask** (binary 0/1 map) for each date window  
✅ **Growth map** = pixels that changed from:
- not built-up (old) → built-up (new)

---

## ✅ Track B Pages (Recommended Order)
Please follow the pages in this order:

1️⃣ **B1 — Load Cairo STAC**
- searches satellite data
- computes NDVI + NDBI
- creates built-up mask + growth map
- (optional) saves/cache results

2️⃣ **B2 — Urban Growth Results**
- visualizes built-up and growth maps
- explains interpretation + statistics

3️⃣ **B3 — Prep (STAC)**
- converts rasters into a tabular dataset (X, y)

4️⃣ **B4 — Train (STAC)**
- trains a simple deep model
- performs hyperparameter tuning (loop)
- evaluates performance (F1, accuracy)

5️⃣ **B5 — Export ONNX**
- exports the trained model to ONNX format

---

## ✅ Notes for reviewers
This is designed to be:
✅ reproducible  
✅ explainable (code shown on every step)  
✅ beginner-friendly  
✅ aligned with a full ML pipeline (load → EDA → prep → train → evaluate → export)

---

➡️ Go to **B1** to start downloading and computing indices for Cairo.
""")
