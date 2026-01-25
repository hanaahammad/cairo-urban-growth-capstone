import streamlit as st

st.title("🏠 Introduction — Urban Growth (Cairo)")

st.markdown("""
## 🎯 Problem

Cities expand over time (new buildings, roads, neighborhoods).  
This project estimates **urban growth in Cairo** using a deep learning workflow.

We want to answer:

✅ Where did the city grow?  
✅ How much built-up area increased?  

## 🧠 Why Deep Learning?

Deep learning helps learn patterns from data and generalize.  
Here we use a small neural network to classify city patches into:

- Built-up (urban) ✅
- Non-built-up (non-urban) ✅

## 🧭 Pipeline Steps (Capstone)

1) Load dataset  
2) EDA  
3) Data preparation (patch features)  
4) Train DL model + hyperparameter tuning  
5) Results interpretation  
6) Deploy with Streamlit Cloud
""")
