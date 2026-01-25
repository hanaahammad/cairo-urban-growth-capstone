import streamlit as st

st.set_page_config(
    page_title="Urban Growth Cairo - Capstone 2",
    page_icon="🏙️",
    layout="wide"
)

st.title("🏙️ Urban Growth Cairo — Capstone 2")
st.markdown("""
Welcome! Use the left menu to navigate through the pipeline:

1) Introduction  
2) Load dataset + EDA  
3) Data preparation (patch dataset)  
4) Train Deep Learning model + Hyperparameter tuning  
5) Urban Growth results + Interpretation  
""")
st.info("⬅️ Use the sidebar to open the pages.")
