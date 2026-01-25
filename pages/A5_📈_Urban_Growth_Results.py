import streamlit as st
import numpy as np

from src.viz import plot_grid, growth_map

st.title("📈 Urban Growth Results + Interpretation")

required = ["built0", "built1", "grid_df"]
missing = [k for k in required if k not in st.session_state]

if missing:
    st.error(f"❌ Missing {missing}. Generate dataset first on Page 2.")
    st.stop()

built0 = st.session_state["built0"]
built1 = st.session_state["built1"]
df = st.session_state["grid_df"]

st.markdown("""
## 🎯 What do we measure here?

Urban growth = pixels that were:

- non-built-up at t0
- built-up at t1

This highlights new urban areas (expansion).
""")

growth = growth_map(built0, built1)

col1, col2 = st.columns(2)
with col1:
    st.pyplot(plot_grid(built0, "Built-up (t0)"))
with col2:
    st.pyplot(plot_grid(built1, "Built-up (t1)"))

st.subheader("🟩 Growth map (new built-up areas)")
st.pyplot(plot_grid(growth, "Urban Growth Map (t1 - t0)"))

st.subheader("📊 Growth statistics")
st.write("Built-up ratio t0:", float(df["built_t0"].mean()))
st.write("Built-up ratio t1:", float(df["built_t1"].mean()))
st.write("Growth ratio:", float(df["growth"].mean()))

st.markdown("""
## 🧠 Interpretation 

- If the growth map shows new built-up pixels far from the center, this suggests **urban sprawl**.
- If growth happens mostly close to existing built-up areas, this suggests **compact expansion**.

This project demonstrates:
            
✅ geospatial change detection  
✅ deep learning workflow  
✅ experiment tracking + tuning  
✅ clear interpretability and visuals  
""")

if "tuning_results" in st.session_state:
    st.subheader("🏆 Best model summary (from training page)")
    st.dataframe(st.session_state["tuning_results"].head(10))
else:
    st.info("No tuning results yet. Train model first in Page 4.")
