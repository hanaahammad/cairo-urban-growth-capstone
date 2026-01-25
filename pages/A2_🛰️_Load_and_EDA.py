import os
import streamlit as st
import pandas as pd

from src.data_generation import generate_city_grid, grid_to_dataframe
from src.viz import plot_grid

st.title("🛰️ Load Dataset + EDA")

st.markdown("""
This page generates a **geospatial grid** representing Cairo built-up intensity at two timestamps.

- **t0** = older built-up state
- **t1** = newer built-up state
""")

col1, col2, col3 = st.columns(3)
with col1:
    h = st.slider("Grid height", 50, 200, 100)
with col2:
    w = st.slider("Grid width", 50, 200, 100)
with col3:
    growth_strength = st.slider("Growth strength", 0.05, 0.50, 0.20)

seed = st.number_input("Random seed", 0, 9999, 42)

if st.button("🎲 Generate Cairo grid dataset"):
    t0, t1, built0, built1 = generate_city_grid(h=h, w=w, seed=int(seed), growth_strength=growth_strength)
    df = grid_to_dataframe(t0, t1, built0, built1)

    st.session_state["t0"] = t0
    st.session_state["t1"] = t1
    st.session_state["built0"] = built0
    st.session_state["built1"] = built1
    st.session_state["grid_df"] = df

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/active_grid.csv", index=False)

    st.success("✅ Dataset generated and saved into session + data/active_grid.csv")

if "grid_df" not in st.session_state:
    st.info("Click **Generate Cairo grid dataset** to create data.")
    st.stop()

df = st.session_state["grid_df"]

st.subheader("🔎 Dataset preview")
st.dataframe(df.head())

st.subheader("📊 Simple EDA")
st.write("Built-up ratio at t0:", df["built_t0"].mean())
st.write("Built-up ratio at t1:", df["built_t1"].mean())
st.write("Growth ratio (new built pixels):", df["growth"].mean())

st.subheader("🗺️ Visual maps")
st.pyplot(plot_grid(st.session_state["built0"], "Built-up mask (t0)"))
st.pyplot(plot_grid(st.session_state["built1"], "Built-up mask (t1)"))

with st.expander("🧾 Show code for dataset generation"):
    st.code("""
t0, t1, built0, built1 = generate_city_grid(...)
df = grid_to_dataframe(t0, t1, built0, built1)
""", language="python")
