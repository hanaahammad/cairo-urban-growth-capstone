import streamlit as st
from src.patching import extract_patch_features

st.title("🧼 Data Preparation (Patch dataset)")

st.markdown("""
## 🎯 Goal

Convert the geospatial grid into a **training dataset** for Deep Learning.

We create features from local neighborhoods (patches):
- mean intensity
- std intensity
- min/max intensity
- neighbor built-up ratio

Label:
- 1 = built-up
- 0 = non-built-up
""")

required = ["t0", "built0"]
missing = [k for k in required if k not in st.session_state]

if missing:
    st.error(f"❌ Missing {missing}. Please generate data first in Page 2.")
    st.stop()

patch = st.slider("Patch size", 3, 11, 5, step=2)

if st.button("🧩 Build patch dataset"):
    patch_df = extract_patch_features(
        st.session_state["t0"],
        st.session_state["built0"],
        patch=patch
    )
    st.session_state["patch_df"] = patch_df
    st.success("✅ Patch dataset created in session_state['patch_df']")

if "patch_df" not in st.session_state:
    st.info("Click **Build patch dataset** to continue.")
    st.stop()

st.dataframe(st.session_state["patch_df"].head())

st.write("Label distribution:")
st.write(st.session_state["patch_df"]["label"].value_counts(normalize=True))

with st.expander("🧾 Show code (patch extraction)"):
    st.code("""
patch_df = extract_patch_features(t0, built0, patch=5)
""", language="python")
