import streamlit as st
import numpy as np
import pandas as pd


# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="B3 - Prep (STAC)", page_icon="🧼", layout="wide")
st.title("🧼 Track B — Data Preparation (STAC → Tabular Dataset)")


# ============================================================
# Check required session state
# ============================================================
required_keys = ["B_ndvi_1", "B_ndbi_1", "B_built_1"]
missing = [k for k in required_keys if k not in st.session_state]

if missing:
    st.error("❌ No STAC computed data found.")
    st.write("Missing keys:", missing)
    st.info("➡️ Please run **B1** first (Load Cairo STAC).")
    st.stop()

ndvi = st.session_state["B_ndvi_1"]
ndbi = st.session_state["B_ndbi_1"]
built = st.session_state["B_built_1"]


# ============================================================
# Explanation
# ============================================================
st.markdown("""
## ✅ Goal of this page
Satellite data comes as **raster maps** (2D arrays).  
Deep Learning models usually require a **tabular dataset**:

### Features (X)
- NDVI
- NDBI
- optional pixel row/col (helps models learn spatial patterns)

### Target (y)
- built-up mask (0/1)

✅ We will sample pixels to create a lightweight dataset for training.
""")


# ============================================================
# User parameters
# ============================================================
st.subheader("⚙️ Dataset creation options")

c1, c2, c3 = st.columns(3)
with c1:
    sample_size = st.slider("Sample size (pixels)", 2000, 150000, 30000, 2000)
with c2:
    add_rowcol = st.checkbox("Add pixel row/col as features", value=True)
with c3:
    random_seed = st.number_input("Random seed", 0, 9999, 42)

st.caption("✅ Larger sample = better training but slower tuning.")

# Optional: balance the dataset (built-up vs non built-up)
st.subheader("⚖️ Optional balancing (recommended for classification)")
balance = st.checkbox("Balance classes (built-up vs non built-up)", value=True)


# ============================================================
# Create dataset
# ============================================================
st.subheader("🧱 Building the tabular dataset")

h, w = ndvi.shape
total_pixels = h * w

st.write(f"Raster shape: **{h} x {w}**  → total pixels: **{total_pixels:,}**")

# flatten
ndvi_flat = ndvi.flatten()
ndbi_flat = ndbi.flatten()
y_flat = built.flatten().astype(int)

# create row/col indices
rows, cols = np.indices((h, w))
rows_flat = rows.flatten()
cols_flat = cols.flatten()

# base dataframe
df = pd.DataFrame({
    "ndvi": ndvi_flat,
    "ndbi": ndbi_flat,
    "label": y_flat,
})

if add_rowcol:
    df["row"] = rows_flat
    df["col"] = cols_flat

# clean invalid values
df = df.replace([np.inf, -np.inf], np.nan).dropna()

st.success(f"✅ Clean dataset size (after dropna): {len(df):,} rows")


# ============================================================
# Sampling
# ============================================================
st.subheader("🎯 Sampling")

rng = np.random.default_rng(int(random_seed))

if len(df) <= sample_size:
    df_sample = df.copy()
else:
    if balance:
        # Split by class
        df0 = df[df["label"] == 0]
        df1 = df[df["label"] == 1]

        # target per class
        per_class = sample_size // 2

        # if one class is small, fallback to random sampling
        if len(df0) < per_class or len(df1) < per_class:
            st.warning("⚠️ Not enough samples in one class. Falling back to random sampling.")
            df_sample = df.sample(sample_size, random_state=int(random_seed))
        else:
            df0_s = df0.sample(per_class, random_state=int(random_seed))
            df1_s = df1.sample(per_class, random_state=int(random_seed))
            df_sample = pd.concat([df0_s, df1_s]).sample(frac=1, random_state=int(random_seed)).reset_index(drop=True)
    else:
        df_sample = df.sample(sample_size, random_state=int(random_seed))

st.success(f"✅ Final training dataset size: {len(df_sample):,} rows")


# ============================================================
# Feature selection
# ============================================================
st.subheader("✅ Select features for Deep Learning (X)")

all_possible_features = ["ndvi", "ndbi"]
if add_rowcol:
    all_possible_features += ["row", "col"]

# default features
default_features = ["ndvi", "ndbi"]
if add_rowcol:
    default_features += ["row", "col"]

selected_features = st.multiselect(
    "Feature columns (X)",
    options=all_possible_features,
    default=default_features
)

target_col = "label"

if len(selected_features) == 0:
    st.error("❌ Please select at least one feature column.")
    st.stop()

st.info(f"✅ Target column (y): **{target_col}**")


# ============================================================
# Preview + distribution
# ============================================================
st.subheader("🔍 Dataset preview (EDA)")

st.dataframe(df_sample.head(30), use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.write("✅ Label distribution")
    st.write(df_sample["label"].value_counts(normalize=True).rename("ratio"))

with c2:
    st.write("✅ Feature summary stats")
    st.dataframe(df_sample[selected_features].describe().T, use_container_width=True)


# ============================================================
# Save to session_state
# ============================================================
st.subheader("💾 Save prepared dataset to session_state")

if st.button("💾 Save dataset for training (B4)"):
    st.session_state["B_stac_df"] = df_sample
    st.session_state["B_features"] = selected_features
    st.session_state["B_target"] = target_col

    st.success("✅ Saved!")
    st.write("Saved keys:")
    st.write({
        "B_stac_df": "DataFrame",
        "B_features": selected_features,
        "B_target": target_col
    })

    st.info("➡️ Next: go to **B4 — Train (STAC)**")


# ============================================================
# Show code (notebook-style)
# ============================================================
with st.expander("🧾 Show the code (notebook-style)", expanded=False):
    st.code(
        f"""
# --- Load raster arrays from session_state ---
ndvi = st.session_state["B_ndvi_1"]
ndbi = st.session_state["B_ndbi_1"]
built = st.session_state["B_built_1"]

# --- Flatten rasters (pixel-level dataset) ---
df = pd.DataFrame({{
    "ndvi": ndvi.flatten(),
    "ndbi": ndbi.flatten(),
    "label": built.flatten().astype(int),
}})

# optional spatial features (row/col)
h, w = ndvi.shape
rows, cols = np.indices((h, w))
df["row"] = rows.flatten()
df["col"] = cols.flatten()

# --- Clean invalid values ---
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# --- Sample a subset of pixels for fast training ---
df_sample = df.sample({len(df_sample)}, random_state={int(random_seed)})

# --- Select features ---
features = {selected_features}
target = "label"

# Save for B4 training
st.session_state["B_stac_df"] = df_sample
st.session_state["B_features"] = features
st.session_state["B_target"] = target
        """,
        language="python"
    )
