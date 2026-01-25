import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="B2 - Urban Growth Results (STAC)", page_icon="📈", layout="wide")
st.title("📈 Track B — Urban Growth Results (Cairo, STAC Sentinel-2)")


# ============================================================
# Check required session state
# ============================================================
required_keys = [
    "B_built_1", "B_built_2", "B_growth",
    "B_ndvi_1", "B_ndbi_1", "B_ndvi_2", "B_ndbi_2"
]

missing = [k for k in required_keys if k not in st.session_state]
if missing:
    st.error("❌ Missing STAC outputs in session_state.")
    st.write("Missing keys:", missing)
    st.info("➡️ Please go to **B1** first and run STAC computation (or click ⚡ Load from cache).")
    st.stop()


# ============================================================
# Load arrays
# ============================================================
built1 = st.session_state["B_built_1"]
built2 = st.session_state["B_built_2"]
growth = st.session_state["B_growth"]

ndvi1 = st.session_state["B_ndvi_1"]
ndbi1 = st.session_state["B_ndbi_1"]
ndvi2 = st.session_state["B_ndvi_2"]
ndbi2 = st.session_state["B_ndbi_2"]


# ============================================================
# Helper plotting
# ============================================================
def show_binary(arr, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(arr)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)


def show_continuous(arr, title):
    fig, ax = plt.subplots(figsize=(6, 4))

    if np.all(np.isnan(arr)):
        ax.text(0.5, 0.5, "All NaN", ha="center", va="center")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        st.pyplot(fig)
        return

    vmin = float(np.nanpercentile(arr, 2))
    vmax = float(np.nanpercentile(arr, 98))
    if abs(vmax - vmin) < 1e-6:
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))

    im = ax.imshow(arr, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)


# ============================================================
# Story / explanation
# ============================================================
st.markdown("""
## ✅ What does this page show?

We compare Cairo in **two time windows** using Sentinel-2 satellite imagery loaded from a STAC API.

### We compute:
- **NDVI** → vegetation signal
- **NDBI** → built-up / urban signal
- **Built-up masks** (binary 0/1)
- **Growth map** (new built-up pixels)

---

### Important note (baseline vs DL)
Built-up masks here are produced by a **simple threshold rule** (baseline).  
Later in Track B, we train a **Deep Learning model** to learn a more stable urban detector.
""")


# ============================================================
# Metrics
# ============================================================
built_ratio_1 = float(np.mean(built1))
built_ratio_2 = float(np.mean(built2))
growth_ratio = float(np.mean(growth))

# pixels that went 1 -> 0
lost_ratio = float(np.mean((built1 == 1) & (built2 == 0)))

st.subheader("📊 Key metrics")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Built-up % (Window 1)", f"{built_ratio_1*100:.2f}%")
m2.metric("Built-up % (Window 2)", f"{built_ratio_2*100:.2f}%")
m3.metric("New Growth % (0 → 1)", f"{growth_ratio*100:.2f}%")
m4.metric("Lost Built-up % (1 → 0)", f"{lost_ratio*100:.2f}%")

if built_ratio_2 < built_ratio_1:
    st.warning(
        "⚠️ Built-up decreased between windows. "
        "This is usually caused by threshold sensitivity or scene differences. "
        "It motivates training a Deep Learning model later."
    )


# ============================================================
# Quick bar chart
# ============================================================
st.subheader("📌 Quick visual summary")

labels = ["Built-up (W1)", "Built-up (W2)", "Growth (0→1)", "Lost (1→0)"]
vals = [built_ratio_1 * 100, built_ratio_2 * 100, growth_ratio * 100, lost_ratio * 100]

fig, ax = plt.subplots(figsize=(8, 3))
ax.bar(labels, vals)
ax.set_ylabel("Percentage (%)")
ax.set_title("Urban change summary")
ax.set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 1)
ax.tick_params(axis="x", rotation=15)
st.pyplot(fig)


# ============================================================
# Masks maps
# ============================================================
st.subheader("🏙️ Built-up masks comparison")

c1, c2 = st.columns(2)
with c1:
    show_binary(built1, "Built-up mask — Window 1")
with c2:
    show_binary(built2, "Built-up mask — Window 2")

st.subheader("🟩 Growth map (new built-up pixels)")
show_binary(growth, "Growth map = built-up in Window 2 AND NOT built-up in Window 1")


# ============================================================
# Indices maps
# ============================================================
st.subheader("🧾 Supporting evidence: NDVI & NDBI indices")

st.markdown("""
These indices explain the logic behind the masks:

- **NDVI**: higher values → vegetation / green areas
- **NDBI**: higher values → built-up / urban surfaces

Typical interpretation:
- **urban** → high NDBI + low NDVI
- **vegetation** → high NDVI
""")

c3, c4 = st.columns(2)
with c3:
    show_continuous(ndvi1, "NDVI — Window 1")
    show_continuous(ndbi1, "NDBI — Window 1")
with c4:
    show_continuous(ndvi2, "NDVI — Window 2")
    show_continuous(ndbi2, "NDBI — Window 2")


# ============================================================
# Interpretation
# ============================================================
st.subheader("🧠 How to interpret the results (beginner-friendly)")

st.markdown(f"""
### ✅ What do the metrics mean?

- **Built-up % (Window 1)**: {built_ratio_1*100:.2f}%  
- **Built-up % (Window 2)**: {built_ratio_2*100:.2f}%  

- **New Growth % (0 → 1)**: {growth_ratio*100:.2f}%  
This approximates **new urban expansion**.

- **Lost Built-up % (1 → 0)**: {lost_ratio*100:.2f}%  
If this is large, it usually means the simple threshold method is **unstable** (not real demolition).

### ✅ Why might built-up decrease?

Because this baseline method depends on:
- the NDVI/NDBI thresholds
- the selected satellite scene
- atmospheric effects (haze/dust)

✅ That’s why we train a **deep learning model in B4** to stabilize results.
""")


st.success("✅ Next step: go to **B3 (Prep STAC)** → then **B4 (Train STAC)**.")
