import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pystac_client import Client
import rasterio
from rasterio.enums import Resampling


# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="B1 - Load Cairo STAC", page_icon="🌍", layout="wide")
st.title("🌍 Track B — Load Cairo STAC (Sentinel-2)")


# ============================================================
# Intro
# ============================================================
st.markdown("""
This page loads **real Sentinel-2 imagery** for Cairo using a **STAC API** and computes:

✅ **NDVI** (vegetation index)  
✅ **NDBI** (built-up index)  
✅ **Built-up masks** (binary)  
✅ **Urban growth map** (new built-up pixels between 2 dates)

We also include:
- progress + ETA
- caching to disk (no need to re-download every time)
- basic EDA + a CSV-like preview for reviewers
""")


# ============================================================
# Pipeline diagram
# ============================================================
st.subheader("🧭 Pipeline Overview")

st.graphviz_chart(r"""
digraph pipeline {
  rankdir=LR;
  node [shape=box, style="rounded,filled", fillcolor="#f7f7f7"];

  A [label="1) Cairo bbox"];
  B [label="2) Search STAC\nSentinel-2 L2A"];
  C [label="3) Select best scenes\n(low clouds + valid pixels)"];
  D [label="4) Read bands\nRED/NIR/SWIR"];
  E [label="5) Resample SWIR\n(20m → 10m)"];
  F [label="6) Compute indices\nNDVI + NDBI"];
  G [label="7) Built-up mask\n(threshold rule)"];
  H [label="8) Growth map\n(new built-up)"];

  A -> B -> C -> D -> E -> F -> G -> H;
}
""")


# ============================================================
# Cairo bbox
# ============================================================
st.subheader("🗺️ Cairo Bounding Box (AOI)")

cairo_bbox = [31.0, 29.7, 31.7, 30.3]  # [min_lon, min_lat, max_lon, max_lat]
st.code(f"cairo_bbox = {cairo_bbox}", language="python")
st.caption("Format: [min_lon, min_lat, max_lon, max_lat]")


# ============================================================
# Inputs
# ============================================================
st.subheader("📅 Choose 2 time windows")

col1, col2, col3 = st.columns(3)
with col1:
    date1 = st.text_input("Time window 1", "2021-01-01/2021-03-01")
with col2:
    date2 = st.text_input("Time window 2", "2023-01-01/2023-03-01")
with col3:
    max_cloud = st.slider("Max cloud cover (%)", 0, 50, 10)

st.info("✅ Tip: choose the same season with 2+ years gap to reduce false change.")
debug = st.checkbox("🪲 Debug mode (band keys + shapes + stats)", value=False)


# ============================================================
# Cache options
# ============================================================
st.subheader("⚡ Cache options (recommended)")

os.makedirs("data", exist_ok=True)
CACHE_PATH = "data/cairo_stac_cache.npz"

use_cache = st.checkbox("Use cached results if available (faster)", value=True)
force_redownload = st.checkbox("Force re-download (ignore cache)", value=False)

if use_cache and (not force_redownload) and os.path.exists(CACHE_PATH):
    st.success("✅ Cache file found! You can load instantly instead of re-downloading.")

    if st.button("⚡ Load from cache now"):
        data = np.load(CACHE_PATH)

        st.session_state["B_ndvi_1"] = data["ndvi1"]
        st.session_state["B_ndbi_1"] = data["ndbi1"]
        st.session_state["B_ndvi_2"] = data["ndvi2"]
        st.session_state["B_ndbi_2"] = data["ndbi2"]
        st.session_state["B_built_1"] = data["built1"]
        st.session_state["B_built_2"] = data["built2"]
        st.session_state["B_growth"] = data["growth"]

        st.success("✅ Loaded cached indices + masks into session_state.")
        st.info("➡️ You can now go to B2/B3/B4 without re-downloading.")
        st.stop()
else:
    st.info("No cache found yet (or forced download). Run once to create it.")


# ============================================================
# Band resolver (supports multiple STAC asset names)
# ============================================================
BAND_ALIASES = {
    "red":  ["B04", "red"],
    "nir":  ["B08", "nir"],
    "swir": ["B11", "swir16", "swir1", "swir"],
}


def get_asset_key(item, band_type: str) -> str:
    candidates = BAND_ALIASES.get(band_type, [])
    for key in candidates:
        if key in item.assets:
            return key
    raise KeyError(
        f"No asset found for '{band_type}'. Tried: {candidates}\n"
        f"Available assets: {list(item.assets.keys())}"
    )


def read_band(item, band_key: str, out_shape=None):
    """
    Robust remote band reader for STAC assets.

    - Works with remote Cloud-Optimized GeoTIFFs (COGs)
    - Supports downsample via out_shape (recommended)
    - Avoids printing huge arrays
    """

    href = item.assets[band_key].href

    # Short logs (safe for Streamlit)
    st.write(f"📥 Reading band: `{band_key}`")
    st.caption(f"href: {href[:80]}...")

    try:
        # ✅ These GDAL options help remote COG reads
        with rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_USE_HEAD="NO",
            GDAL_HTTP_MULTIRANGE="YES",
            GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        ):
            with rasterio.open(href) as src:
                st.write(f"✅ Opened raster: {src.width} x {src.height}")

                if out_shape is None:
                    st.warning("⚠️ Full resolution read (can be slow). Consider using out_shape.")
                    arr = src.read(1).astype("float32")
                    profile = src.profile
                else:
                    st.write(f"⚡ Downsample read to out_shape={out_shape}")
                    arr = src.read(
                        1,
                        out_shape=out_shape,
                        resampling=Resampling.bilinear
                    ).astype("float32")

                    profile = src.profile.copy()
                    profile.update({"height": out_shape[0], "width": out_shape[1]})

        # ✅ Do NOT st.write(arr) (too big) — show stats instead
        st.write(
            "✅ Band stats:",
            {
                "shape": arr.shape,
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "mean": float(np.nanmean(arr)),
                "zeros_%": float((arr == 0).mean() * 100),
            },
        )

        return arr, profile

    except Exception as e:
        st.error(f"❌ Failed to read band `{band_key}`")
        st.exception(e)
        raise



def normalized_diff(a, b, eps=1e-6):
    return (a - b) / (a + b + eps)


def compute_indices(item, debug=False):
    """
    Robust NDVI/NDBI computation:
    - handles asset key differences
    - resamples SWIR to match NIR shape
    - converts NoData=0 to NaN (prevents blank maps)
    """
    st.write('compue idices', item)
    red_key = get_asset_key(item, "red")
    nir_key = get_asset_key(item, "nir")
    swir_key = get_asset_key(item, "swir")
    out_shape=(512, 512)

    red, _ = read_band(item, red_key, out_shape)
    nir, _ = read_band(item, nir_key, out_shape)

    # resample SWIR to match NIR shape
    swir, profile = read_band(item, swir_key, out_shape=nir.shape)

    # NoData fix
    red = np.where(red == 0, np.nan, red)
    nir = np.where(nir == 0, np.nan, nir)
    swir = np.where(swir == 0, np.nan, swir)

    ndvi = normalized_diff(nir, red)
    ndbi = normalized_diff(swir, nir)

    if debug:
        st.write("✅ Band keys used:", {"red": red_key, "nir": nir_key, "swir": swir_key})
        st.write("RED shape:", red.shape, "| NaN %:", float(np.isnan(red).mean()) * 100)
        st.write("NIR shape:", nir.shape, "| NaN %:", float(np.isnan(nir).mean()) * 100)
        st.write("SWIR shape:", swir.shape, "| NaN %:", float(np.isnan(swir).mean()) * 100)
        st.write("NDVI stats:", float(np.nanmin(ndvi)), float(np.nanmax(ndvi)), float(np.nanmean(ndvi)))
        st.write("NDBI stats:", float(np.nanmin(ndbi)), float(np.nanmax(ndbi)), float(np.nanmean(ndbi)))

    return ndvi, ndbi, profile


def valid_pixel_ratio(item):
    try:
        red_key = get_asset_key(item, "red")
        href = item.assets[red_key].href

        with rasterio.open(href) as src:
            # Read a SMALL version (downsample to ~256x256)
            out_h, out_w = 256, 256
            red_small = src.read(
                1,
                out_shape=(out_h, out_w),
                resampling=Resampling.nearest
            ).astype("float32")

        red_small = np.where(red_small == 0, np.nan, red_small)
        ratio = 1.0 - float(np.isnan(red_small).mean())
        return ratio

    except Exception:
        return 0.0




def get_best_item(client, bbox, date_range, max_cloud=10, max_items=30):
    """
    Smart scene selection:
    - searches STAC
    - filters by cloud cover
    - chooses best valid pixel ratio scene (prevents blank results)

    ✅ Uses search.items() (no deprecated get_items warning)
    """
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        max_items=max_items,
    )
    
    items = list(search.items())  # ✅ FIX
    
    if len(items) == 0:
        return None

    # filter by cloud
    filtered = []
    for it in items:
        cc = it.properties.get("eo:cloud_cover", 100)
        
        if cc <= max_cloud:
            filtered.append(it)

    if len(filtered) == 0:
        filtered = items
    
    # select best
    best_item = None
    best_score = -1e9

    for it in filtered:
        cc = it.properties.get("eo:cloud_cover", 100)
        
        ratio = valid_pixel_ratio(it)  # 0..1
        score = (ratio * 100.0) - cc

        if score > best_score:
            best_score = score
            best_item = it
    
    return best_item


def show_image(arr, title):
    fig, ax = plt.subplots(figsize=(6, 4))

    if np.all(np.isnan(arr)):
        ax.text(0.5, 0.5, "All values are NaN", ha="center", va="center")
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
# Run STAC + compute
# ============================================================
st.subheader("🚀 Run STAC search + compute indices")

run = st.button("📥 Search STAC + Compute NDVI/NDBI")

if run:
    t_total_start = time.perf_counter()

    st.info("Connecting to STAC API...")
    client = Client.open("https://earth-search.aws.element84.com/v1")

    st.write("🔎 Searching best scene for Window 1...")
    item1 = get_best_item(client, cairo_bbox, date1, max_cloud=max_cloud, max_items=30)

    st.write("🔎 Searching best scene for Window 2...")
    item2 = get_best_item(client, cairo_bbox, date2, max_cloud=max_cloud, max_items=30)

    if item1 is None or item2 is None:
        st.error("❌ Could not find scenes for one or both windows. Try wider dates or higher cloud threshold.")
        st.stop()

    st.success("✅ Scenes selected!")
    st.write("📌 Scene 1 date:", item1.datetime)
    st.write("☁️ Scene 1 cloud cover:", item1.properties.get("eo:cloud_cover"))
    st.write("📌 Scene 2 date:", item2.datetime)
    st.write("☁️ Scene 2 cloud cover:", item2.properties.get("eo:cloud_cover"))

    # ------------------------------------------------------------
    # Progress + ETA
    # ------------------------------------------------------------
    st.subheader("⏳ Computing NDVI + NDBI (progress + ETA)")

    progress = st.progress(0)
    status = st.empty()
    eta_box = st.empty()

    step_times = []
    total_steps = 2

    def update_eta(done_steps):
        if len(step_times) == 0:
            eta_box.info("ETA: calculating...")
            return
        avg_step = sum(step_times) / len(step_times)
        remaining = total_steps - done_steps
        eta_sec = remaining * avg_step
        eta_box.info(f"⏱ ETA ≈ {eta_sec:.1f} sec remaining")

    with st.spinner("Downloading bands + computing indices (may take time)..."):
        status.write("🛰️ Scene 1/2 → computing NDVI/NDBI...")
        progress.progress(10)
        t1 = time.perf_counter()
        #st.write('T1', t1)
        ndvi1, ndbi1, _ = compute_indices(item1, debug=debug)
        #st.write(ndvi1, ndbi1)
        t2 = time.perf_counter()
        #st.write('T2', t2)
        step_times.append(t2 - t1)
        progress.progress(50)
        update_eta(done_steps=1)

        status.write("🛰️ Scene 2/2 → computing NDVI/NDBI...")
        t3 = time.perf_counter()
        ndvi2, ndbi2, _ = compute_indices(item2, debug=debug)
        t4 = time.perf_counter()
        step_times.append(t4 - t3)
        progress.progress(100)
        update_eta(done_steps=2)

    status.success("✅ NDVI/NDBI computed successfully!")

    # Window 2 validity check
    valid_ratio2 = 1.0 - float(np.isnan(ndvi2).mean())
    st.write(f"✅ Window 2 valid pixel ratio: {valid_ratio2*100:.2f}%")
    if valid_ratio2 < 0.05:

        st.error("❌ Window 2 has almost no valid pixels. Try another time window.")
        st.stop()

    # ------------------------------------------------------------
    # Built-up thresholds
    # ------------------------------------------------------------
    st.subheader("🏙️ Built-up mask + Growth thresholds")

    colT1, colT2 = st.columns(2)
    with colT1:
        ndbi_thr = st.slider("NDBI threshold", -0.2, 0.5, 0.10, 0.01)
    with colT2:
        ndvi_thr = st.slider("NDVI threshold", -0.2, 0.9, 0.30, 0.01)

    built1 = ((ndbi1 > ndbi_thr) & (ndvi1 < ndvi_thr)).astype(np.uint8)
    built2 = ((ndbi2 > ndbi_thr) & (ndvi2 < ndvi_thr)).astype(np.uint8)
    growth = ((built1 == 0) & (built2 == 1)).astype(np.uint8)

    # save to session_state
    st.session_state["B_ndvi_1"] = ndvi1
    st.session_state["B_ndbi_1"] = ndbi1
    st.session_state["B_ndvi_2"] = ndvi2
    st.session_state["B_ndbi_2"] = ndbi2
    st.session_state["B_built_1"] = built1
    st.session_state["B_built_2"] = built2
    st.session_state["B_growth"] = growth

    # cache save
    np.savez_compressed(
        CACHE_PATH,
        ndvi1=ndvi1, ndbi1=ndbi1,
        ndvi2=ndvi2, ndbi2=ndbi2,
        built1=built1, built2=built2,
        growth=growth
    )
    st.success(f"💾 Cached results saved to: {CACHE_PATH}")

    # total time
    t_total_end = time.perf_counter()
    st.caption(f"✅ Total time taken: **{(t_total_end - t_total_start):.2f} seconds**")

    # ------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------
    st.subheader("🖼️ Indices preview (2 × 2)")
    c1, c2 = st.columns(2)
    with c1:
        show_image(ndvi1, "NDVI — Window 1")
        show_image(ndbi1, "NDBI — Window 1")
    with c2:
        show_image(ndvi2, "NDVI — Window 2")
        show_image(ndbi2, "NDBI — Window 2")

    st.subheader("🟩 Built-up & Growth maps")
    c3, c4 = st.columns(2)
    with c3:
        show_image(built1, "Built-up — Window 1")
        show_image(growth, "Urban Growth (new built-up)")
    with c4:
        show_image(built2, "Built-up — Window 2")
        st.metric("Growth ratio (%)", f"{growth.mean() * 100:.2f}%")

    # ------------------------------------------------------------
    # EDA + mini CSV preview
    # ------------------------------------------------------------
    st.subheader("🔍 EDA — Summary")
    summary = pd.DataFrame({
        "Variable": ["NDVI (t1)", "NDBI (t1)", "NDVI (t2)", "NDBI (t2)", "Built-up (t1)", "Built-up (t2)", "Growth"],
        "Shape": [str(ndvi1.shape), str(ndbi1.shape), str(ndvi2.shape), str(ndbi2.shape),
                  str(built1.shape), str(built2.shape), str(growth.shape)],
        "Min": [float(np.nanmin(ndvi1)), float(np.nanmin(ndbi1)), float(np.nanmin(ndvi2)), float(np.nanmin(ndbi2)),
                float(np.nanmin(built1)), float(np.nanmin(built2)), float(np.nanmin(growth))],
        "Max": [float(np.nanmax(ndvi1)), float(np.nanmax(ndbi1)), float(np.nanmax(ndvi2)), float(np.nanmax(ndbi2)),
                float(np.nanmax(built1)), float(np.nanmax(built2)), float(np.nanmax(growth))],
        "Mean": [float(np.nanmean(ndvi1)), float(np.nanmean(ndbi1)), float(np.nanmean(ndvi2)), float(np.nanmean(ndbi2)),
                 float(np.nanmean(built1)), float(np.nanmean(built2)), float(np.nanmean(growth))],
    })
    st.dataframe(summary, use_container_width=True)

    st.subheader("🧾 Mini dataset preview (CSV-like sample)")
    n_preview = st.slider("Preview sample size", 10, 200, 30)
    h, w = ndvi1.shape
    rng = np.random.default_rng(42)
    rr = rng.integers(0, h, size=n_preview)
    cc = rng.integers(0, w, size=n_preview)

    mini_df = pd.DataFrame({
        "row": rr,
        "col": cc,
        "ndvi_t1": ndvi1[rr, cc],
        "ndbi_t1": ndbi1[rr, cc],
        "ndvi_t2": ndvi2[rr, cc],
        "ndbi_t2": ndbi2[rr, cc],
        "built_t1": built1[rr, cc],
        "built_t2": built2[rr, cc],
        "growth": growth[rr, cc],
    })
    st.dataframe(mini_df, use_container_width=True)

    st.success("✅ Done! Next: go to **B2 Results** then **B3 Prep** then **B4 Train**.")
