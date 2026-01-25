import numpy as np
import pandas as pd

def generate_city_grid(
    h=100, w=100,
    seed=42,
    growth_strength=0.20
):
    """
    Generates a synthetic geospatial grid representing built-up probability (0..1)
    for 2 timestamps:
      - t0: older city state
      - t1: newer city state (expanded)
    """
    rng = np.random.default_rng(seed)

    # Base "city center" gaussian
    y = np.linspace(-2, 2, h)
    x = np.linspace(-2, 2, w)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    center = np.exp(-(xx**2 + yy**2))

    noise = rng.normal(0, 0.08, size=(h, w))
    t0 = np.clip(center + noise, 0, 1)

    # Growth = add more built-up around edges (simulating expansion)
    ring = np.exp(-((np.sqrt(xx**2 + yy**2) - 1.3) ** 2) / 0.2)
    t1 = np.clip(t0 + growth_strength * ring + rng.normal(0, 0.05, size=(h, w)), 0, 1)

    # binary masks (built-up / non-built-up)
    built0 = (t0 > 0.45).astype(int)
    built1 = (t1 > 0.45).astype(int)

    return t0, t1, built0, built1


def grid_to_dataframe(t0, t1, built0, built1):
    """
    Converts the 2 time grids into a flat dataframe for EDA.
    """
    h, w = t0.shape
    rows = []
    for i in range(h):
        for j in range(w):
            rows.append({
                "row": i,
                "col": j,
                "built_t0": int(built0[i, j]),
                "built_t1": int(built1[i, j]),
                "intensity_t0": float(t0[i, j]),
                "intensity_t1": float(t1[i, j]),
            })
    df = pd.DataFrame(rows)

    # growth label: newly built in t1 but not in t0
    df["growth"] = ((df["built_t1"] == 1) & (df["built_t0"] == 0)).astype(int)
    return df
