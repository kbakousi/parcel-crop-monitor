"""Synthetic parcels and Sentinel-2-like index time series for tests and demos.

Each crop group gets a simple, published-shape phenology (double-logistic
green-up and senescence, or repeated cuts for grassland), with noise and
random cloud cover. Values are illustrative, not calibrated to any region.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["PHENOLOGY", "make_scene", "to_dataset"]


def _double_logistic(doy, base, amp, up, down, k_up=0.12, k_down=0.12):
    return base + amp * (1 / (1 + np.exp(-k_up * (doy - up))) - 1 / (1 + np.exp(-k_down * (doy - down))))


def _grass(doy, cuts=(140, 190, 240)):
    v = np.full(doy.shape, 0.75)
    for c in cuts:
        days = doy - c
        v = np.where((days >= 0) & (days < 25), 0.35 + 0.4 * days / 25, v)
    return v


#: NDVI-like curves per synthetic group (day of year -> value).
PHENOLOGY = {
    "winter_cereal": lambda d: _double_logistic(d, 0.25, 0.6, 90, 185),
    "maize": lambda d: _double_logistic(d, 0.18, 0.7, 170, 270),
    "grassland": _grass,
    "fallow_bare": lambda d: np.full(d.shape, 0.12),
}


def make_scene(n_side: int = 12, parcel_px: int = 8, year: int = 2025, step_days: int = 5,
               cloud_prob: float = 0.3, mislabel_rate: float = 0.05, seed: int = 0, pixel_size_m: float = 10.0):
    """Create a square scene of ``n_side x n_side`` rectangular parcels.

    Returns
    -------
    dict with ``ndvi`` (T, H, W), ``clear`` (T, H, W), ``dates``, ``labels``
    (H, W), ``parcels`` (DataFrame: parcel_id, true_group, declared_group,
    x, y), and ``pixel_size_m``. A share ``mislabel_rate`` of parcels gets a
    wrong declared group, to exercise the verdict logic.
    """
    rng = np.random.default_rng(seed)
    groups = list(PHENOLOGY)
    n = n_side * n_side
    true = rng.choice(groups, size=n)
    declared = true.copy()
    flip = rng.random(n) < mislabel_rate
    for i in np.flatnonzero(flip):
        declared[i] = rng.choice([g for g in groups if g != true[i]])

    gap = 1  # one-pixel border between parcels (like field margins)
    side = n_side * (parcel_px + gap) + gap
    labels = np.zeros((side, side), dtype="int32")
    xs, ys = [], []
    for k in range(n):
        r, c = divmod(k, n_side)
        r0, c0 = gap + r * (parcel_px + gap), gap + c * (parcel_px + gap)
        labels[r0 : r0 + parcel_px, c0 : c0 + parcel_px] = k + 1
        xs.append((c0 + parcel_px / 2) * pixel_size_m)
        ys.append((side - r0 - parcel_px / 2) * pixel_size_m)

    dates = pd.date_range(f"{year}-03-01", f"{year}-10-31", freq=f"{step_days}D")
    doy = dates.day_of_year.to_numpy()
    T = len(dates)
    ndvi = np.full((T, side, side), 0.1)
    for k in range(n):
        curve = PHENOLOGY[true[k]](doy.astype(float)) + rng.normal(0, 0.02, T)
        mask = labels == k + 1
        ndvi[:, mask] = curve[:, None] + rng.normal(0, 0.03, (T, mask.sum()))
    clear = np.ones_like(ndvi, dtype=bool)
    cloudy_dates = rng.random(T) < cloud_prob
    for t in np.flatnonzero(cloudy_dates):
        r0, c0 = rng.integers(0, side, 2)
        clear[t, max(0, r0 - side // 3) : r0 + side // 3, max(0, c0 - side // 3) : c0 + side // 3] = False
        ndvi[t][~clear[t]] = rng.uniform(0.0, 0.1)  # clouds look like low NDVI
    parcels = pd.DataFrame(
        {"parcel_id": np.arange(1, n + 1), "true_group": true, "declared_group": declared, "x": xs, "y": ys}
    )
    return {"ndvi": ndvi, "clear": clear, "dates": dates, "labels": labels, "parcels": parcels,
            "pixel_size_m": pixel_size_m}


def to_dataset(scene: dict):
    """Return a :func:`make_scene` result as an ``xarray.Dataset``.

    Variables ``ndvi`` and ``clear`` (time, y, x) and ``labels`` (y, x), on
    pixel-centre coordinates in metres (north-up, origin at the lower-left
    corner) that match the parcels' ``x``/``y`` centroids. The scene has no
    CRS: it is not a real place.
    """
    import xarray

    side = scene["labels"].shape[0]
    px = scene["pixel_size_m"]
    coords = {
        "time": pd.DatetimeIndex(scene["dates"]),
        "y": (side - np.arange(side) - 0.5) * px,
        "x": (np.arange(side) + 0.5) * px,
    }
    return xarray.Dataset(
        {
            "ndvi": (("time", "y", "x"), scene["ndvi"]),
            "clear": (("time", "y", "x"), scene["clear"]),
            "labels": (("y", "x"), scene["labels"]),
        },
        coords=coords,
        attrs={"description": "synthetic parcels and NDVI-like series (parcelmon.synthetic)"},
    )
