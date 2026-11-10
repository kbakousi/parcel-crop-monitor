"""Cleaning, regularising and smoothing per-parcel index time series.

Observations with a low share of clear pixels are dropped; the remaining
irregular series are linearly interpolated onto a regular time grid, without
bridging gaps longer than ``max_gap_days`` and without extrapolating beyond
the first and last observation. Optional Savitzky-Golay smoothing is applied
to runs of consecutive valid grid values.

References
----------
Inglada, J. et al. (2015). Assessment of an operational system for crop
    type map production using high temporal and spatial resolution satellite
    optical imagery. Remote Sensing 7, 12356-12379.
    https://doi.org/10.3390/rs70912356 (regular gap-filled time series)
Savitzky, A., Golay, M.J.E. (1964). Smoothing and differentiation of data by
    simplified least squares procedures. Analytical Chemistry 36,
    1627-1639. https://doi.org/10.1021/ac60214a047
Chen, J. et al. (2004). A simple method for reconstructing a high-quality
    NDVI time-series data set based on the Savitzky-Golay filter. Remote
    Sensing of Environment 91, 332-344.
    https://doi.org/10.1016/j.rse.2004.03.014 (context: the upper-envelope
    refinement of that paper is not applied here)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

__all__ = ["filter_observations", "regularize", "smooth", "feature_matrix", "n_valid_dates"]


def filter_observations(obs: pd.DataFrame, min_valid_fraction: float, min_valid_pixels: int = 1) -> pd.DataFrame:
    """Keep observations with enough clear pixels and a finite statistic."""
    keep = (obs["valid_fraction"] >= min_valid_fraction) & (obs["n_valid"] >= min_valid_pixels) & obs["mean"].notna()
    return obs.loc[keep].copy()


def _interp_with_gaps(t_obs, v_obs, t_grid, max_gap):
    out = np.full(t_grid.shape, np.nan)
    if len(t_obs) == 0:
        return out
    if len(t_obs) == 1:
        out[t_grid == t_obs[0]] = v_obs[0]
        return out
    inside = (t_grid >= t_obs[0]) & (t_grid <= t_obs[-1])
    out[inside] = np.interp(t_grid[inside], t_obs, v_obs)
    right = np.searchsorted(t_obs, t_grid, side="left")
    exact = (right < len(t_obs)) & (t_obs[np.minimum(right, len(t_obs) - 1)] == t_grid)
    left = np.clip(right - 1, 0, len(t_obs) - 1)
    right = np.clip(right, 0, len(t_obs) - 1)
    gap = t_obs[right] - t_obs[left]
    out[inside & ~exact & (gap > max_gap)] = np.nan
    return out


def regularize(
    obs: pd.DataFrame,
    index: str,
    start,
    end,
    step_days: int,
    max_gap_days: int,
    stat: str = "mean",
) -> pd.DataFrame:
    """Interpolate one index onto a regular grid for every parcel.

    Parameters
    ----------
    obs : pandas.DataFrame
        Long table from :func:`parcelmon.extract.parcel_timeseries`, already
        filtered with :func:`filter_observations`.
    index : str
        Index name, e.g. ``"ndvi"``.
    start, end : date-like
        Grid limits (inclusive).
    step_days : int
        Grid spacing.
    max_gap_days : int
        Grid points inside a gap between observations longer than this are
        NaN.
    stat : {"mean", "median"}

    Returns
    -------
    pandas.DataFrame indexed by ``parcel_id`` with one column per grid date.
    """
    grid = pd.date_range(start, end, freq=f"{step_days}D")
    tg = grid.to_numpy().astype("datetime64[D]").astype("int64")
    sub = obs[obs["index"] == index]
    rows = {}
    for pid, g in sub.groupby("parcel_id"):
        g = g.sort_values("date").drop_duplicates("date")
        to = pd.to_datetime(g["date"]).to_numpy().astype("datetime64[D]").astype("int64")
        rows[pid] = _interp_with_gaps(to, g[stat].to_numpy(dtype="float64"), tg, max_gap_days)
    wide = pd.DataFrame.from_dict(rows, orient="index", columns=grid)
    wide.index.name = "parcel_id"
    return wide


def smooth(wide: pd.DataFrame, window: int, polyorder: int) -> pd.DataFrame:
    """Savitzky-Golay smoothing of each row, applied to runs of consecutive
    finite values at least ``window`` long; shorter runs are left as they are.
    """
    if window % 2 == 0 or window <= polyorder:
        raise ValueError("window must be odd and greater than polyorder")
    arr = wide.to_numpy(dtype="float64").copy()
    for r in range(arr.shape[0]):
        finite = np.isfinite(arr[r])
        idx = np.flatnonzero(finite)
        if idx.size == 0:
            continue
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        for run in runs:
            if run.size >= window:
                arr[r, run] = savgol_filter(arr[r, run], window, polyorder)
    return pd.DataFrame(arr, index=wide.index, columns=wide.columns)


def n_valid_dates(obs: pd.DataFrame, index: str) -> pd.Series:
    """Number of retained observation dates per parcel for one index."""
    return obs[obs["index"] == index].groupby("parcel_id")["date"].nunique().rename("n_valid_dates")


def feature_matrix(wides: dict) -> pd.DataFrame:
    """Join regularised series of several indices into one feature table.

    Columns are named ``<index>_t<k>`` by position on the regular grid, so
    names never collide (even across a year boundary) and a model trained on
    one season applies to another season with the same grid. Parcels missing
    from an index get NaN features (imputed in :mod:`parcelmon.classify`).
    """
    parts = []
    for name, wide in wides.items():
        w = wide.copy()
        w.columns = [f"{name}_t{k:02d}" for k in range(w.shape[1])]
        parts.append(w)
    return pd.concat(parts, axis=1)
