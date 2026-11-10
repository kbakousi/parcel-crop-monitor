"""Rule-based land-use states and events from regular parcel time series.

States per date are ``bare``, ``low_cover`` or ``vegetated`` from two index
thresholds. Thresholds are calibrated from labelled parcels with Youden's J
rather than fixed by hand. Drop events (e.g. harvest or mowing) are abrupt
falls after a peak.

References
----------
Kanjir, U., Duric, N., Veljanovski, T. (2018). Sentinel-2 based temporal
    detection of agricultural land use anomalies in support of Common
    Agricultural Policy monitoring. ISPRS International Journal of
    Geo-Information 7, 405. https://doi.org/10.3390/ijgi7100405
Youden, W.J. (1950). Index for rating diagnostic tests. Cancer 3, 32-35.
    https://doi.org/10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["youden_threshold", "classify_states", "detect_drops", "parcel_rule_summary"]


def youden_threshold(negative, positive):
    """Threshold maximising Youden's J = sensitivity + specificity - 1.

    A value ``>= threshold`` is predicted positive. Use e.g. NDVI of parcels
    known to be bare (negative) and known to be vegetated (positive).

    Returns
    -------
    threshold : float
    j : float
    """
    neg = np.asarray(negative, dtype="float64")
    pos = np.asarray(positive, dtype="float64")
    neg, pos = neg[np.isfinite(neg)], pos[np.isfinite(pos)]
    if neg.size == 0 or pos.size == 0:
        raise ValueError("both groups need finite values")
    cand = np.unique(np.concatenate([neg, pos]))
    tpr = 1 - np.searchsorted(np.sort(pos), cand, side="left") / pos.size
    fpr = 1 - np.searchsorted(np.sort(neg), cand, side="left") / neg.size
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(cand[k]), float(j[k])


def classify_states(wide: pd.DataFrame, bare_max: float, vegetated_min: float) -> pd.DataFrame:
    """Label every parcel-date as bare, low_cover, vegetated or no_data.

    ``bare``: value < ``bare_max``; ``vegetated``: value >= ``vegetated_min``;
    ``low_cover`` in between.
    """
    if not bare_max <= vegetated_min:
        raise ValueError("bare_max must be <= vegetated_min")
    v = wide.to_numpy(dtype="float64")
    out = np.full(v.shape, "low_cover", dtype=object)
    out[v < bare_max] = "bare"
    out[v >= vegetated_min] = "vegetated"
    out[~np.isfinite(v)] = "no_data"
    return pd.DataFrame(out, index=wide.index, columns=wide.columns)


def detect_drops(dates, values, min_drop: float, max_drop_days: int, min_peak: float | None = None) -> list[dict]:
    """Find abrupt drops: a fall of at least ``min_drop`` within
    ``max_drop_days`` after the most recent peak.

    After an event the reference peak restarts at the post-drop value, so
    regrowth followed by another cut is a new event (e.g. repeated mowing).
    If the series declines slowly for longer than ``max_drop_days`` the
    reference peak moves forward, so gradual senescence is not an event.

    Returns a list of ``{"peak_date", "drop_date", "peak", "value", "drop"}``.
    """
    d = pd.to_datetime(np.asarray(dates))
    v = np.asarray(values, dtype="float64")
    ok = np.isfinite(v)
    d, v = d[ok], v[ok]
    events = []
    if v.size < 2:
        return events
    peak_v, peak_t = v[0], d[0]
    for t, x in zip(d[1:], v[1:], strict=True):
        days = (t - peak_t).days
        if x >= peak_v:
            peak_v, peak_t = x, t
        elif peak_v - x >= min_drop and days <= max_drop_days and (min_peak is None or peak_v >= min_peak):
            events.append({"peak_date": peak_t, "drop_date": t, "peak": peak_v, "value": x, "drop": peak_v - x})
            peak_v, peak_t = x, t
        elif days > max_drop_days:
            peak_v, peak_t = x, t
    return events


def parcel_rule_summary(
    states: pd.DataFrame,
    wide: pd.DataFrame,
    min_drop: float,
    max_drop_days: int,
    bare_season_fraction: float,
    min_peak: float | None = None,
) -> pd.DataFrame:
    """Per-parcel season summary and rule reason codes.

    Columns: ``frac_bare``, ``frac_vegetated``, ``n_drop_events``,
    ``first_drop_date``, ``max_value`` and ``rule_reasons`` (a list):

    - ``bare_all_season``: bare on at least ``bare_season_fraction`` of the
      valid dates;
    - ``no_vegetation_peak``: never reaches the vegetated state;
    - ``no_drop_event``: no harvest/mowing-like drop found.
    """
    rows = []
    for pid in states.index:
        s = states.loc[pid]
        valid = s != "no_data"
        n = int(valid.sum())
        frac_bare = float((s == "bare").sum() / n) if n else np.nan
        frac_veg = float((s == "vegetated").sum() / n) if n else np.nan
        ev = detect_drops(wide.columns, wide.loc[pid].to_numpy(), min_drop, max_drop_days, min_peak)
        reasons = []
        if n and frac_bare >= bare_season_fraction:
            reasons.append("bare_all_season")
        if n and frac_veg == 0:
            reasons.append("no_vegetation_peak")
        if n and not ev:
            reasons.append("no_drop_event")
        rows.append(
            {
                "parcel_id": pid,
                "frac_bare": frac_bare,
                "frac_vegetated": frac_veg,
                "n_drop_events": len(ev),
                "first_drop_date": ev[0]["drop_date"] if ev else pd.NaT,
                "max_value": float(np.nanmax(wide.loc[pid].to_numpy())) if n else np.nan,
                "rule_reasons": reasons,
            }
        )
    return pd.DataFrame(rows).set_index("parcel_id")
