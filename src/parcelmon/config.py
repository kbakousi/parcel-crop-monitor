"""Configuration: column names, crop-group rules and method parameters.

Thresholds that change the result (probabilities, minimum sizes, drop sizes)
have no hidden defaults: they must be stated in the config file, so every
run documents its own choices.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

__all__ = ["RULE_REASONS", "Config", "load_config", "map_crop_groups"]


@dataclass
class Config:
    """Parsed configuration. See ``configs/brp_example.yaml`` for every field."""

    parcels: dict
    crop_groups: dict
    preprocessing: dict
    timeseries: dict
    events: dict
    classifier: dict
    verdict: dict
    states: dict = field(default_factory=dict)
    exclude: dict = field(default_factory=dict)

    _required = {
        "parcels": ["id_column", "crop_column"],
        "preprocessing": ["inward_buffer_m", "min_pixels"],
        "timeseries": [
            "indices", "primary_index", "season_start", "season_end",
            "min_valid_fraction", "min_valid_dates", "step_days", "max_gap_days",
        ],
        "states": ["bare_max", "vegetated_min", "bare_season_fraction"],
        "events": ["min_drop", "max_drop_days"],
        "classifier": ["n_splits", "block_size_m", "random_state"],
        "verdict": ["p_high", "p_low"],
    }

    def validate(self) -> Config:
        """Check presence, types, ranges and cross-field rules; report every problem at once.

        Raises ``ValueError`` listing all problems. Unknown keys in a section
        (often a typo, e.g. ``p_hihg``) give a ``UserWarning``.
        """
        errors: list[str] = []
        for section, keys in self._required.items():
            block = getattr(self, section) or {}
            if not isinstance(block, dict):
                errors.append(f"section '{section}' must be a mapping")
                continue
            missing = [k for k in keys if k not in block]
            if missing:
                errors.append(f"section '{section}' is missing {missing}")
        if not errors:
            _check_values(self, errors)
        if errors:
            raise ValueError("invalid config:\n- " + "\n- ".join(errors))
        _warn_unknown(self)
        return self


#: Rule reason codes produced by :func:`parcelmon.states.parcel_rule_summary`.
RULE_REASONS = ("bare_all_season", "no_vegetation_peak", "no_drop_event")

#: Known keys per section (required and optional), for typo warnings.
_KNOWN = {
    "parcels": {"id_column", "crop_column", "crs", "source"},
    "preprocessing": {"inward_buffer_m", "min_pixels"},
    "timeseries": {"indices", "primary_index", "season_start", "season_end", "min_valid_fraction",
                   "min_valid_pixels", "min_valid_dates", "step_days", "max_gap_days", "smoothing"},
    "states": {"bare_max", "vegetated_min", "bare_season_fraction"},
    "events": {"min_drop", "max_drop_days", "min_peak"},
    "classifier": {"n_splits", "block_size_m", "random_state", "n_estimators", "max_features", "min_samples_leaf"},
    "verdict": {"p_high", "p_low", "downgrade_rules"},
}


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _check_values(cfg: Config, errors: list) -> None:
    def num(section, key, lo=None, hi=None, lo_open=False, integer=False, optional=False):
        block = getattr(cfg, section)
        if key not in block:
            if not optional:
                errors.append(f"{section}.{key} is missing")
            return None
        x = block[key]
        if not (_is_int(x) if integer else _is_num(x)):
            kind = "an integer" if integer else "a number"
            errors.append(f"{section}.{key} must be {kind}, got {x!r}")
            return None
        if lo is not None and (x <= lo if lo_open else x < lo):
            errors.append(f"{section}.{key} must be {'>' if lo_open else '>='} {lo}, got {x}")
        if hi is not None and x > hi:
            errors.append(f"{section}.{key} must be <= {hi}, got {x}")
        return x

    for key in ("id_column", "crop_column"):
        if not isinstance(cfg.parcels[key], str) or not cfg.parcels[key]:
            errors.append(f"parcels.{key} must be a column name")

    num("preprocessing", "inward_buffer_m", lo=0)
    num("preprocessing", "min_pixels", lo=1, integer=True)

    ts = cfg.timeseries
    idx = ts["indices"]
    if not (isinstance(idx, list) and idx and all(isinstance(i, str) for i in idx)):
        errors.append("timeseries.indices must be a non-empty list of index names")
    elif ts["primary_index"] not in idx:
        errors.append(f"timeseries.primary_index ({ts['primary_index']!r}) must be one of timeseries.indices")
    try:
        start, end = pd.Timestamp(ts["season_start"]), pd.Timestamp(ts["season_end"])
        if not start < end:
            errors.append("timeseries.season_start must be before season_end")
    except (ValueError, TypeError):
        errors.append("timeseries.season_start and season_end must be dates (YYYY-MM-DD)")
    num("timeseries", "min_valid_fraction", lo=0, hi=1, lo_open=True)
    num("timeseries", "min_valid_pixels", lo=1, integer=True, optional=True)
    num("timeseries", "min_valid_dates", lo=1, integer=True)
    step = num("timeseries", "step_days", lo=1, integer=True)
    gap = num("timeseries", "max_gap_days", lo=0, lo_open=True)
    if step is not None and gap is not None and gap < step:
        errors.append("timeseries.max_gap_days must be >= step_days")
    sm = ts.get("smoothing")
    if sm:
        sm = sm if isinstance(sm, dict) else {}
        w, o = sm.get("window"), sm.get("polyorder")
        if not (_is_int(w) and w >= 3 and w % 2 == 1):
            errors.append("timeseries.smoothing.window must be an odd integer >= 3")
        if not (_is_int(o) and o >= 0 and (not _is_int(w) or o < w)):
            errors.append("timeseries.smoothing.polyorder must be an integer >= 0 and < window")

    bare = num("states", "bare_max", lo=-1, hi=1)
    veg = num("states", "vegetated_min", lo=-1, hi=1)
    if bare is not None and veg is not None and not bare < veg:
        errors.append("states.bare_max must be below states.vegetated_min")
    num("states", "bare_season_fraction", lo=0, hi=1, lo_open=True)

    num("events", "min_drop", lo=0, lo_open=True)
    num("events", "max_drop_days", lo=1, integer=True)
    num("events", "min_peak", lo=-1, hi=1, optional=True)

    num("classifier", "n_splits", lo=2, integer=True)
    num("classifier", "block_size_m", lo=0, lo_open=True)
    num("classifier", "random_state", lo=0, integer=True)
    num("classifier", "n_estimators", lo=1, integer=True, optional=True)
    num("classifier", "min_samples_leaf", lo=1, integer=True, optional=True)

    lo, hi = num("verdict", "p_low", lo=0, hi=1), num("verdict", "p_high", lo=0, hi=1)
    if lo is not None and hi is not None and not lo < hi:
        errors.append("verdict thresholds must satisfy 0 <= p_low < p_high <= 1")
    for grp, rules in (cfg.verdict.get("downgrade_rules") or {}).items():
        bad = [r for r in (rules or []) if r not in RULE_REASONS]
        if bad:
            errors.append(f"verdict.downgrade_rules.{grp}: unknown rule(s) {bad}; known: {list(RULE_REASONS)}")

    cg = cfg.crop_groups
    if "codes" not in cg and "patterns" not in cg:
        errors.append("crop_groups needs 'codes' (exact) or 'patterns' (regular expressions)")
    for grp, pats in (cg.get("patterns") or {}).items():
        for pat in pats or []:
            try:
                re.compile(pat)
            except re.error as exc:
                errors.append(f"crop_groups.patterns.{grp}: invalid regular expression {pat!r} ({exc})")


def _warn_unknown(cfg: Config) -> None:
    for section, known in _KNOWN.items():
        unknown = sorted(set(getattr(cfg, section) or {}) - known)
        if unknown:
            warnings.warn(f"config section '{section}' has unknown key(s) {unknown} (typo?)", UserWarning,
                          stacklevel=4)


def load_config(path: str | Path) -> Config:
    """Read a YAML config file and validate it."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    keys = {k: raw.get(k, {}) for k in Config.__dataclass_fields__ if not k.startswith("_")}
    return Config(**keys).validate()


def map_crop_groups(values: pd.Series, crop_groups: dict) -> pd.Series:
    """Map declared crop codes or names to crop groups.

    ``crop_groups`` holds either ``codes`` (``{group: [code, ...]}``, exact
    match after conversion to string) or ``patterns`` (``{group: [regex,
    ...]}``, case-insensitive, first matching group wins, in file order).
    Unmatched values map to ``crop_groups.get("other")``, which may be
    ``None`` (unmatched parcels are then left out of training and
    verification).
    """
    s = values.astype(str)
    out = pd.Series([None] * len(s), index=s.index, dtype=object)
    if "codes" in crop_groups:
        lookup = {str(code): grp for grp, codes in crop_groups["codes"].items() for code in codes}
        out = s.map(lookup).astype(object)
    else:
        for grp, pats in crop_groups["patterns"].items():
            rx = re.compile("|".join(f"(?:{p})" for p in pats), flags=re.IGNORECASE)
            hit = out.isna() & s.str.contains(rx)
            out[hit] = grp
    other = crop_groups.get("other")
    if other is not None:
        out = out.where(out.notna(), other)
    return out
