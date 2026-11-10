"""Config validation: wrong types, ranges and contradictions fail early, all reported together."""

import copy
from pathlib import Path

import pytest
import yaml

from parcelmon.config import Config

HERE = Path(__file__).parent
BASE = yaml.safe_load((HERE / "synthetic_config.yaml").read_text(encoding="utf-8"))


def build(**changes):
    raw = copy.deepcopy(BASE)
    for dotted, value in changes.items():
        *path, key = dotted.split("__")
        block = raw
        for part in path:
            block = block.setdefault(part, {})
        if value is _DROP:
            block.pop(key)
        else:
            block[key] = value
    return Config(**{k: raw.get(k, {}) for k in Config.__dataclass_fields__ if not k.startswith("_")}).validate()


_DROP = object()


def test_base_config_is_valid():
    assert build().verdict["p_high"] == 0.7


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"verdict__p_high": "0.7"}, "verdict.p_high must be a number"),
        ({"verdict__p_low": 0.8}, "p_low < p_high"),
        ({"verdict__p_high": 1.5}, "verdict.p_high must be <= 1"),
        ({"preprocessing__min_pixels": True}, "min_pixels must be an integer"),
        ({"preprocessing__min_pixels": 2.5}, "min_pixels must be an integer"),
        ({"timeseries__season_end": "2025-01-01"}, "season_start must be before season_end"),
        ({"timeseries__season_start": "spring"}, "must be dates"),
        ({"timeseries__primary_index": "ndmi"}, "must be one of timeseries.indices"),
        ({"timeseries__max_gap_days": 5}, "max_gap_days must be >= step_days"),
        ({"timeseries__smoothing": {"window": 4, "polyorder": 2}}, "window must be an odd integer"),
        ({"timeseries__smoothing": {"window": 5, "polyorder": 5}}, "polyorder must be an integer >= 0 and < window"),
        ({"timeseries__min_valid_fraction": 0}, "min_valid_fraction must be > 0"),
        ({"states__bare_max": 0.6}, "bare_max must be below states.vegetated_min"),
        ({"classifier__n_splits": 1}, "n_splits must be >= 2"),
        ({"events__min_drop": -0.1}, "min_drop must be > 0"),
        ({"verdict__downgrade_rules": {"maize": ["bare_all_seasons"]}}, "unknown rule(s) ['bare_all_seasons']"),
        ({"crop_groups": {"patterns": {"maize": ["^Mais("]}}}, "invalid regular expression"),
        ({"verdict__p_low": _DROP}, "section 'verdict' is missing ['p_low']"),
    ],
)
def test_bad_values_are_rejected(changes, message):
    with pytest.raises(ValueError) as err:
        build(**changes)
    assert message in str(err.value)


def test_all_problems_are_reported_together():
    with pytest.raises(ValueError) as err:
        build(verdict__p_low=0.9, states__bare_max=0.7, classifier__n_splits=1)
    text = str(err.value)
    assert text.count("\n- ") == 3


def test_unknown_keys_warn():
    with pytest.warns(UserWarning, match=r"unknown key\(s\) \['p_hihg'\]"):
        build(verdict__p_hihg=0.8)
