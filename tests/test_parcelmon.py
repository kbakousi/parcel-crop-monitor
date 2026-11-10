from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from parcelmon import classify, config, extract, pipeline, states, synthetic, timeseries, validate, verdict
from parcelmon.parcels import read_cpg_encoding

HERE = Path(__file__).parent


def test_config_requires_thresholds(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("parcels: {id_column: a, crop_column: b}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        config.load_config(p)
    cfg = config.load_config(HERE / "synthetic_config.yaml")
    assert cfg.verdict["p_high"] == 0.7


def test_example_config_is_valid():
    cfg = config.load_config(HERE.parent / "configs" / "brp_example.yaml")
    names = pd.Series(
        ["Grasland, blijvend", "Tarwe, winter-", "Mais, snij-", "Sloot", "Groene braak, spontane opkomst"]
    )
    assert config.map_crop_groups(names, cfg.crop_groups).tolist() == [
        "grassland", "winter_cereals", "maize", None, "fallow"]


def test_cpg_encoding(tmp_path):
    (tmp_path / "x.cpg").write_text("UTF-8", encoding="ascii")
    assert read_cpg_encoding(tmp_path / "x.shp") == "UTF-8"
    assert read_cpg_encoding(tmp_path / "y.shp", default="ISO-8859-7") == "ISO-8859-7"


def test_extract_uses_clear_pixels_only():
    labels = np.array([[1, 1], [2, 0]])
    ndvi = np.array([[[0.2, 0.4], [0.8, 0.9]]])
    clear = np.array([[[True, False], [True, True]]])
    obs = extract.parcel_timeseries({"ndvi": ndvi}, labels, ["2025-05-01"], clear)
    p1 = obs[obs.parcel_id == 1].iloc[0]
    assert p1["mean"] == pytest.approx(0.2) and p1["valid_fraction"] == pytest.approx(0.5)


def test_regularize_does_not_bridge_long_gaps_or_extrapolate():
    obs = pd.DataFrame({
        "parcel_id": 1, "index": "ndvi",
        "date": pd.to_datetime(["2025-03-10", "2025-03-20", "2025-06-20"]),
        "mean": [0.2, 0.4, 0.8], "n_valid": 10, "valid_fraction": 1.0,
    })
    wide = timeseries.regularize(obs, "ndvi", "2025-03-01", "2025-07-01", step_days=10, max_gap_days=40)
    row = wide.loc[1]
    assert np.isnan(row[pd.Timestamp("2025-03-01")])           # before first observation
    assert row[pd.Timestamp("2025-03-11")] == pytest.approx(0.22)
    assert np.isnan(row[pd.Timestamp("2025-04-30")])           # inside a 92-day gap
    assert np.isnan(row[pd.Timestamp("2025-06-29")])           # after last observation


def test_youden_threshold_separates_groups():
    thr, j = states.youden_threshold([0.10, 0.15, 0.2, 0.22], [0.5, 0.6, 0.7, 0.3])
    assert 0.22 < thr <= 0.3 and j == pytest.approx(1.0)


def test_detect_drops_counts_repeated_cuts_not_senescence():
    dates = pd.date_range("2025-04-01", periods=12, freq="10D")
    mowing = [0.4, 0.6, 0.8, 0.4, 0.6, 0.8, 0.8, 0.4, 0.6, 0.7, 0.7, 0.7]
    assert len(states.detect_drops(dates, mowing, min_drop=0.25, max_drop_days=30)) == 2  # two cuts
    slow = np.linspace(0.8, 0.2, 12)  # 0.055 per 10 days: never 0.25 within 30 days
    assert states.detect_drops(dates, slow, min_drop=0.25, max_drop_days=30) == []


def test_out_of_fold_never_trains_on_predicted_parcel():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 3)))
    y = pd.Series(np.repeat(["a", "b"], 20))
    groups = np.arange(40) // 5
    proba, fold = classify.out_of_fold_proba(X, y, groups, n_splits=4, random_state=0, n_estimators=20)
    assert np.allclose(proba.sum(axis=1), 1.0)
    for g in np.unique(groups):  # a block is never split across folds
        assert fold[groups == g].nunique() == 1
    with pytest.raises(ValueError, match="spatial blocks"):
        classify.out_of_fold_proba(X, y, np.zeros(40), n_splits=4, random_state=0)


def test_verdict_rules():
    proba = pd.DataFrame({"maize": [0.9, 0.1, 0.5, 0.9], "grass": [0.1, 0.9, 0.5, 0.1]}, index=[1, 2, 3, 4])
    declared = pd.Series({1: "maize", 2: "maize", 3: "maize", 4: "maize", 5: "maize", 6: None})
    out = verdict.verdicts(declared, proba, 0.7, 0.3, blocking_reasons={5: ["too_small"]},
                           rule_reasons={4: ["bare_all_season"]}, downgrade_rules={"maize": ["bare_all_season"]})
    assert out.verdict.to_dict() == {1: "agree", 2: "disagree", 3: "inconclusive", 4: "inconclusive",
                                     5: "inconclusive", 6: "inconclusive"}
    assert "rule_conflict" in out.loc[4, "reasons"] and out.loc[5, "reasons"] == ["too_small"]
    with pytest.raises(ValueError):
        verdict.verdicts(declared, proba, 0.3, 0.7)


def test_accuracy_report():
    rep = validate.accuracy_report(["a", "a", "b", "b"], ["a", "b", "b", "b"])
    assert rep["overall_accuracy"] == 0.75
    assert rep["confusion"].loc["a", "b"] == 1


def test_end_to_end_on_synthetic_scene():
    scene = synthetic.make_scene(n_side=10, seed=1, mislabel_rate=0.08)
    cfg = config.load_config(HERE / "synthetic_config.yaml")
    parcels = scene["parcels"].set_index("parcel_id")
    res = pipeline.run(cfg, {"ndvi": scene["ndvi"]}, scene["labels"], scene["dates"], parcels, scene["clear"])
    v = res.verdicts
    assert res.report["overall_accuracy"] > 0.85
    flipped = parcels.index[parcels.true_group != parcels.declared_group]
    honest = parcels.index[parcels.true_group == parcels.declared_group]
    # most mislabelled parcels are flagged (disagree or inconclusive); most honest ones agree
    assert (v.loc[flipped, "verdict"] != "agree").mean() >= 0.75
    assert (v.loc[honest, "verdict"] == "agree").mean() >= 0.8
    grass = parcels.index[parcels.true_group == "grassland"]
    assert res.rules.loc[grass, "n_drop_events"].median() >= 2


def test_too_small_flags_parcels_lost_in_buffer():
    from parcelmon.parcels import too_small

    labels = np.array([[1, 1, 0], [1, 2, 0]])
    id_map = pd.Series(["p1", "p2"], index=[1, 2])
    out = too_small(labels, id_map, min_pixels=2, all_ids=["p1", "p2", "p3"])
    assert out.to_dict() == {"p1": False, "p2": True, "p3": True}


def test_feature_names_do_not_collide_across_years():
    grid = pd.date_range("2024-11-01", "2025-11-10", freq="10D")
    wide = pd.DataFrame([np.arange(len(grid), dtype=float)], index=[1], columns=grid)
    X = timeseries.feature_matrix({"ndvi": wide})
    assert X.columns.is_unique and X.shape[1] == len(grid)
