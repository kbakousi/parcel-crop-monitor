"""Tiled parcel statistics must equal the whole-image result exactly, wherever parcels fall."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from parcelmon import config, extract, pipeline, synthetic

HERE = Path(__file__).parent


def irregular_scene(seed=0, shape=(41, 37), n_parcels=30, n_dates=6):
    """Voronoi-like parcels of uneven size and shape, background gaps, one parcel in two pieces, NaNs and clouds."""
    rng = np.random.default_rng(seed)
    rr, cc = np.indices(shape)
    seeds = rng.uniform(0, max(shape), (n_parcels, 2))
    d = (rr[..., None] - seeds[:, 0]) ** 2 + (cc[..., None] - seeds[:, 1]) ** 2
    labels = (np.argmin(d, axis=-1) + 1).astype("int32")
    labels[rng.random(shape) < 0.1] = 0  # field margins / non-parcel pixels
    labels[0:3, 0:3] = 99  # parcel 99 has two separate pieces, far apart
    labels[-3:, -3:] = 99
    stacks = {name: rng.normal(0.5, 0.2, (n_dates,) + shape) for name in ("ndvi", "ndmi")}
    stacks["ndvi"][rng.random(stacks["ndvi"].shape) < 0.05] = np.nan
    clear = rng.random((n_dates,) + shape) > 0.2
    dates = pd.date_range("2025-04-01", periods=n_dates, freq="5D")
    id_map = pd.Series({lab: f"p{lab}" for lab in np.unique(labels) if lab})
    return stacks, labels, dates, clear, id_map


@pytest.mark.parametrize("tile_size", [1, 4, 9, 16, 1000])
def test_tiled_equals_whole_image(tile_size):
    stacks, labels, dates, clear, id_map = irregular_scene()
    ref = extract.parcel_timeseries(stacks, labels, dates, clear, id_map)
    got = extract.parcel_timeseries_tiled(stacks, labels, dates, clear, id_map, tile_size=tile_size)
    pd.testing.assert_frame_equal(got, ref)


def test_border_parcels():
    labels = np.zeros((6, 6), dtype="int32")
    labels[0:2, 0:2] = 1  # inside the first block
    labels[2:5, 1:2] = 2  # crosses the row edge at 3
    labels[4:6, 4:6] = 3  # inside the last block
    labels[0, 5] = labels[5, 0] = 4  # two pieces in different blocks
    lookup = extract.border_parcels(labels, np.array([0, 3, 6]), np.array([0, 3, 6]))
    assert lookup.tolist() == [False, False, True, False, True]
    _, labels, *_ = irregular_scene(seed=1)
    assert extract.border_parcels(labels, np.array([0, 9, 18, 27, 36, 41]), np.array([0, 9, 18, 27, 37])).sum() > 5


def test_dask_input_is_read_by_window_with_the_same_result():
    da = pytest.importorskip("dask.array")
    stacks, labels, dates, clear, id_map = irregular_scene(seed=2)
    ref = extract.parcel_timeseries(stacks, labels, dates, clear, id_map)
    lazy = {k: da.from_array(v, chunks=(2, 10, 10)) for k, v in stacks.items()}
    got = extract.parcel_timeseries_tiled(lazy, labels, dates, da.from_array(clear, chunks=(3, 10, 10)), id_map,
                                          tile_size=10)
    pd.testing.assert_frame_equal(got, ref)


def test_shape_checks_and_empty_labels():
    stacks, labels, dates, clear, _ = irregular_scene()
    with pytest.raises(ValueError, match="spatial shape"):
        extract.parcel_timeseries_tiled(stacks, labels[:-1], dates)
    with pytest.raises(ValueError, match="clear has shape"):
        extract.parcel_timeseries_tiled(stacks, labels, dates, clear[:-1])
    empty = extract.parcel_timeseries_tiled(stacks, np.zeros_like(labels), dates)
    assert empty.empty and list(empty.columns)[:3] == ["parcel_id", "date", "index"]


def test_pipeline_tiled_equals_untiled():
    scene = synthetic.make_scene(n_side=8, seed=4, mislabel_rate=0.08)
    cfg = config.load_config(HERE / "synthetic_config.yaml")
    parcels = scene["parcels"].set_index("parcel_id")
    args = (cfg, {"ndvi": scene["ndvi"]}, scene["labels"], scene["dates"], parcels, scene["clear"])
    ref = pipeline.run(*args)
    got = pipeline.run(*args, tiled=True, tile_size=13)
    pd.testing.assert_frame_equal(got.verdicts, ref.verdicts)
