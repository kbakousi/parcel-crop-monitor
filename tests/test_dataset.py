"""The labelled (xarray) entry point gives the same result as the array one and refuses misaligned grids."""

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("xarray")

from parcelmon import config, pipeline, synthetic  # noqa: E402

HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def setup():
    scene = synthetic.make_scene(n_side=8, seed=2, mislabel_rate=0.08)
    cfg = config.load_config(HERE / "synthetic_config.yaml")
    parcels = scene["parcels"].set_index("parcel_id")
    return scene, synthetic.to_dataset(scene), cfg, parcels


def test_to_dataset_coordinates_match_parcel_centroids(setup):
    scene, ds, _, parcels = setup
    assert ds["ndvi"].dims == ("time", "y", "x") and ds["labels"].dims == ("y", "x")
    lab = ds["labels"]
    for pid in parcels.index[:5]:
        cell = lab.where(lab == pid, drop=True)
        assert cell.x.mean() == pytest.approx(parcels.loc[pid, "x"])
        assert cell.y.mean() == pytest.approx(parcels.loc[pid, "y"])


def test_run_dataset_equals_array_run(setup):
    scene, ds, cfg, parcels = setup
    ref = pipeline.run(cfg, {"ndvi": scene["ndvi"]}, scene["labels"], scene["dates"], parcels, scene["clear"])
    got = pipeline.run_dataset(cfg, ds, parcels, labels="labels", clear="clear")
    pd.testing.assert_frame_equal(got.verdicts, ref.verdicts)
    pd.testing.assert_frame_equal(got.observations, ref.observations)


def test_run_dataset_accepts_dask_backed_data(setup):
    pytest.importorskip("dask")
    _, ds, cfg, parcels = setup
    ref = pipeline.run_dataset(cfg, ds, parcels, labels="labels", clear="clear")
    got = pipeline.run_dataset(cfg, ds.chunk({"time": 10, "y": 20}), parcels, labels="labels", clear="clear")
    pd.testing.assert_frame_equal(got.verdicts, ref.verdicts)


def test_misaligned_inputs_are_refused(setup):
    _, ds, cfg, parcels = setup
    shifted = ds["labels"].assign_coords(x=ds.x + 10.0)  # one pixel east
    with pytest.raises(ValueError, match="same grid"):
        pipeline.run_dataset(cfg, ds, parcels, labels=shifted, clear="clear")
    with pytest.raises(ValueError, match="coordinates"):
        pipeline.run_dataset(cfg, ds, parcels, labels=ds["labels"].drop_vars(["x", "y"]))
    with pytest.raises(ValueError, match="ndvi"):
        pipeline.run_dataset(cfg, ds.drop_vars("ndvi"), parcels)
