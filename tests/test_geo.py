"""Tests of the geospatial code paths (skipped when geopandas/rasterio are not installed)."""

import numpy as np
import pandas as pd
import pytest

gpd = pytest.importorskip("geopandas")
pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import box  # noqa: E402

from parcelmon import parcels  # noqa: E402
from parcelmon.loaders import brp  # noqa: E402


def squares():
    # two 100 m x 100 m parcels and one 15 m x 15 m parcel, Dutch RD New (metres)
    geoms = [box(0, 0, 100, 100), box(120, 0, 220, 100), box(240, 0, 255, 15)]
    return gpd.GeoDataFrame({"parcel_id": ["a", "b", "tiny"], "gewas": ["Tarwe, winter-", "Uien", "Grasland"]},
                            geometry=geoms, crs="EPSG:28992")


def test_read_parcels_uses_cpg_encoding(tmp_path):
    g = squares()
    g["gewas"] = ["Σιτάρι", "Βαμβάκι", "Ελαιώνες"]
    path = tmp_path / "parcels.shp"
    g.to_file(path, encoding="UTF-8")
    assert (tmp_path / "parcels.cpg").read_text().strip().upper().replace("-", "") == "UTF8"
    back = parcels.read_parcels(path)
    assert back["gewas"].tolist() == ["Σιτάρι", "Βαμβάκι", "Ελαιώνες"]


def test_inward_buffer_requires_projected_crs():
    g = squares()
    shrunk = parcels.inward_buffer(g, 10)
    assert shrunk.area.iloc[0] == pytest.approx(80 * 80)
    assert shrunk.geometry.iloc[2].is_empty  # 15 m square vanishes with a 10 m inward buffer
    with pytest.raises(ValueError, match="projected"):
        parcels.inward_buffer(g.to_crs("EPSG:4326"), 10)


def test_rasterize_centre_based_and_too_small():
    g = parcels.inward_buffer(squares(), 10)
    transform = from_origin(0, 100, 10, 10)  # 10 m pixels, north-up
    labels, id_map = parcels.rasterize_parcels(g, "parcel_id", transform, (10, 26))
    counts = pd.Series(labels.ravel()).value_counts()
    assert counts[1] == 64 and counts[2] == 64  # 80 m x 80 m = 8 x 8 pixel centres
    assert id_map.tolist() == ["a", "b"]  # the vanished parcel has no label
    small = parcels.too_small(labels, id_map, min_pixels=8, all_ids=g["parcel_id"])
    assert small.to_dict() == {"a": False, "b": False, "tiny": True}
    assert labels[0, 0] == 0  # the buffered-away edge pixel is outside the parcel


def test_rasterize_like_reads_the_grid_and_checks_the_crs():
    xarray = pytest.importorskip("xarray")
    from pyproj import CRS

    g = parcels.inward_buffer(squares(), 10)
    wkt = CRS.from_epsg(28992).to_wkt()
    grid = xarray.Dataset(
        coords={"y": 95.0 - 10 * np.arange(10), "x": 5.0 + 10 * np.arange(26),
                "spatial_ref": xarray.DataArray(0, attrs={"crs_wkt": wkt})}
    )
    labels, id_map = parcels.rasterize_like(g, "parcel_id", grid)
    ref, ref_map = parcels.rasterize_parcels(g, "parcel_id", from_origin(0, 100, 10, 10), (10, 26))
    assert np.array_equal(labels.values, ref) and id_map.equals(ref_map)
    assert labels.dims == ("y", "x") and np.array_equal(labels.x.values, grid.x.values)
    assert "spatial_ref" in labels.coords
    with pytest.raises(ValueError, match="reproject"):
        parcels.rasterize_like(g.to_crs("EPSG:3035"), "parcel_id", grid)
    with pytest.raises(ValueError, match="no CRS"):
        parcels.rasterize_like(g, "parcel_id", grid.drop_vars("spatial_ref"))


def test_brp_features_to_geodataframe():
    feats = [{"type": "Feature", "id": "x1",
              "properties": {"category": "Bouwland", "gewas": "Mais, snij-", "gewascode": 259, "jaar": 2025},
              "geometry": {"type": "Polygon",
                           "coordinates": [[[5.60, 52.50], [5.601, 52.50], [5.601, 52.501], [5.60, 52.501],
                                            [5.60, 52.50]]]}}]
    g = brp.to_geodataframe(feats)
    assert g.crs.to_epsg() == 28992 and g["parcel_id"].tolist() == ["x1"]
    assert 4000 < g.area.iloc[0] < 9000  # about 68 m x 111 m
    assert np.isfinite(g.geometry.centroid.x.iloc[0])
