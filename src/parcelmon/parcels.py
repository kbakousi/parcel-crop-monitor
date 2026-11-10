"""Parcel preparation: reading, inward buffering and rasterising.

Parcels are shrunk by an inward (negative) buffer before rasterising, so
that pixels mixing the parcel with its neighbours are dropped; parcels with
too few remaining pixels are reported as too small rather than classified
(Sitokonstantinou et al. 2018).

Optional dependencies: ``pip install parcelmon[geo]`` (geopandas, shapely,
rasterio).

References
----------
Sitokonstantinou, V. et al. (2018). Scalable parcel-based crop
    identification scheme using Sentinel-2 data time-series for the
    monitoring of the Common Agricultural Policy. Remote Sensing 10, 911.
    https://doi.org/10.3390/rs10060911
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "read_cpg_encoding",
    "read_parcels",
    "inward_buffer",
    "rasterize_parcels",
    "grid_crs",
    "grid_transform",
    "rasterize_like",
    "too_small",
]


def read_cpg_encoding(path: str | Path, default: str = "UTF-8") -> str:
    """Return the encoding declared in a shapefile's ``.cpg`` sidecar file."""
    cpg = Path(path).with_suffix(".cpg")
    if cpg.exists():
        text = cpg.read_text(encoding="ascii", errors="ignore").strip()
        if text:
            return text
    return default


def read_parcels(path: str | Path, encoding: str | None = None):
    """Read a parcel file with geopandas, taking the encoding from ``.cpg``.

    Reading a UTF-8 shapefile with another encoding garbles non-ASCII crop
    names (e.g. Greek), so the declared encoding is used unless overridden.
    """
    import geopandas as gpd

    path = Path(path)
    if encoding is None and path.suffix.lower() == ".shp":
        encoding = read_cpg_encoding(path)
    kwargs = {"encoding": encoding} if encoding else {}
    return gpd.read_file(path, **kwargs)


def inward_buffer(gdf, distance_m: float):
    """Shrink parcels by ``distance_m`` metres; parcels that vanish become empty.

    The layer must be in a projected CRS with metre units.
    """
    if distance_m < 0:
        raise ValueError("distance_m is the inward distance and must be >= 0")
    crs = gdf.crs
    if crs is None or not crs.is_projected:
        raise ValueError("parcels must be in a projected CRS (metres) before buffering")
    out = gdf.copy()
    out["geometry"] = gdf.geometry.buffer(-distance_m)
    return out


def rasterize_parcels(gdf, id_column: str, transform, out_shape):
    """Burn parcels into an integer label raster (0 = no parcel).

    Pixels are assigned by their centre (``all_touched=False``). Returns the
    label raster and a ``pandas.Series`` mapping label -> parcel id. Empty
    geometries are skipped.
    """
    from rasterio import features

    valid = gdf[[g is not None and not g.is_empty for g in gdf.geometry]]
    labels = np.arange(1, len(valid) + 1, dtype="int32")
    shapes = zip(valid.geometry, labels, strict=True)
    raster = features.rasterize(shapes, out_shape=out_shape, transform=transform, fill=0, dtype="int32")
    id_map = pd.Series(valid[id_column].to_numpy(), index=labels, name=id_column)
    id_map.index.name = "label"
    return raster, id_map


def grid_crs(obj):
    """Return the ``pyproj.CRS`` stored in an xarray object's ``spatial_ref`` coordinate, or None.

    odc-stac and rioxarray write the CRS there (attributes ``crs_wkt`` or
    ``spatial_ref``).
    """
    if "spatial_ref" not in obj.coords:
        return None
    attrs = obj.coords["spatial_ref"].attrs
    wkt = attrs.get("crs_wkt") or attrs.get("spatial_ref")
    if not wkt:
        return None
    from pyproj import CRS

    return CRS.from_user_input(wkt)


def grid_transform(obj, x: str = "x", y: str = "y"):
    """Affine transform of a regular grid from its pixel-centre coordinates."""
    from affine import Affine

    xs = np.asarray(obj[x].values, dtype="float64")
    ys = np.asarray(obj[y].values, dtype="float64")
    if xs.size < 2 or ys.size < 2:
        raise ValueError("the grid needs at least two pixels along x and y")
    dx, dy = np.diff(xs), np.diff(ys)
    if not (np.allclose(dx, dx[0]) and np.allclose(dy, dy[0])):
        raise ValueError("the grid is not regular")
    return Affine(dx[0], 0.0, xs[0] - dx[0] / 2, 0.0, dy[0], ys[0] - dy[0] / 2)


def rasterize_like(gdf, id_column: str, like, crs=None, x: str = "x", y: str = "y"):
    """Rasterise parcels onto the grid of an xarray object (e.g. the image Dataset).

    The grid (transform and shape) is read from ``like``'s coordinates, so the
    label raster is aligned with the images by construction. The parcels'
    CRS must equal the grid's CRS (read from ``spatial_ref``, or given as
    ``crs``); otherwise a ``ValueError`` asks you to reproject.

    Returns
    -------
    labels : xarray.DataArray (y, x)
        Integer labels (0 = no parcel) with ``like``'s coordinates.
    id_map : pandas.Series
        label -> parcel id, as from :func:`rasterize_parcels`.
    """
    import xarray
    from pyproj import CRS

    grid = CRS.from_user_input(crs) if crs is not None else grid_crs(like)
    if grid is None:
        raise ValueError("the grid has no CRS ('spatial_ref' coordinate); pass crs=")
    if gdf.crs is None or CRS.from_user_input(gdf.crs) != grid:
        raise ValueError(f"parcels are in {gdf.crs} but the grid is in {grid.to_string()}; "
                         "reproject the parcels with gdf.to_crs(...) first")
    shape = (like.sizes[y], like.sizes[x])
    raster, id_map = rasterize_parcels(gdf, id_column, grid_transform(like, x, y), shape)
    coords = {y: like[y], x: like[x]}
    if "spatial_ref" in like.coords:
        coords["spatial_ref"] = like.coords["spatial_ref"]
    return xarray.DataArray(raster, dims=(y, x), coords=coords, name="labels"), id_map


def too_small(label_raster, id_map: pd.Series, min_pixels: int, all_ids=None) -> pd.Series:
    """Return a boolean Series (indexed by parcel id): fewer than ``min_pixels`` pixels.

    Pass ``all_ids`` (every parcel id) so that parcels which vanished in the
    inward buffer, and are therefore not in ``id_map``, are flagged too.
    """
    counts = pd.Series(np.bincount(np.asarray(label_raster).ravel(), minlength=len(id_map) + 1))
    n = counts.reindex(id_map.index, fill_value=0).to_numpy()
    out = pd.Series(n < min_pixels, index=id_map.to_numpy(), name="too_small")
    if all_ids is not None:
        out = out.reindex(pd.Index(all_ids), fill_value=True)
    return out
