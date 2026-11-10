"""Real-data demo: Dutch BRP parcels + Sentinel-2 L2A from Microsoft Planetary Computer.

Requirements: network access and ``pip install -e ".[geo,stac]"``. No account
is needed. Not part of the automated tests.

Run:  python examples/demo_brp_sentinel2.py
The images are loaded lazily (dask) and kept as labelled xarray objects; the
parcel raster is built on the image grid itself, and parcel statistics are
computed one chunk at a time. The small bounding box keeps the download short.
"""

from pathlib import Path

import pandas as pd
import xarray as xr
from eolandscape import xr as exr
from eolandscape.loaders import stac

from parcelmon import config, parcels, pipeline
from parcelmon.loaders import brp

HERE = Path(__file__).resolve().parent
cfg = config.load_config(HERE.parent / "configs" / "brp_example.yaml")
BBOX = (5.55, 52.50, 5.62, 52.54)  # small area in Flevoland (lon/lat)
CRS = cfg.parcels["crs"]

# 1. Parcels: download, project to metres, shrink inwards
feats = brp.fetch_brp(BBOX)
gdf = brp.to_geodataframe(feats, to_crs=CRS)
gdf_in = parcels.inward_buffer(gdf, cfg.preprocessing["inward_buffer_m"])

# 2. Sentinel-2 L2A for the season, lazily, on a 10 m grid in the parcels' CRS
items = stac.search("sentinel-2-l2a", BBOX, cfg.timeseries["season_start"], cfg.timeseries["season_end"],
                    query={"eo:cloud_cover": {"lt": 70}})
bands = ["B04", "B05", "B08", "B8A", "B11", "SCL"]
ds = stac.load(items, bands, crs=CRS, resolution=10, bbox=BBOX, groupby="solar_day",
               chunks={"x": 1024, "y": 1024})

# Processing baseline of each date (the +1000 DN offset applies from baseline 04.00)
by_day = {pd.Timestamp(it.datetime).normalize(): it.properties.get("s2:processing_baseline", "00.00") for it in items}
baseline = xr.DataArray([by_day.get(pd.Timestamp(t).normalize(), "00.00") for t in ds.time.values],
                        dims="time", coords={"time": ds.time})

# 3. Reflectance, indices and clear-sky mask: labelled, still lazy
refl = exr.s2_l2a_reflectance(ds[["B04", "B05", "B08", "B8A", "B11"]], baseline)
cube = exr.spectral_indices(refl, cfg.timeseries["indices"], sensor="sentinel2")
cube["clear"] = exr.s2_scl_clear(ds["SCL"])

# 4. Parcel labels on the image grid (the CRS is checked) and too-small parcels
labels, id_map = parcels.rasterize_like(gdf_in, "parcel_id", cube)
small = parcels.too_small(labels.values, id_map, cfg.preprocessing["min_pixels"], all_ids=gdf["parcel_id"])

# 5. Parcel table: declared crop + centroid (metres), then the full check
table = gdf.set_index("parcel_id")
table["x"], table["y"] = table.geometry.centroid.x, table.geometry.centroid.y
res = pipeline.run_dataset(cfg, cube, table[[cfg.parcels["crop_column"], "x", "y"]], labels=labels,
                           clear="clear", id_map=id_map, too_small=small)

print(res.report["per_class"])
print(res.summary())
out = res.save(HERE.parent / "outputs" / "brp")
result = gdf.set_index("parcel_id").join(res.verdicts)
result["reasons"] = result["reasons"].map(lambda r: ";".join(r) if isinstance(r, list) else "")
result.to_file(out / "brp_verdicts.gpkg", driver="GPKG")
print(f"Saved {out / 'brp_verdicts.gpkg'}")
