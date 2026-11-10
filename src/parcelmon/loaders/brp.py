"""Download Dutch crop parcels (BRP Gewaspercelen) from the PDOK OGC API.

Dataset: Basisregistratie Gewaspercelen (BRP), Rijksdienst voor Ondernemend
Nederland, published by PDOK. Licence stated by the API: Public Domain
Mark 1.0. One dataset per year, reference date 15 May; the API serves the
most recent year. Properties: ``category``, ``gewas`` (crop name),
``gewascode``, ``jaar`` (year), ``status``.

Needs network access; not covered by the offline tests.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

__all__ = ["BRP_ITEMS_URL", "fetch_brp", "to_geodataframe"]

BRP_ITEMS_URL = "https://api.pdok.nl/rvo/gewaspercelen/ogc/v1/collections/brpgewas/items"


def fetch_brp(bbox, categories=("Bouwland", "Grasland", "Braakland"), page_size: int = 1000,
              max_features: int | None = None, timeout: int = 60) -> list[dict]:
    """Return GeoJSON features (WGS84 lon/lat) inside ``bbox``.

    Parameters
    ----------
    bbox : (min_lon, min_lat, max_lon, max_lat)
    categories : iterable of str or None
        Keep only these BRP categories. The default drops landscape elements
        (e.g. ditches) and nature areas. ``None`` keeps everything.
    page_size : int
        Features per request (the API pages with ``next`` links).
    max_features : int, optional
        Stop after this many kept features.
    """
    query = urllib.parse.urlencode({"f": "json", "limit": page_size, "bbox": ",".join(str(v) for v in bbox)})
    url = f"{BRP_ITEMS_URL}?{query}"
    keep = None if categories is None else set(categories)
    out: list[dict] = []
    while url:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            page = json.load(resp)
        for feat in page.get("features", []):
            if keep is None or feat["properties"].get("category") in keep:
                out.append(feat)
                if max_features is not None and len(out) >= max_features:
                    return out
        url = next((link["href"] for link in page.get("links", []) if link.get("rel") == "next"), None)
    return out


def to_geodataframe(features: list[dict], to_crs: str | None = "EPSG:28992"):
    """Convert features to a GeoDataFrame, by default in Dutch RD New (metres)."""
    import geopandas as gpd

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    gdf["parcel_id"] = [f.get("id") for f in features]
    return gdf.to_crs(to_crs) if to_crs else gdf
