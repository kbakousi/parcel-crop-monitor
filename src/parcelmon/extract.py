"""Per-parcel, per-date statistics from an image time series.

Builds on :func:`eolandscape.zonal.zonal_stats`: for each date and index the
mean, median, standard deviation and valid-pixel fraction of every parcel
are computed from clear pixels only.

Two entry points return the same table:

- :func:`parcel_timeseries` reads the whole stack into memory.
- :func:`parcel_timeseries_tiled` reads one block at a time (the dask chunks,
  or ``tile_size`` tiles), each exactly once. Parcels inside a block are
  summarised at once. The pixels of parcels that cross a block border are
  set aside, put back in raster order, and summarised at the end, so every
  statistic, including the median, comes from all of a parcel's pixels. The
  result is identical to :func:`parcel_timeseries` (tested); peak memory is
  about one block plus the border parcels' pixels, not the whole area.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from eolandscape.zonal import zonal_stats
from scipy import ndimage

__all__ = ["parcel_timeseries", "parcel_timeseries_tiled", "border_parcels"]

_STATS = ["n_total", "n_valid", "valid_fraction", "mean", "median", "std"]
_COLS = ["parcel_id", "date", "index"] + _STATS


def _stack_stats(stack, labels, dates, clear, name):
    """Statistics for every date of one (T, h, w) stack; ``clear`` is (T, h, w) or None."""
    frames = []
    for t, date in enumerate(dates):
        valid = None if clear is None else clear[t]
        st = zonal_stats(stack[t], labels, valid=valid)[_STATS].reset_index()
        st["date"] = date
        st["index"] = name
        frames.append(st)
    return frames


def _finish(frames, id_map):
    if not frames:
        return pd.DataFrame(columns=_COLS)
    out = pd.concat(frames, ignore_index=True)
    out["parcel_id"] = out["label"].map(id_map) if id_map is not None else out["label"]
    return out[_COLS].sort_values(["parcel_id", "index", "date"]).reset_index(drop=True)


def _check(index_stacks, labels_shape, dates):
    for name, stack in index_stacks.items():
        if stack.shape[0] != len(dates):
            raise ValueError(f"{name}: time axis has {stack.shape[0]} layers but {len(dates)} dates")
        if tuple(stack.shape[1:]) != tuple(labels_shape):
            raise ValueError(f"{name}: spatial shape {stack.shape[1:]} differs from the labels {labels_shape}")


def parcel_timeseries(index_stacks: dict, labels, dates, clear=None, id_map: pd.Series | None = None) -> pd.DataFrame:
    """Return a long table of parcel statistics per date and index.

    Parameters
    ----------
    index_stacks : dict
        ``{index_name: array (T, H, W)}``, e.g. ``{"ndvi": ..., "ndmi": ...}``.
    labels : array_like of int, (H, W)
        Parcel label raster (0 = no parcel), e.g. from
        :func:`parcelmon.parcels.rasterize_parcels`.
    dates : array_like, length T
    clear : array_like of bool, (T, H, W), optional
        Clear-sky mask (e.g. from the Sentinel-2 SCL).
    id_map : pandas.Series, optional
        label -> parcel id. Without it the label is used as id.

    Returns
    -------
    pandas.DataFrame with columns ``parcel_id``, ``date``, ``index``,
    ``n_total``, ``n_valid``, ``valid_fraction``, ``mean``, ``median``,
    ``std``.
    """
    dates = pd.to_datetime(np.asarray(dates))
    labels = np.asarray(labels)
    stacks = {k: np.asarray(v, dtype="float64") for k, v in index_stacks.items()}
    _check(stacks, labels.shape, dates)
    clear = None if clear is None else np.asarray(clear, dtype=bool)
    frames = []
    for name, stack in stacks.items():
        frames += _stack_stats(stack, labels, dates, clear, name)
    return _finish(frames, id_map)


def _edges(n: int, sizes) -> np.ndarray:
    edges = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    if edges[-1] != n:
        raise ValueError("block sizes do not add up to the raster size")
    return edges


def border_parcels(labels, row_edges, col_edges) -> np.ndarray:
    """Labels of parcels whose pixels fall in more than one block.

    ``row_edges`` and ``col_edges`` are block boundaries, from 0 to the
    raster size. Returns a boolean lookup array indexed by label.
    """
    labels = np.asarray(labels)
    if labels.size and labels.min() < 0:
        raise ValueError("labels must be >= 0 (0 = no parcel)")
    lookup = np.zeros(int(labels.max(initial=0)) + 1, dtype=bool)
    for lab, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        r = np.searchsorted(row_edges, [sl[0].start, sl[0].stop - 1], side="right")
        c = np.searchsorted(col_edges, [sl[1].start, sl[1].stop - 1], side="right")
        lookup[lab] = r[0] != r[1] or c[0] != c[1]
    return lookup


def _read(slices):
    """Turn block slices into numpy; dask slices are computed together so shared inputs are read once."""
    lazy = [s for s in slices if hasattr(s, "dask")]
    if lazy:
        import dask

        done = iter(dask.compute(*lazy))
        return [np.asarray(next(done)) if hasattr(s, "dask") else np.asarray(s) for s in slices]
    return [np.asarray(s) for s in slices]


def parcel_timeseries_tiled(index_stacks: dict, labels, dates, clear=None, id_map: pd.Series | None = None,
                            tile_size: int | None = None) -> pd.DataFrame:
    """Same table as :func:`parcel_timeseries`, computed block by block.

    Parameters
    ----------
    index_stacks : dict
        ``{index_name: (T, H, W)}`` numpy or dask arrays (anything that can be
        sliced and converted with ``numpy.asarray``). Dask arrays are computed
        one block at a time, so lazily loaded imagery never has to fit in
        memory at once.
    labels : (H, W) int array, in memory
        At 4 bytes per pixel it is far smaller than the image stack.
    dates, clear, id_map
        As in :func:`parcel_timeseries`; ``clear`` may also be a dask array.
    tile_size : int, optional
        Block edge in pixels. By default the spatial chunks of the first dask
        stack are used (each chunk is then computed exactly once), or 1024
        for numpy input.

    Notes
    -----
    Peak memory is about one block (``T x block pixels x 8`` bytes per
    index) plus the pixels of parcels crossing block borders, a small share
    when blocks are much larger than parcels. The output table still grows
    with parcels x dates x indices.
    """
    dates = pd.to_datetime(np.asarray(dates))
    labels = np.asarray(labels)
    _check(index_stacks, labels.shape, dates)
    shape = (len(dates),) + labels.shape
    if clear is not None and tuple(clear.shape) != shape:
        raise ValueError(f"clear has shape {clear.shape}, expected {shape}")
    names = list(index_stacks)
    H, W = labels.shape
    first_lazy = next((s for s in index_stacks.values() if hasattr(s, "chunks") and hasattr(s, "dask")), None)
    if tile_size is None and first_lazy is not None:
        row_edges, col_edges = _edges(H, first_lazy.chunks[1]), _edges(W, first_lazy.chunks[2])
    else:
        size = tile_size or 1024
        if size < 1:
            raise ValueError("tile_size must be >= 1")
        row_edges = np.unique(np.r_[np.arange(0, H, size), H])
        col_edges = np.unique(np.r_[np.arange(0, W, size), W])
    crossing = border_parcels(labels, row_edges, col_edges)

    frames = []
    buf_pos, buf_lab, buf_clear = [], [], []
    buf_val = {n: [] for n in names}
    for r0, r1 in zip(row_edges[:-1], row_edges[1:], strict=True):
        for c0, c1 in zip(col_edges[:-1], col_edges[1:], strict=True):
            lab = labels[r0:r1, c0:c1]
            if not lab.any():
                continue
            slices = [index_stacks[n][:, r0:r1, c0:c1] for n in names]
            if clear is not None:
                slices.append(clear[:, r0:r1, c0:c1])
            arrays = _read(slices)
            clr = arrays.pop().astype(bool) if clear is not None else None
            is_border = crossing[lab]
            inner = np.where(is_border, 0, lab)
            if inner.any():
                for name, arr in zip(names, arrays, strict=True):
                    frames += _stack_stats(arr.astype("float64"), inner, dates, clr, name)
            if is_border.any():
                rr, cc = np.nonzero(is_border)
                buf_pos.append((rr + r0) * W + (cc + c0))
                buf_lab.append(lab[rr, cc])
                for name, arr in zip(names, arrays, strict=True):
                    buf_val[name].append(arr[:, rr, cc].astype("float64"))
                if clr is not None:
                    buf_clear.append(clr[:, rr, cc])
    if buf_pos:
        order = np.argsort(np.concatenate(buf_pos), kind="stable")  # back to raster order
        lab = np.concatenate(buf_lab)[order][None, :]
        clr = np.concatenate(buf_clear, axis=1)[:, order][:, None, :] if clear is not None else None
        for name in names:
            vals = np.concatenate(buf_val[name], axis=1)[:, order][:, None, :]
            frames += _stack_stats(vals, lab, dates, clr, name)
    return _finish(frames, id_map)
