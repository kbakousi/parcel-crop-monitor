"""Memory and time of parcel statistics: whole image vs tiled (window by window).

The NDVI stack is generated lazily with dask, block by block, so the tiled
run never holds the full stack. The whole-image run needs it in memory.

Run:  python benchmarks/bench_extract.py [--sizes 1000 3000] [--dates 25] [--tile 1024]
Requires dask (``pip install -e ".[dev]"``).
"""

from __future__ import annotations

import argparse
import time
import tracemalloc

import dask
import dask.array as da
import numpy as np
import pandas as pd

from parcelmon import extract, synthetic


def scene(n_px: int, n_dates: int, parcel_px: int = 10, chunk: int = 1024, seed: int = 0):
    """Square scene of ``parcel_px`` parcels with 1-pixel margins; lazy NDVI from the synthetic phenologies."""
    step = parcel_px + 1
    rr, cc = np.indices((n_px, n_px))
    inside = (rr % step < parcel_px) & (cc % step < parcel_px)
    n_cols = n_px // step + 1
    labels = np.where(inside, (rr // step) * n_cols + cc // step + 1, 0).astype("int32")
    rng = np.random.default_rng(seed)
    groups = list(synthetic.PHENOLOGY)
    group_of = np.concatenate([[0], rng.integers(1, len(groups) + 1, labels.max())])  # 0 = background
    dates = pd.date_range("2025-03-01", periods=n_dates, freq="10D")
    doy = dates.day_of_year.to_numpy().astype(float)
    curves = np.vstack([np.full(n_dates, 0.1)] + [synthetic.PHENOLOGY[g](doy) for g in groups])  # (G+1, T)

    def block(lab, block_info=None):
        loc = block_info[0]["chunk-location"]
        noise = np.random.default_rng([seed, *loc]).normal(0, 0.03, (n_dates,) + lab.shape)
        return np.moveaxis(curves[group_of[lab]], -1, 0) + noise

    lab_da = da.from_array(labels, chunks=chunk)
    ndvi = da.map_blocks(block, lab_da, new_axis=0, chunks=((n_dates,),) + lab_da.chunks, dtype="float64")
    return labels, ndvi, dates


def measure(func):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = func()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, elapsed, peak / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 3000])
    ap.add_argument("--dates", type=int, default=25)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--whole-max", type=int, default=1500, help="largest size also run as a whole image")
    args = ap.parse_args()
    dask.config.set(scheduler="threads")
    rows = []
    for n in args.sizes:
        labels, ndvi, dates = scene(n, args.dates, chunk=args.tile)
        stack_mib = ndvi.nbytes / 2**20
        tiled, t_tiled, m_tiled = measure(
            lambda: extract.parcel_timeseries_tiled({"ndvi": ndvi}, labels, dates))  # noqa: B023
        row = {"pixels": f"{n} x {n}", "area_km2": n * n / 10_000, "parcels": int(labels.max()),
               "dates": args.dates, "stack_MiB": round(stack_mib), "tiled_s": round(t_tiled, 1),
               "tiled_peak_MiB": round(m_tiled), "table_MiB": round(tiled.memory_usage(deep=True).sum() / 2**20)}
        if n <= args.whole_max:
            whole, t_whole, m_whole = measure(
                lambda: extract.parcel_timeseries({"ndvi": ndvi.compute()}, labels, dates))  # noqa: B023
            pd.testing.assert_frame_equal(tiled, whole)
            row.update(whole_s=round(t_whole, 1), whole_peak_MiB=round(m_whole), identical=True)
        rows.append(row)
        print(row, flush=True)
    print()
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
