# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org/).

## [0.1.0] - 2026-11-10

### Added
- Parcel preparation (inward buffer, centre-based rasterising, too-small flag, `.cpg` encoding).
- Per-parcel per-date statistics from clear pixels (`extract`).
- Time-series filtering, gap-aware regularisation, Savitzky-Golay smoothing, feature matrix.
- Rule-based states (bare / low cover / vegetated), drop events, Youden threshold calibration.
- Out-of-fold Random Forest with spatial-block folds; accuracy report.
- Traffic-light verdicts with reason codes and rule-based downgrades.
- BRP (PDOK OGC API) loader; synthetic scene generator; example config.
- Labelled entry point `pipeline.run_dataset` (xarray, exact grid alignment); `parcels.rasterize_like`
  (grid from image coordinates, CRS check); `synthetic.to_dataset`.
- Block-by-block parcel statistics `extract.parcel_timeseries_tiled` (each dask chunk computed once; parcels
  crossing chunk borders handled exactly), used by `run_dataset`; `pipeline.run(tiled=True)`.
- `benchmarks/bench_extract.py`: memory and time, whole image vs block by block.
- Config validation of types, ranges and cross-field rules, all problems reported together; warnings for
  unknown keys.
- `pipeline.RunResult` (typed result) with `summary()` and `save()`, which writes all tables, the config used
  and `summary.json`.
- Tutorial notebook `tutorials/01_parcel_check_walkthrough.ipynb`, executed by the test suite.
