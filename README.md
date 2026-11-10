# parcel-crop-monitor

[![Tests](https://github.com/kbakousi/parcel-crop-monitor/actions/workflows/ci.yml/badge.svg)](https://github.com/kbakousi/parcel-crop-monitor/actions/workflows/ci.yml)

Parcel-level crop and land-use checks with Sentinel-2 time series. Given agricultural parcels with a declared crop (e.g. an IACS / LPIS declaration), the package:

1. extracts **per-parcel, per-date statistics from clear pixels only**, after an inward buffer that removes mixed edge pixels;
2. builds **gap-aware regular time series** (NDVI, NDMI, NDRE), optionally smoothed;
3. derives **rule-based land-use states** (bare / low cover / vegetated) and **drop events** (harvest, mowing);
4. predicts each parcel's crop group with a **Random Forest that never trained on that parcel** (out-of-fold, with spatial-block cross-validation);
5. issues a **traffic-light verdict**: 🟢 agree / 🔴 disagree / 🟡 inconclusive, with reason codes.

**Methods.** Each module is an independent implementation of the methods described in the publications cited in its docstring (see [References](#references)). Thresholds that change results have no hidden defaults: they are set in a YAML config.

## Install

```bash
pip install -e .              # core (numpy, pandas, scipy, scikit-learn, pyyaml, xarray, eolandscape)
pip install -e ".[geo,stac]"  # + geopandas/rasterio and STAC loaders for real data
```

`eolandscape` ([eo-landscape-indicators](https://github.com/kbakousi/eo-landscape-indicators)) provides the spectral indices, Sentinel-2 scaling, SCL cloud masking and zonal statistics.

## Quick start (offline)

```bash
python examples/demo_synthetic.py
```

The demo builds a synthetic scene of 144 parcels with four crop phenologies, partial cloud cover and 6% wrong declarations, then runs the whole check. Example output:

```
Agreement with declarations (out-of-fold): 0.910, macro F1 0.909
Mislabelled parcels: 13; flagged (disagree or inconclusive): 13
```

### Tutorial

[`tutorials/01_parcel_check_walkthrough.ipynb`](tutorials/01_parcel_check_walkthrough.ipynb) walks through the same synthetic scene step by step:

- config validation;
- out-of-fold accuracy;
- parcel states and mowing events;
- calibrating a threshold with Youden's J;
- verdicts compared with the known truth;
- saving a reproducible run;
- the chunked run.

### Results and reproducibility

`run` and `run_dataset` return a `RunResult`, with typed fields such as `verdicts`, `report`, `series` and `rules`.
`res.save("outputs/run1")` writes:

- every table as CSV;
- the **configuration actually used** (`config.yaml`), which loads back as an identical, valid config;
- `summary.json`, with the version, verdict counts and agreement.

## Real data: Dutch BRP + Sentinel-2

```bash
python examples/demo_brp_sentinel2.py
```

- **Parcels:** downloaded from the PDOK OGC API ([BRP Gewaspercelen](https://api.pdok.nl/rvo/gewaspercelen/ogc/v1), RVO, Public Domain Mark 1.0) for a small Flevoland bounding box.
- **Imagery:** Sentinel-2 L2A from Microsoft Planetary Computer.
- **No account needed.** Data are fetched at run time and not stored in this repository.
- **Configuration:** [`configs/brp_example.yaml`](configs/brp_example.yaml).

The images stay labelled `xarray` objects from loading to the parcel statistics, and dask keeps them lazy until then:

```python
refl = exr.s2_l2a_reflectance(ds[bands], baseline)            # one processing baseline per date
cube = exr.spectral_indices(refl, ["ndvi", "ndmi", "ndre"])
cube["clear"] = exr.s2_scl_clear(ds["SCL"])
labels, id_map = parcels.rasterize_like(gdf_in, "parcel_id", cube)   # grid read from the images, CRS checked
res = pipeline.run_dataset(cfg, cube, table, labels=labels, clear="clear", id_map=id_map, too_small=small)
```

`rasterize_like` reads the transform and shape from the image coordinates and refuses parcels in another CRS. `run_dataset` refuses index layers, labels or masks that are not on exactly the same grid. A raster shifted by one pixel raises an error instead of being matched to the wrong parcels.

## Scaling

`run_dataset` computes parcel statistics block by block, on the dask chunks (`extract.parcel_timeseries_tiled`):

- **Each chunk is computed once.** Parcels inside a chunk are summarised immediately.
- **Border parcels are handled exactly.** Pixels of parcels that cross a chunk border are set aside, put back in raster order, and summarised at the end.
- **The results are the same as reading the whole image**, median included (tested, down to 1-pixel blocks and parcels split into separate pieces).

Benchmark: [`benchmarks/bench_extract.py`](benchmarks/bench_extract.py), with synthetic 10 × 10-pixel parcels, 25 dates, one index, 1024-pixel chunks and peak memory from `tracemalloc`.

| Area (10 m pixels) | Parcels | Image stack | Whole image: time / peak memory | Block by block: time / peak memory |
|---|---|---|---|---|
| 100 km² (1000²) | 8,281 | 191 MiB | 2.7 s / 572 MiB | 3.3 s / 572 MiB (one chunk) |
| 400 km² (2000²) | 33,124 | 763 MiB | 13.9 s / 2,289 MiB | 12.7 s / 804 MiB |
| 1,600 km² (4000²) | 132,496 | 3,052 MiB | not run (about 9 GiB) | 51.0 s / 1,009 MiB |

The peak memory of the block-by-block run still grows slowly with area, because the output table grows: 357 MiB at 1,600 km² for one index. For hundreds of thousands of parcels and several indices, run sub-regions or write the table out in parts. Dask distributed clusters are not set up here.

## How it works

| Step | Module | Method |
|---|---|---|
| Inward buffer, centre-based rasterising, too-small parcels, `.cpg` encoding | `parcels` | Sitokonstantinou et al. 2018 |
| Labelled entry point: index layers, labels and mask must share one grid | `pipeline.run_dataset` | — |
| Clear-pixel parcel statistics per date | `extract` (uses `eolandscape.zonal`) | — |
| Drop low-coverage dates; regular grid without bridging long gaps or extrapolating; Savitzky–Golay | `timeseries` | Inglada et al. 2015; Savitzky & Golay 1964 |
| States from two thresholds; threshold calibration by Youden's J; drop events | `states` | Kanjir et al. 2018; Youden 1950 |
| Random Forest, median imputation, out-of-fold probabilities, spatial-block folds | `classify` | Breiman 2001; Belgiu & Drăguţ 2016; Pelletier et al. 2017; Roberts et al. 2017 |
| Confusion matrix, overall accuracy, per-class precision/recall/F1 | `validate` | Olofsson et al. 2014 |
| Traffic-light verdict with reason codes; rule-based downgrade of "agree" | `verdict` | Defourny et al. 2019; Reg. (EU) 2022/1173 |

### Verdict rules

| Verdict | Rule (`p_high`, `p_low` set in the config) |
|---|---|
| 🟢 `agree` | P(declared group) ≥ `p_high`, and no conflicting rule reason |
| 🔴 `disagree` | P(declared group) ≤ `p_low` and another group has P ≥ `p_high` |
| 🟡 `inconclusive` | anything else, plus `too_small`, `low_valid_dates`, `declared_group_not_modelled`, `rule_conflict` |

### Limits

- **Declared crops are not field truth.** The accuracy report measures agreement with declarations, and declarations contain errors. That is why predictions are out-of-fold.
- **Thresholds need calibration.** State thresholds (e.g. NDVI 0.25 / 0.5 in the example) depend on region and sensor. Calibrate them on reference observations with `states.youden_threshold`.
- **Drop events are generic.** A drop is labelled harvest or mowing only through the crop context.
- **Folds have no shuffling.** `GroupKFold` is deterministic. Block size and fold count are config choices and change the estimate.
- **Memory.** Images are processed chunk by chunk (see [Scaling](#scaling)). The label raster (4 bytes per pixel) and the output table stay in memory.

## Using your own parcels

The same code runs on any parcel layer with an id and a declared-crop column. Map crop codes to groups in the config, either with `codes` (exact) or `patterns` (regular expressions).

The config is checked when it loads:

- types and ranges (probabilities in [0, 1], integer counts, dates);
- consistency: `p_low < p_high`, `bare_max < vegetated_min`, season start before end, an odd smoothing window, known rule names, valid regular expressions.

All problems are reported together. Unknown keys, which are usually typos, give a warning. Keep restricted parcel data and its config outside the repository; `.gitignore` excludes `data/`, `outputs/`, `private_configs/` and vector formats.

## Tests

```bash
pytest
```

The tests run offline. They cover:
- **config:** required parameters, types, ranges, cross-field rules, typo warnings;
- **results:** `RunResult.save` writes every table, and the saved config reloads identically;
- **parcel statistics:** clear-pixel extraction;
- **time series:** gap handling and no extrapolation;
- **states and events:** Youden calibration, mowing vs senescence;
- **classifier:** out-of-fold integrity (a spatial block is never split across folds);
- **verdicts:** every verdict branch;
- **pipeline:** a full end-to-end run on a synthetic scene;
- **labelled path:** `run_dataset` gives exactly the array result (also on dask-chunked data), refuses shifted grids, and `rasterize_like` matches the array rasteriser and checks the CRS;
- **block-by-block statistics:** identical to the whole-image table for block sizes from 1 to 1000 pixels, irregular parcels, parcels in two pieces, NaNs and clouds, and dask input;
- **tutorial:** the notebook in `tutorials/` is executed as a test.

## References

- Belgiu, M., Drăguţ, L. (2016). *ISPRS J. Photogramm. Remote Sens.* 114, 24–31. https://doi.org/10.1016/j.isprsjprs.2016.01.011
- Breiman, L. (2001). Random Forests. *Machine Learning* 45, 5–32. https://doi.org/10.1023/A:1010933404324
- Defourny, P. et al. (2019). *Remote Sensing of Environment* 221, 551–568. https://doi.org/10.1016/j.rse.2018.11.007
- Inglada, J. et al. (2015). *Remote Sensing* 7, 12356–12379. https://doi.org/10.3390/rs70912356
- Kanjir, U., Đurić, N., Veljanovski, T. (2018). *ISPRS Int. J. Geo-Information* 7, 405. https://doi.org/10.3390/ijgi7100405
- Olofsson, P. et al. (2014). *Remote Sensing of Environment* 148, 42–57. https://doi.org/10.1016/j.rse.2014.02.015
- Pelletier, C. et al. (2017). *Remote Sensing* 9, 173. https://doi.org/10.3390/rs9020173
- Roberts, D.R. et al. (2017). *Ecography* 40, 913–929. https://doi.org/10.1111/ecog.02881
- Savitzky, A., Golay, M.J.E. (1964). *Analytical Chemistry* 36, 1627–1639. https://doi.org/10.1021/ac60214a047
- Sitokonstantinou, V. et al. (2018). *Remote Sensing* 10, 911. https://doi.org/10.3390/rs10060911
- Youden, W.J. (1950). *Cancer* 3, 32–35. https://doi.org/10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3
- Commission Implementing Regulation (EU) 2022/1173 (Area Monitoring System). CELEX 32022R1173.

## Licence and citation

Apache-2.0 (see `LICENSE` and `NOTICE`). Cite via `CITATION.cff`.
