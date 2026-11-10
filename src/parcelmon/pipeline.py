"""End-to-end run: parcel statistics -> time series -> rules + classifier -> verdicts."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from . import classify, extract, states, timeseries, validate, verdict
from .config import Config, map_crop_groups

__all__ = ["RunResult", "run", "run_dataset"]


def _joined(col: pd.Series) -> pd.Series:
    return col.map(lambda r: ";".join(r) if isinstance(r, (list, tuple)) else ("" if r is None else str(r)))


@dataclass
class RunResult:
    """Everything one run produces.

    Attributes
    ----------
    observations : DataFrame
        Long table of parcel statistics per date and index, after filtering.
    series : dict of DataFrame
        Regular (and optionally smoothed) series per index; rows = parcels.
    states : DataFrame
        bare / low_cover / vegetated / no_data per parcel and grid date.
    rules : DataFrame
        Rule summary per parcel, including ``rule_reasons``.
    proba : DataFrame
        Out-of-fold class probabilities of the training parcels.
    folds : Series
        Cross-validation fold of each training parcel.
    report : dict
        ``overall_accuracy``, ``macro_f1``, ``per_class`` and ``confusion``.
    verdicts : DataFrame
        Traffic-light verdict and reasons per parcel.
    config : Config
        The configuration the run used.
    """

    observations: pd.DataFrame
    series: dict
    states: pd.DataFrame
    rules: pd.DataFrame
    proba: pd.DataFrame
    folds: pd.Series
    report: dict
    verdicts: pd.DataFrame
    config: Config

    def summary(self) -> dict:
        """Headline numbers: parcels, verdict counts and out-of-fold agreement."""
        return {
            "n_parcels": int(len(self.verdicts)),
            "verdicts": {k: int(v) for k, v in self.verdicts["verdict"].value_counts().sort_index().items()},
            "overall_accuracy": float(self.report["overall_accuracy"]),
            "macro_f1": float(self.report["macro_f1"]),
        }

    def save(self, directory) -> Path:
        """Write every table, the config used and a summary to ``directory``.

        Files: ``verdicts.csv``, ``rules.csv``, ``proba.csv``, ``folds.csv``,
        ``observations.csv``, ``states.csv``, ``series_<index>.csv``,
        ``accuracy_per_class.csv``, ``confusion.csv``, ``config.yaml`` and
        ``summary.json``. List columns (reasons) are written as
        ``;``-separated text. Returns the directory.
        """
        from . import __version__

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.verdicts.assign(reasons=_joined(self.verdicts["reasons"])).to_csv(out / "verdicts.csv")
        self.rules.assign(rule_reasons=_joined(self.rules["rule_reasons"])).to_csv(out / "rules.csv")
        self.proba.to_csv(out / "proba.csv")
        self.folds.rename("fold").to_csv(out / "folds.csv")
        self.observations.to_csv(out / "observations.csv", index=False)
        self.states.to_csv(out / "states.csv")
        for name, wide in self.series.items():
            wide.to_csv(out / f"series_{name}.csv")
        self.report["per_class"].to_csv(out / "accuracy_per_class.csv")
        self.report["confusion"].to_csv(out / "confusion.csv")
        cfg = {k: v for k, v in dataclasses.asdict(self.config).items() if not k.startswith("_")}
        (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        summary = {"parcelmon_version": __version__, **self.summary()}
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return out


def run_dataset(
    cfg: Config,
    ds,
    parcels: pd.DataFrame,
    labels="labels",
    clear=None,
    id_map: pd.Series | None = None,
    too_small: pd.Series | None = None,
    dims: tuple[str, str, str] = ("time", "y", "x"),
    tiled: bool = True,
    tile_size: int | None = None,
) -> RunResult:
    """Run the full check on an ``xarray.Dataset`` of index time series.

    The labelled entry point: dates come from the Dataset's time coordinate,
    and the index layers, the parcel label raster and the clear-sky mask must
    sit on exactly the same grid. A raster on another grid (shifted, other
    resolution or extent) raises instead of being matched pixel by pixel.

    Parameters
    ----------
    cfg : Config
    ds : xarray.Dataset
        One ``(time, y, x)`` variable per index in
        ``cfg.timeseries["indices"]``, e.g. from
        :func:`eolandscape.xr.spectral_indices`. Dask-backed data stays lazy
        and is read one chunk at a time (see ``tiled``).
    parcels : pandas.DataFrame
        As in :func:`run`.
    labels : str or xarray.DataArray
        Parcel label raster ``(y, x)``, or the name of that variable in
        ``ds``; e.g. from :func:`parcelmon.parcels.rasterize_like`.
    clear : str or xarray.DataArray, optional
        Clear-sky mask ``(time, y, x)``, or the name of that variable in ``ds``.
    id_map, too_small : optional
        As in :func:`run`.
    dims : (time, y, x)
        Dimension names.
    tiled, tile_size :
        As in :func:`run`. By default parcel statistics are computed block by
        block on the dask chunks. ``tiled=False`` reads everything at once.
        Both give identical results.

    Returns
    -------
    RunResult
    """
    import xarray

    tdim, ydim, xdim = dims
    names = list(cfg.timeseries["indices"])
    missing = [n for n in names if n not in ds]
    if missing:
        raise ValueError(f"the dataset has no variable(s) {missing} required by the config")
    lab = ds[labels] if isinstance(labels, str) else labels
    clr = ds[clear] if isinstance(clear, str) else clear
    parts = [ds[names], lab] + ([clr] if clr is not None else [])
    for part in parts:
        if ydim not in part.coords or xdim not in part.coords:
            raise ValueError(f"every input needs {ydim!r} and {xdim!r} coordinates to check the grid")
    try:
        parts = xarray.align(*parts, join="exact")
    except ValueError as exc:
        raise ValueError("index layers, parcel labels and clear mask are not on the same grid") from exc
    stacks_ds, lab = parts[0], parts[1]
    stacks = {n: stacks_ds[n].transpose(tdim, ydim, xdim).data for n in names}  # numpy or dask, not computed
    clear_arr = parts[2].transpose(tdim, ydim, xdim).data if clr is not None else None
    return run(cfg, stacks, lab.transpose(ydim, xdim).values, ds[tdim].values, parcels, clear_arr,
               id_map=id_map, too_small=too_small, tiled=tiled, tile_size=tile_size)


def run(
    cfg: Config,
    index_stacks: dict,
    labels,
    dates,
    parcels: pd.DataFrame,
    clear=None,
    id_map: pd.Series | None = None,
    too_small: pd.Series | None = None,
    tiled: bool = False,
    tile_size: int | None = None,
) -> RunResult:
    """Run the full check for one season.

    Parameters
    ----------
    cfg : Config
    index_stacks : dict
        ``{index_name: (T, H, W)}``; must contain every index in
        ``cfg.timeseries["indices"]``.
    labels : (H, W) int raster of parcels (0 = none)
    dates : length-T dates
    parcels : pandas.DataFrame
        Index = parcel id; columns: the declared crop column named in the
        config, plus ``x`` and ``y`` centroid coordinates in metres.
    clear : (T, H, W) bool, optional
    id_map : label -> parcel id, optional
    too_small : bool Series by parcel id, optional
        From :func:`parcelmon.parcels.too_small`.
    tiled : bool
        Compute parcel statistics block by block
        (:func:`parcelmon.extract.parcel_timeseries_tiled`); needed for dask
        arrays larger than memory. The result is the same either way.
    tile_size : int, optional
        Block size for ``tiled``; default: the dask chunks, or 1024.

    Returns
    -------
    RunResult
        Tables, accuracy report and verdicts; ``.save(directory)`` writes them.
    """
    ts, st, ev, cl, vd = cfg.timeseries, cfg.states, cfg.events, cfg.classifier, cfg.verdict
    crop_col = cfg.parcels["crop_column"]

    stacks = {k: index_stacks[k] for k in ts["indices"]}
    if not tiled:
        obs = extract.parcel_timeseries(stacks, labels, dates, clear, id_map)
    else:
        obs = extract.parcel_timeseries_tiled(stacks, labels, dates, clear, id_map, tile_size=tile_size)
    obs = timeseries.filter_observations(obs, ts["min_valid_fraction"], ts.get("min_valid_pixels", 1))

    series = {}
    for name in ts["indices"]:
        wide = timeseries.regularize(obs, name, ts["season_start"], ts["season_end"], ts["step_days"],
                                     ts["max_gap_days"])
        if ts.get("smoothing"):
            wide = timeseries.smooth(wide, ts["smoothing"]["window"], ts["smoothing"]["polyorder"])
        series[name] = wide.reindex(parcels.index)

    primary = series[ts["primary_index"]]
    state_tab = states.classify_states(primary, st["bare_max"], st["vegetated_min"])
    rules = states.parcel_rule_summary(state_tab, primary, ev["min_drop"], ev["max_drop_days"],
                                       st["bare_season_fraction"], ev.get("min_peak"))

    blocking: dict = {}
    n_dates = timeseries.n_valid_dates(obs, ts["primary_index"]).reindex(parcels.index, fill_value=0)
    for pid in parcels.index:
        reasons = []
        if too_small is not None and bool(too_small.get(pid, False)):
            reasons.append("too_small")
        if n_dates.get(pid, 0) < ts["min_valid_dates"]:
            reasons.append("low_valid_dates")
        if reasons:
            blocking[pid] = reasons

    groups = map_crop_groups(parcels[crop_col], cfg.crop_groups)
    X = timeseries.feature_matrix(series)
    train = groups.notna() & ~groups.index.isin(list(blocking))
    blocks = classify.spatial_blocks(parcels.loc[train, "x"], parcels.loc[train, "y"], cl["block_size_m"])
    model_kwargs = {k: v for k, v in cl.items() if k in ("n_estimators", "max_features", "min_samples_leaf")}
    proba, folds = classify.out_of_fold_proba(X.loc[train], groups[train], blocks, cl["n_splits"],
                                              cl["random_state"], **model_kwargs)
    report = validate.accuracy_report(groups[train], proba.idxmax(axis=1))
    verdicts = verdict.verdicts(
        groups,
        proba,
        vd["p_high"],
        vd["p_low"],
        blocking_reasons=blocking,
        rule_reasons=rules["rule_reasons"].to_dict(),
        downgrade_rules=vd.get("downgrade_rules"),
    )
    return RunResult(observations=obs, series=series, states=state_tab, rules=rules, proba=proba, folds=folds,
                     report=report, verdicts=verdicts, config=cfg)
