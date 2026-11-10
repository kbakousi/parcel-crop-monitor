"""Crop-group classification with out-of-fold predictions and spatial blocks.

Declared crops are noisy labels. Every parcel is therefore predicted by a
Random Forest that did not see that parcel during training (out-of-fold),
so a wrong declaration cannot confirm itself. Folds are built from spatial
blocks, so neighbouring parcels never sit on both sides of a split.

References
----------
Breiman, L. (2001). Random Forests. Machine Learning 45, 5-32.
    https://doi.org/10.1023/A:1010933404324
Belgiu, M., Dragut, L. (2016). Random forest in remote sensing: a review of
    applications and future directions. ISPRS Journal of Photogrammetry and
    Remote Sensing 114, 24-31. https://doi.org/10.1016/j.isprsjprs.2016.01.011
Pelletier, C. et al. (2017). Effect of training class label noise on
    classification performances for land cover mapping with satellite image
    time series. Remote Sensing 9, 173. https://doi.org/10.3390/rs9020173
Roberts, D.R. et al. (2017). Cross-validation strategies for data with
    temporal, spatial, hierarchical, or phylogenetic structure. Ecography 40,
    913-929. https://doi.org/10.1111/ecog.02881
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

__all__ = ["spatial_blocks", "make_model", "out_of_fold_proba", "fit_model"]


def spatial_blocks(x, y, block_size_m: float) -> np.ndarray:
    """Assign each parcel (by centroid ``x``, ``y`` in metres) to a square block."""
    bx = np.floor(np.asarray(x, dtype="float64") / block_size_m).astype("int64")
    by = np.floor(np.asarray(y, dtype="float64") / block_size_m).astype("int64")
    codes, _ = pd.factorize(pd.Series(list(zip(bx, by, strict=True))))
    return codes


def make_model(random_state: int, n_estimators: int = 500, **rf_kwargs):
    """Median imputation + Random Forest (missing features are imputed per column)."""
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1, **rf_kwargs)
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), rf)


def out_of_fold_proba(
    X: pd.DataFrame,
    y: pd.Series,
    groups,
    n_splits: int,
    random_state: int,
    **model_kwargs,
):
    """Class probabilities for every parcel from a model that did not train on it.

    Parameters
    ----------
    X : pandas.DataFrame
        Features (rows = parcels).
    y : pandas.Series
        Declared crop group per parcel (same index as ``X``).
    groups : array_like
        Spatial block per parcel (see :func:`spatial_blocks`).
    n_splits : int
        Number of folds (``GroupKFold``); needs at least as many blocks.

    Returns
    -------
    proba : pandas.DataFrame
        Index = parcels, columns = all classes in ``y``. A class absent from
        a fold's training data gets probability 0 for that fold's parcels.
    fold : pandas.Series
        Fold number per parcel.
    """
    if not X.index.equals(y.index):
        raise ValueError("X and y must share the same index")
    groups = np.asarray(groups)
    if len(np.unique(groups)) < n_splits:
        raise ValueError("fewer spatial blocks than n_splits: use smaller blocks or fewer folds")
    classes = np.array(sorted(y.unique()))
    proba = pd.DataFrame(0.0, index=X.index, columns=classes)
    fold = pd.Series(-1, index=X.index, name="fold")
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        model = make_model(random_state=random_state, **model_kwargs)
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict_proba(X.iloc[te])
        proba.iloc[te, proba.columns.get_indexer(model.classes_)] = p
        fold.iloc[te] = k
    return proba, fold


def fit_model(X: pd.DataFrame, y: pd.Series, random_state: int, **model_kwargs):
    """Fit one model on all parcels (for applying to a new area or year)."""
    model = make_model(random_state=random_state, **model_kwargs)
    return model.fit(X, y)
