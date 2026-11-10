"""Traffic-light verdict per parcel: agree, disagree or inconclusive.

Follows the logic of CAP area monitoring, where a parcel is confirmed,
flagged, or left for further checks. "Inconclusive" is a valid outcome:
small, cloudy or ambiguous parcels are never forced into a decision.

References
----------
Commission Implementing Regulation (EU) 2022/1173 (Area Monitoring System,
    IACS). CELEX 32022R1173.
Defourny, P. et al. (2019). Near real-time agriculture monitoring at
    national scale at parcel resolution: performance assessment of the
    Sen2-Agri automated system in various cropping systems around the world.
    Remote Sensing of Environment 221, 551-568.
    https://doi.org/10.1016/j.rse.2018.11.007
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["verdicts"]


def verdicts(
    declared: pd.Series,
    proba: pd.DataFrame,
    p_high: float,
    p_low: float,
    blocking_reasons: dict | None = None,
    rule_reasons: dict | None = None,
    downgrade_rules: dict | None = None,
) -> pd.DataFrame:
    """Combine classifier probabilities and rule checks into a verdict.

    Parameters
    ----------
    declared : pandas.Series
        Declared crop group per parcel (index = parcel id). ``None``/NaN means
        the declaration maps to no modelled group.
    proba : pandas.DataFrame
        Out-of-fold class probabilities (index = parcel id). Parcels missing
        here (e.g. too small) are inconclusive.
    p_high, p_low : float
        ``agree`` if P(declared) >= p_high; ``disagree`` if P(declared) <=
        p_low and another group has P >= p_high; otherwise ``inconclusive``.
    blocking_reasons : dict, optional
        ``{parcel_id: [reason, ...]}`` that force ``inconclusive`` (e.g.
        ``too_small``, ``low_valid_dates``).
    rule_reasons : dict, optional
        ``{parcel_id: [reason, ...]}`` from :func:`parcelmon.states.parcel_rule_summary`.
    downgrade_rules : dict, optional
        ``{group: [reason, ...]}``: an ``agree`` for a parcel declared as
        ``group`` becomes ``inconclusive`` if it carries one of these rule
        reasons (e.g. ``{"maize": ["bare_all_season"]}``).

    Returns
    -------
    pandas.DataFrame with ``declared_group``, ``predicted_group``,
    ``p_declared``, ``p_predicted``, ``verdict`` and ``reasons``.
    """
    if not 0 <= p_low < p_high <= 1:
        raise ValueError("need 0 <= p_low < p_high <= 1")
    blocking_reasons = blocking_reasons or {}
    rule_reasons = rule_reasons or {}
    downgrade_rules = downgrade_rules or {}
    rows = []
    for pid, grp in declared.items():
        reasons = list(blocking_reasons.get(pid, []))
        rules = list(rule_reasons.get(pid, []))
        rec = {"parcel_id": pid, "declared_group": grp, "predicted_group": None,
               "p_declared": np.nan, "p_predicted": np.nan}
        if pid in proba.index:
            row = proba.loc[pid]
            rec["predicted_group"] = row.idxmax()
            rec["p_predicted"] = float(row.max())
        if reasons:
            verdict = "inconclusive"
        elif pd.isna(grp):
            verdict, reasons = "inconclusive", ["declared_group_not_modelled"]
        elif pid not in proba.index:
            verdict, reasons = "inconclusive", ["no_prediction"]
        elif grp not in proba.columns:
            verdict, reasons = "inconclusive", ["declared_group_not_modelled"]
        else:
            p_dec = float(proba.at[pid, grp])
            p_other = float(proba.loc[pid].drop(grp).max()) if proba.shape[1] > 1 else 0.0
            rec["p_declared"] = p_dec
            if p_dec >= p_high:
                verdict = "agree"
                conflict = [r for r in rules if r in downgrade_rules.get(grp, [])]
                if conflict:
                    verdict, reasons = "inconclusive", ["rule_conflict", *conflict]
            elif p_dec <= p_low and p_other >= p_high:
                verdict = "disagree"
            else:
                verdict, reasons = "inconclusive", ["low_confidence"]
        rec["verdict"] = verdict
        rec["reasons"] = reasons + [r for r in rules if r not in reasons]
        rows.append(rec)
    return pd.DataFrame(rows).set_index("parcel_id")
