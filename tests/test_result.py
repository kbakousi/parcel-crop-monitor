"""RunResult: typed fields, summary, and a save() that lets the run be reproduced."""

import json
from pathlib import Path

import pandas as pd

from parcelmon import config, pipeline, synthetic

HERE = Path(__file__).parent


def test_run_result_save_and_reload(tmp_path):
    scene = synthetic.make_scene(n_side=8, seed=5, mislabel_rate=0.08)
    cfg = config.load_config(HERE / "synthetic_config.yaml")
    parcels = scene["parcels"].set_index("parcel_id")
    res = pipeline.run(cfg, {"ndvi": scene["ndvi"]}, scene["labels"], scene["dates"], parcels, scene["clear"])
    assert isinstance(res, pipeline.RunResult) and res.config is cfg

    s = res.summary()
    assert s["n_parcels"] == len(parcels) == sum(s["verdicts"].values())
    assert 0 <= s["overall_accuracy"] <= 1

    out = res.save(tmp_path / "run")
    expected = {"verdicts.csv", "rules.csv", "proba.csv", "folds.csv", "observations.csv", "states.csv",
                "series_ndvi.csv", "accuracy_per_class.csv", "confusion.csv", "config.yaml", "summary.json"}
    assert expected == {p.name for p in out.iterdir()}

    back = pd.read_csv(out / "verdicts.csv", index_col=0, keep_default_na=False)
    assert back["verdict"].tolist() == res.verdicts["verdict"].tolist()
    flagged = res.verdicts[res.verdicts["reasons"].map(len) > 0].index[0]
    assert back.loc[flagged, "reasons"] == ";".join(res.verdicts.loc[flagged, "reasons"])

    again = config.load_config(out / "config.yaml")  # the saved config is valid and identical
    assert again == cfg
    summary = json.loads((out / "summary.json").read_text())
    assert summary["parcelmon_version"] and summary["verdicts"] == s["verdicts"]
