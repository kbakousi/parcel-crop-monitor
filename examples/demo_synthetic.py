"""Offline demo: the full check on a synthetic scene (no network, no geo libraries).

The scene is an ``xarray.Dataset`` (ndvi, clear, labels on one grid), the
same shape of input the real-data demo builds from Sentinel-2.

Run:  python examples/demo_synthetic.py
Writes outputs/synthetic_verdicts.csv and prints the accuracy report.
"""

from pathlib import Path

from parcelmon import config, pipeline, synthetic

HERE = Path(__file__).resolve().parent
cfg = config.load_config(HERE.parent / "tests" / "synthetic_config.yaml")

scene = synthetic.make_scene(n_side=12, seed=7, mislabel_rate=0.06)
ds = synthetic.to_dataset(scene)
parcels = scene["parcels"].set_index("parcel_id")
res = pipeline.run_dataset(cfg, ds, parcels, labels="labels", clear="clear")

report = res.report
print(f"Agreement with declarations (out-of-fold): {report['overall_accuracy']:.3f}, macro F1 {report['macro_f1']:.3f}")
print(report["confusion"], "\n")

v = res.verdicts.join(parcels[["true_group"]])
print(v["verdict"].value_counts(), "\n")
wrong = v[v["declared_group"] != v["true_group"]]
print(f"Mislabelled parcels: {len(wrong)}; flagged (disagree or inconclusive): {(wrong['verdict'] != 'agree').sum()}")

out = res.save(HERE.parent / "outputs" / "synthetic")
print(f"Saved all tables, the config used and summary.json to {out}")
