"""parcelmon: parcel-level crop and land-use checks with Sentinel-2 time series.

Each module is an independent implementation of the methods described in the
publications cited in its docstring.
"""

from . import classify, config, extract, pipeline, states, synthetic, timeseries, validate, verdict

__all__ = ["classify", "config", "extract", "pipeline", "states", "synthetic", "timeseries", "validate", "verdict"]
__version__ = "0.1.0"
