from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from seaborn._stats.base import Stat



class PolyFit(Stat):
    def __call__(self, data, groupby, orient, scales):

        return (
            groupby
            .apply(data.dropna(subset=["x", "y"]), self._fit_predict)
        )