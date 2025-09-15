from numbers import Number
import numpy as np
import pandas as pd
from .algorithms import bootstrap
from .utils import _check_argument
from scipy.stats import gaussian_kde
from .external.kde import gaussian_kde



class Histogram:
    def _define_bin_edges(self, x, weights, bins, binwidth, binrange, discrete):
        """Inner function that takes bin parameters as arguments."""
        if binrange is None:
            start, stop = x.min(), x.max()
        else:
            start, stop = binrange

        if discrete:
            bin_edges = np.arange(start - .5, stop + 1.5)
        elif binwidth is not None:
            step = binwidth
            bin_edges = np.arange(start, stop + step, step)
            # Handle roundoff error (maybe there is a less clumsy way?)
            if bin_edges.max() < stop or len(bin_edges) < 2:
                bin_edges = np.append(bin_edges, bin_edges.max() + step)
        else:
            bin_edges = np.histogram_bin_edges(
                x, bins, binrange, weights,
            )
        return bin_edges