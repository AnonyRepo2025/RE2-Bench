import warnings
import numpy as np
from inspect import signature
from astropy.utils.exceptions import AstropyUserWarning
__all__ = ['FitnessFunc', 'Events', 'RegularEvents', 'PointMeasures', 'bayesian_blocks']

class FitnessFunc:

    def fit(self, t, x=None, sigma=None):
        t, x, sigma = self.validate_input(t, x, sigma)
        if 'a_k' in self._fitness_args:
            ak_raw = np.ones_like(x) / sigma ** 2
        if 'b_k' in self._fitness_args:
            bk_raw = x / sigma ** 2
        if 'c_k' in self._fitness_args:
            ck_raw = x * x / sigma ** 2
        edges = np.concatenate([t[:1], 0.5 * (t[1:] + t[:-1]), t[-1:]])
        block_length = t[-1] - edges
        N = len(t)
        best = np.zeros(N, dtype=float)
        last = np.zeros(N, dtype=int)
        if self.ncp_prior is None:
            ncp_prior = self.compute_ncp_prior(N)
        else:
            ncp_prior = self.ncp_prior
        for R in range(N):
            kwds = {}
            if 'T_k' in self._fitness_args:
                kwds['T_k'] = block_length[:R + 1] - block_length[R + 1]
            if 'N_k' in self._fitness_args:
                kwds['N_k'] = np.cumsum(x[:R + 1][::-1])[::-1]
            if 'a_k' in self._fitness_args:
                kwds['a_k'] = 0.5 * np.cumsum(ak_raw[:R + 1][::-1])[::-1]
            if 'b_k' in self._fitness_args:
                kwds['b_k'] = -np.cumsum(bk_raw[:R + 1][::-1])[::-1]
            if 'c_k' in self._fitness_args:
                kwds['c_k'] = 0.5 * np.cumsum(ck_raw[:R + 1][::-1])[::-1]
            fit_vec = self.fitness(**kwds)
            A_R = fit_vec - ncp_prior
            A_R[1:] += best[:R]
            i_max = np.argmax(A_R)
            last[R] = i_max
            best[R] = A_R[i_max]
        change_points = np.zeros(N, dtype=int)
        i_cp = N
        ind = N
        while True:
            i_cp -= 1
            change_points[i_cp] = ind
            if ind == 0:
                break
            ind = last[ind - 1]
        change_points = change_points[i_cp:]
        return edges[change_points]