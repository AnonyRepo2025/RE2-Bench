import warnings
import numpy as np
from inspect import signature
from astropy.utils.exceptions import AstropyUserWarning

__all__ = ['FitnessFunc', 'Events', 'RegularEvents', 'PointMeasures',
           'bayesian_blocks']

class FitnessFunc:
    def fit(self, t, x=None, sigma=None):
        """Fit the Bayesian Blocks model given the specified fitness function.

        Parameters
        ----------
        t : array_like
            data times (one dimensional, length N)
        x : array_like (optional)
            data values
        sigma : array_like or float (optional)
            data errors

        Returns
        -------
        edges : ndarray
            array containing the (M+1) edges defining the M optimal bins
        """
        t, x, sigma = self.validate_input(t, x, sigma)

        # compute values needed for computation, below
        if 'a_k' in self._fitness_args:
            ak_raw = np.ones_like(x) / sigma ** 2
        if 'b_k' in self._fitness_args:
            bk_raw = x / sigma ** 2
        if 'c_k' in self._fitness_args:
            ck_raw = x * x / sigma ** 2

        # create length-(N + 1) array of cell edges
        edges = np.concatenate([t[:1],
                                0.5 * (t[1:] + t[:-1]),
                                t[-1:]])
        block_length = t[-1] - edges

        # arrays to store the best configuration
        N = len(t)
        best = np.zeros(N, dtype=float)
        last = np.zeros(N, dtype=int)

        # Compute ncp_prior if not defined
        if self.ncp_prior is None:
            ncp_prior = self.compute_ncp_prior(N)
        else:
            ncp_prior = self.ncp_prior

        # ----------------------------------------------------------------
        # Start with first data cell; add one cell at each iteration
        # ----------------------------------------------------------------
        for R in range(N):
            # Compute fit_vec : fitness of putative last block (end at R)
            kwds = {}

            # T_k: width/duration of each block
            if 'T_k' in self._fitness_args:
                kwds['T_k'] = block_length[:R + 1] - block_length[R + 1]

            # N_k: number of elements in each block
            if 'N_k' in self._fitness_args:
                kwds['N_k'] = np.cumsum(x[:R + 1][::-1])[::-1]

            # a_k: eq. 31
            if 'a_k' in self._fitness_args:
                kwds['a_k'] = 0.5 * np.cumsum(ak_raw[:R + 1][::-1])[::-1]

            # b_k: eq. 32
            if 'b_k' in self._fitness_args:
                kwds['b_k'] = - np.cumsum(bk_raw[:R + 1][::-1])[::-1]

            # c_k: eq. 33
            if 'c_k' in self._fitness_args:
                kwds['c_k'] = 0.5 * np.cumsum(ck_raw[:R + 1][::-1])[::-1]

            # evaluate fitness function
            fit_vec = self.fitness(**kwds)

            A_R = fit_vec - ncp_prior
            A_R[1:] += best[:R]

            i_max = np.argmax(A_R)
            last[R] = i_max
            best[R] = A_R[i_max]

        # ----------------------------------------------------------------
        # Now find changepoints by iteratively peeling off the last block
        # ----------------------------------------------------------------
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