from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Hashable,
    Iterator,
    Mapping,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)
import numpy as np
import pandas as pd
from . import formatting, indexing
from .indexes import Indexes
from .merge import merge_coordinates_without_align, merge_coords
from .utils import Frozen, ReprObject, either_dict_or_kwargs
from .variable import Variable
from .dataarray import DataArray
from .dataset import Dataset
from .dataarray import DataArray
from .dataset import Dataset
from .dataset import calculate_dimensions
from .dataset import calculate_dimensions
from .dataset import Dataset

_THIS_ARRAY = ReprObject("<this-array>")

class Coordinates(Mapping[(Hashable, 'DataArray')]
):
    __slots__ = ()
    def to_index(self, ordered_dims: Sequence[Hashable] = None) -> pd.Index:
        """Convert all index coordinates into a :py:class:`pandas.Index`.

        Parameters
        ----------
        ordered_dims : sequence of hashable, optional
            Possibly reordered version of this object's dimensions indicating
            the order in which dimensions should appear on the result.

        Returns
        -------
        pandas.Index
            Index subclass corresponding to the outer-product of all dimension
            coordinates. This will be a MultiIndex if this object is has more
            than more dimension.
        """
        if ordered_dims is None:
            ordered_dims = list(self.dims)
        elif set(ordered_dims) != set(self.dims):
            raise ValueError(
                "ordered_dims must match dims, but does not: "
                "{} vs {}".format(ordered_dims, self.dims)
            )

        if len(ordered_dims) == 0:
            raise ValueError("no valid index for a 0-dimensional object")
        elif len(ordered_dims) == 1:
            (dim,) = ordered_dims
            return self._data.get_index(dim)  # type: ignore
        else:
            indexes = [self._data.get_index(k) for k in ordered_dims]  # type: ignore

            # compute the sizes of the repeat and tile for the cartesian product
            # (taken from pandas.core.reshape.util)
            index_lengths = np.fromiter(
                (len(index) for index in indexes), dtype=np.intp
            )
            cumprod_lengths = np.cumproduct(index_lengths)

            if cumprod_lengths[-1] != 0:
                # sizes of the repeats
                repeat_counts = cumprod_lengths[-1] / cumprod_lengths
            else:
                # if any factor is empty, the cartesian product is empty
                repeat_counts = np.zeros_like(cumprod_lengths)

            # sizes of the tiles
            tile_counts = np.roll(cumprod_lengths, 1)
            tile_counts[0] = 1

            # loop over the indexes
            # for each MultiIndex or Index compute the cartesian product of the codes

            code_list = []
            level_list = []
            names = []

            for i, index in enumerate(indexes):
                if isinstance(index, pd.MultiIndex):
                    codes, levels = index.codes, index.levels
                else:
                    code, level = pd.factorize(index)
                    codes = [code]
                    levels = [level]

                # compute the cartesian product
                code_list += [
                    np.tile(np.repeat(code, repeat_counts[i]), tile_counts[i])
                    for code in codes
                ]
                level_list += levels
                names += index.names

            return pd.MultiIndex(level_list, code_list, names=names)