from __future__ import annotations
import datetime
import typing
import numpy as np
import pandas as pd
from xarray.coding.cftime_offsets import (
    BaseCFTimeOffset,
    Day,
    MonthEnd,
    QuarterEnd,
    Tick,
    YearEnd,
    cftime_range,
    normalize_date,
    to_offset,
)
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core.types import SideOptions
from xarray.core.types import CFTimeDatetime
import cftime



class CFTimeGrouper:
    def first_items(self, index: CFTimeIndex):
        """Meant to reproduce the results of the following

        grouper = pandas.Grouper(...)
        first_items = pd.Series(np.arange(len(index)),
                                index).groupby(grouper).first()

        with index being a CFTimeIndex instead of a DatetimeIndex.
        """

        datetime_bins, labels = _get_time_bins(
            index, self.freq, self.closed, self.label, self.origin, self.offset
        )
        if self.loffset is not None:
            if not isinstance(
                self.loffset, (str, datetime.timedelta, BaseCFTimeOffset)
            ):
                # BaseCFTimeOffset is not public API so we do not include it in
                # the error message for now.
                raise ValueError(
                    f"`loffset` must be a str or datetime.timedelta object. "
                    f"Got {self.loffset}."
                )

            if isinstance(self.loffset, datetime.timedelta):
                labels = labels + self.loffset
            else:
                labels = labels + to_offset(self.loffset)

        # check binner fits data
        if index[0] < datetime_bins[0]:
            raise ValueError("Value falls before first bin")
        if index[-1] > datetime_bins[-1]:
            raise ValueError("Value falls after last bin")

        integer_bins = np.searchsorted(index, datetime_bins, side=self.closed)[:-1]
        first_items = pd.Series(integer_bins, labels)

        # Mask duplicate values with NaNs, preserving the last values
        non_duplicate = ~first_items.duplicated("last")
        return first_items.where(non_duplicate)