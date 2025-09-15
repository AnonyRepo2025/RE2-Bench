from .index import SlicedIndex, TableIndices, TableLoc, TableILoc, TableLocIndices
import sys
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
import warnings
from copy import deepcopy
import types
import itertools
import weakref
import numpy as np
from numpy import ma
from astropy import log
from astropy.units import Quantity, QuantityInfo
from astropy.utils import isiterable, ShapedLikeNDArray
from astropy.utils.console import color_print
from astropy.utils.exceptions import AstropyUserWarning
from astropy.utils.masked import Masked
from astropy.utils.metadata import MetaData, MetaAttribute
from astropy.utils.data_info import BaseColumnInfo, MixinInfo, DataInfo
from astropy.utils.decorators import format_doc
from astropy.io.registry import UnifiedReadWriteMethod
from . import groups
from .pprint import TableFormatter
from .column import (BaseColumn, Column, MaskedColumn, _auto_names, FalseArray,
                     col_copy, _convert_sequence_data_to_array)
from .row import Row
from .info import TableInfo
from .index import Index, _IndexModeContext, get_index
from .connect import TableRead, TableWrite
from .ndarray_mixin import NdarrayMixin
from .mixins.registry import get_mixin_handler
from . import conf
from .jsviewer import JSViewer
from IPython.display import HTML
import os
import webbrowser
import tempfile
from .jsviewer import DEFAULT_CSS
from urllib.parse import urljoin
from urllib.request import pathname2url
from .operations import _merge_table_meta
from pandas import DataFrame, Series
from astropy.utils.xml.writer import xml_escape
from . import serialize
from astropy.time import TimeBase, TimeDelta
from astropy.time import Time
from astropy.time import TimeDelta

_implementation_notes = """
This string has informal notes concerning Table implementation for developers.

Things to remember:

- Table has customizable attributes ColumnClass, Column, MaskedColumn.
  Table.Column is normally just column.Column (same w/ MaskedColumn)
  but in theory they can be different.  Table.ColumnClass is the default
  class used to create new non-mixin columns, and this is a function of
  the Table.masked attribute.  Column creation / manipulation in a Table
  needs to respect these.

- Column objects that get inserted into the Table.columns attribute must
  have the info.parent_table attribute set correctly.  Beware just dropping
  an object into the columns dict since an existing column may
  be part of another Table and have parent_table set to point at that
  table.  Dropping that column into `columns` of this Table will cause
  a problem for the old one so the column object needs to be copied (but
  not necessarily the data).

  Currently replace_column is always making a copy of both object and
  data if parent_table is set.  This could be improved but requires a
  generic way to copy a mixin object but not the data.

- Be aware of column objects that have indices set.

- `cls.ColumnClass` is a property that effectively uses the `masked` attribute
  to choose either `cls.Column` or `cls.MaskedColumn`.
"""
__doctest_skip__ = ['Table.read', 'Table.write', 'Table._read',
                    'Table.convert_bytestring_to_unicode',
                    'Table.convert_unicode_to_bytestring',
                    ]
__doctest_requires__ = {'*pandas': ['pandas>=1.1']}
_pprint_docs = """
    {__doc__}

    Parameters
    ----------
    max_lines : int or None
        Maximum number of lines in table output.

    max_width : int or None
        Maximum character width of output.

    show_name : bool
        Include a header row for column names. Default is True.

    show_unit : bool
        Include a header row for unit.  Default is to show a row
        for units only if one or more columns has a defined value
        for the unit.

    show_dtype : bool
        Include a header row for column dtypes. Default is False.

    align : str or list or tuple or None
        Left/right alignment of columns. Default is right (None) for all
        columns. Other allowed values are '>', '<', '^', and '0=' for
        right, left, centered, and 0-padded, respectively. A list of
        strings can be provided for alignment of tables with multiple
        columns.
    """
_pformat_docs = """
    {__doc__}

    Parameters
    ----------
    max_lines : int or None
        Maximum number of rows to output

    max_width : int or None
        Maximum character width of output

    show_name : bool
        Include a header row for column names. Default is True.

    show_unit : bool
        Include a header row for unit.  Default is to show a row
        for units only if one or more columns has a defined value
        for the unit.

    show_dtype : bool
        Include a header row for column dtypes. Default is True.

    html : bool
        Format the output as an HTML table. Default is False.

    tableid : str or None
        An ID tag for the table; only used if html is set.  Default is
        "table{id}", where id is the unique integer id of the table object,
        id(self)

    align : str or list or tuple or None
        Left/right alignment of columns. Default is right (None) for all
        columns. Other allowed values are '>', '<', '^', and '0=' for
        right, left, centered, and 0-padded, respectively. A list of
        strings can be provided for alignment of tables with multiple
        columns.

    tableclass : str or list of str or None
        CSS classes for the table; only used if html is set.  Default is
        None.

    Returns
    -------
    lines : list
        Formatted table as a list of strings.
    """

class Table:
    meta = MetaData(copy=False)
    Row = Row
    Column = Column
    MaskedColumn = MaskedColumn
    TableColumns = TableColumns
    TableFormatter = TableFormatter
    read = UnifiedReadWriteMethod(TableRead)
    write = UnifiedReadWriteMethod(TableWrite)
    pprint_exclude_names = PprintIncludeExclude()
    pprint_include_names = PprintIncludeExclude()
    info = TableInfo()
    def as_array(self, keep_byteorder=False, names=None):
        """
        Return a new copy of the table in the form of a structured np.ndarray or
        np.ma.MaskedArray object (as appropriate).

        Parameters
        ----------
        keep_byteorder : bool, optional
            By default the returned array has all columns in native byte
            order.  However, if this option is `True` this preserves the
            byte order of all columns (if any are non-native).

        names : list, optional:
            List of column names to include for returned structured array.
            Default is to include all table columns.

        Returns
        -------
        table_array : array or `~numpy.ma.MaskedArray`
            Copy of table as a numpy structured array.
            ndarray for unmasked or `~numpy.ma.MaskedArray` for masked.
        """
        masked = self.masked or self.has_masked_columns or self.has_masked_values
        empty_init = ma.empty if masked else np.empty
        if len(self.columns) == 0:
            return empty_init(0, dtype=None)

        dtype = []

        cols = self.columns.values()

        if names is not None:
            cols = [col for col in cols if col.info.name in names]

        for col in cols:
            col_descr = descr(col)

            if not (col.info.dtype.isnative or keep_byteorder):
                new_dt = np.dtype(col_descr[1]).newbyteorder('=')
                col_descr = (col_descr[0], new_dt, col_descr[2])

            dtype.append(col_descr)

        data = empty_init(len(self), dtype=dtype)
        for col in cols:
            # When assigning from one array into a field of a structured array,
            # Numpy will automatically swap those columns to their destination
            # byte order where applicable
            data[col.info.name] = col

            # For masked out, masked mixin columns need to set output mask attribute.
            if masked and has_info_class(col, MixinInfo) and hasattr(col, 'mask'):
                data[col.info.name].mask = col.mask

        return data
    def __array__(self, dtype=None):
        """Support converting Table to np.array via np.array(table).

        Coercion to a different dtype via np.array(table, dtype) is not
        supported and will raise a ValueError.
        """
        if dtype is not None:
            if np.dtype(dtype) != object:
                raise ValueError('Datatype coercion is not allowed')

            out = np.array(None, dtype=object)
            out[()] = self
            return out

        # This limitation is because of the following unexpected result that
        # should have made a table copy while changing the column names.
        #
        # >>> d = astropy.table.Table([[1,2],[3,4]])
        # >>> np.array(d, dtype=[('a', 'i8'), ('b', 'i8')])
        # array([(0, 0), (0, 0)],
        #       dtype=[('a', '<i8'), ('b', '<i8')])

        out = self.as_array()
        return out.data if isinstance(out, np.ma.MaskedArray) else out
    def itercols(self):
        """
        Iterate over the columns of this table.

        Examples
        --------

        To iterate over the columns of a table::

            >>> t = Table([[1], [2]])
            >>> for col in t.itercols():
            ...     print(col)
            col0
            ----
               1
            col1
            ----
               2

        Using ``itercols()`` is similar to  ``for col in t.columns.values()``
        but is syntactically preferred.
        """
        for colname in self.columns:
            yield self[colname]
    @property
    def has_masked_columns(self):
        """True if table has any ``MaskedColumn`` columns.

        This does not check for mixin columns that may have masked values, use the
        ``has_masked_values`` property in that case.

        """
        return any(isinstance(col, MaskedColumn) for col in self.itercols())
    @property
    def has_masked_values(self):
        """True if column in the table has values which are masked.

        This may be relatively slow for large tables as it requires checking the mask
        values of each column.
        """
        for col in self.itercols():
            if hasattr(col, 'mask') and np.any(col.mask):
                return True
        else:
            return False
    def __getitem__(self, item):
        if isinstance(item, str):
            return self.columns[item]
        elif isinstance(item, (int, np.integer)):
            return self.Row(self, item)
        elif (isinstance(item, np.ndarray) and item.shape == () and item.dtype.kind == 'i'):
            return self.Row(self, item.item())
        elif self._is_list_or_tuple_of_str(item):
            out = self.__class__([self[x] for x in item],
                                 copy_indices=self._copy_indices)
            out._groups = groups.TableGroups(out, indices=self.groups._indices,
                                             keys=self.groups._keys)
            out.meta = self.meta.copy()  # Shallow copy for meta
            return out
        elif ((isinstance(item, np.ndarray) and item.size == 0)
              or (isinstance(item, (tuple, list)) and not item)):
            # If item is an empty array/list/tuple then return the table with no rows
            return self._new_from_slice([])
        elif (isinstance(item, slice)
              or isinstance(item, np.ndarray)
              or isinstance(item, list)
              or isinstance(item, tuple) and all(isinstance(x, np.ndarray)
                                                 for x in item)):
            # here for the many ways to give a slice; a tuple of ndarray
            # is produced by np.where, as in t[np.where(t['a'] > 2)]
            # For all, a new table is constructed with slice of all columns
            return self._new_from_slice(item)
        else:
            raise ValueError(f'Illegal type {type(item)} for table item access')
    @masked.setter
    def masked(self, masked):
        raise Exception('Masked attribute is read-only (use t = Table(t, masked=True)'
                        ' to convert to a masked table)')
    def __len__(self):
        # For performance reasons (esp. in Row) cache the first column name
        # and use that subsequently for the table length.  If might not be
        # available yet or the column might be gone now, in which case
        # try again in the except block.
        try:
            return len(OrderedDict.__getitem__(self.columns, self._first_colname))
        except (AttributeError, KeyError):
            if len(self.columns) == 0:
                return 0

            # Get the first column name
            self._first_colname = next(iter(self.columns))
            return len(self.columns[self._first_colname])