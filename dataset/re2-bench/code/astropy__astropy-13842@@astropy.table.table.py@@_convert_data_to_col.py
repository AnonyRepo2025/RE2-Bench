import itertools
import sys
import types
import warnings
import weakref
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from copy import deepcopy
import numpy as np
from numpy import ma
from astropy import log
from astropy.io.registry import UnifiedReadWriteMethod
from astropy.units import Quantity, QuantityInfo
from astropy.utils import ShapedLikeNDArray, isiterable
from astropy.utils.console import color_print
from astropy.utils.data_info import BaseColumnInfo, DataInfo, MixinInfo
from astropy.utils.decorators import format_doc
from astropy.utils.exceptions import AstropyUserWarning
from astropy.utils.masked import Masked
from astropy.utils.metadata import MetaAttribute, MetaData
from . import conf, groups
from .column import (
    BaseColumn, Column, FalseArray, MaskedColumn, _auto_names, _convert_sequence_data_to_array,
    col_copy)
from .connect import TableRead, TableWrite
from .index import (
    Index, SlicedIndex, TableILoc, TableIndices, TableLoc, TableLocIndices, _IndexModeContext,
    get_index)
from .info import TableInfo
from .mixins.registry import get_mixin_handler
from .ndarray_mixin import NdarrayMixin
from .pprint import TableFormatter
from .row import Row
from IPython.display import HTML
from .jsviewer import JSViewer
import os
import tempfile
import webbrowser
from urllib.parse import urljoin
from urllib.request import pathname2url
from .jsviewer import DEFAULT_CSS
from .operations import _merge_table_meta
from pandas import DataFrame, Series
from astropy.utils.xml.writer import xml_escape
from astropy.time import TimeBase, TimeDelta
from . import serialize
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
    def _convert_data_to_col(self, data, copy=True, default_name=None, dtype=None, name=None):
        """
        Convert any allowed sequence data ``col`` to a column object that can be used
        directly in the self.columns dict.  This could be a Column, MaskedColumn,
        or mixin column.

        The final column name is determined by::

            name or data.info.name or def_name

        If ``data`` has no ``info`` then ``name = name or def_name``.

        The behavior of ``copy`` for Column objects is:
        - copy=True: new class instance with a copy of data and deep copy of meta
        - copy=False: new class instance with same data and a key-only copy of meta

        For mixin columns:
        - copy=True: new class instance with copy of data and deep copy of meta
        - copy=False: original instance (no copy at all)

        Parameters
        ----------
        data : object (column-like sequence)
            Input column data
        copy : bool
            Make a copy
        default_name : str
            Default name
        dtype : np.dtype or None
            Data dtype
        name : str or None
            Column name

        Returns
        -------
        col : Column, MaskedColumn, mixin-column type
            Object that can be used as a column in self
        """

        data_is_mixin = self._is_mixin_for_table(data)
        masked_col_cls = (self.ColumnClass
                          if issubclass(self.ColumnClass, self.MaskedColumn)
                          else self.MaskedColumn)

        try:
            data0_is_mixin = self._is_mixin_for_table(data[0])
        except Exception:
            # Need broad exception, cannot predict what data[0] raises for arbitrary data
            data0_is_mixin = False

        # If the data is not an instance of Column or a mixin class, we can
        # check the registry of mixin 'handlers' to see if the column can be
        # converted to a mixin class
        if (handler := get_mixin_handler(data)) is not None:
            original_data = data
            data = handler(data)
            if not (data_is_mixin := self._is_mixin_for_table(data)):
                fully_qualified_name = (original_data.__class__.__module__ + '.'
                                        + original_data.__class__.__name__)
                raise TypeError('Mixin handler for object of type '
                                f'{fully_qualified_name} '
                                'did not return a valid mixin column')

        # Get the final column name using precedence.  Some objects may not
        # have an info attribute. Also avoid creating info as a side effect.
        if not name:
            if isinstance(data, Column):
                name = data.name or default_name
            elif 'info' in getattr(data, '__dict__', ()):
                name = data.info.name or default_name
            else:
                name = default_name

        if isinstance(data, Column):
            # If self.ColumnClass is a subclass of col, then "upgrade" to ColumnClass,
            # otherwise just use the original class.  The most common case is a
            # table with masked=True and ColumnClass=MaskedColumn.  Then a Column
            # gets upgraded to MaskedColumn, but the converse (pre-4.0) behavior
            # of downgrading from MaskedColumn to Column (for non-masked table)
            # does not happen.
            col_cls = self._get_col_cls_for_table(data)

        elif data_is_mixin:
            # Copy the mixin column attributes if they exist since the copy below
            # may not get this attribute. If not copying, take a slice
            # to ensure we get a new instance and we do not share metadata
            # like info.
            col = col_copy(data, copy_indices=self._init_indices) if copy else data[:]
            col.info.name = name
            return col

        elif data0_is_mixin:
            # Handle case of a sequence of a mixin, e.g. [1*u.m, 2*u.m].
            try:
                col = data[0].__class__(data)
                col.info.name = name
                return col
            except Exception:
                # If that didn't work for some reason, just turn it into np.array of object
                data = np.array(data, dtype=object)
                col_cls = self.ColumnClass

        elif isinstance(data, (np.ma.MaskedArray, Masked)):
            # Require that col_cls be a subclass of MaskedColumn, remembering
            # that ColumnClass could be a user-defined subclass (though more-likely
            # could be MaskedColumn).
            col_cls = masked_col_cls

        elif data is None:
            # Special case for data passed as the None object (for broadcasting
            # to an object column). Need to turn data into numpy `None` scalar
            # object, otherwise `Column` interprets data=None as no data instead
            # of a object column of `None`.
            data = np.array(None)
            col_cls = self.ColumnClass

        elif not hasattr(data, 'dtype'):
            # `data` is none of the above, convert to numpy array or MaskedArray
            # assuming only that it is a scalar or sequence or N-d nested
            # sequence. This function is relatively intricate and tries to
            # maintain performance for common cases while handling things like
            # list input with embedded np.ma.masked entries. If `data` is a
            # scalar then it gets returned unchanged so the original object gets
            # passed to `Column` later.
            data = _convert_sequence_data_to_array(data, dtype)
            copy = False  # Already made a copy above
            col_cls = masked_col_cls if isinstance(data, np.ma.MaskedArray) else self.ColumnClass

        else:
            col_cls = self.ColumnClass

        try:
            col = col_cls(name=name, data=data, dtype=dtype,
                          copy=copy, copy_indices=self._init_indices)
        except Exception:
            # Broad exception class since we don't know what might go wrong
            raise ValueError('unable to convert data to Column for Table')

        col = self._convert_col_for_table(col)

        return col
    def _get_col_cls_for_table(self, col):
        """Get the correct column class to use for upgrading any Column-like object.

        For a masked table, ensure any Column-like object is a subclass
        of the table MaskedColumn.

        For unmasked table, ensure any MaskedColumn-like object is a subclass
        of the table MaskedColumn.  If not a MaskedColumn, then ensure that any
        Column-like object is a subclass of the table Column.
        """

        col_cls = col.__class__

        if self.masked:
            if isinstance(col, Column) and not isinstance(col, self.MaskedColumn):
                col_cls = self.MaskedColumn
        else:
            if isinstance(col, MaskedColumn):
                if not isinstance(col, self.MaskedColumn):
                    col_cls = self.MaskedColumn
            elif isinstance(col, Column) and not isinstance(col, self.Column):
                col_cls = self.Column

        return col_cls
    def _convert_col_for_table(self, col):
        """
        Make sure that all Column objects have correct base class for this type of
        Table.  For a base Table this most commonly means setting to
        MaskedColumn if the table is masked.  Table subclasses like QTable
        override this method.
        """
        if isinstance(col, Column) and not isinstance(col, self.ColumnClass):
            col_cls = self._get_col_cls_for_table(col)
            if col_cls is not col.__class__:
                col = col_cls(col, copy=False)

        return col
    def _is_mixin_for_table(self, col):
        """
        Determine if ``col`` should be added to the table directly as
        a mixin column.
        """
        if isinstance(col, BaseColumn):
            return False

        # Is it a mixin but not [Masked]Quantity (which gets converted to
        # [Masked]Column with unit set).
        return has_info_class(col, MixinInfo) and not has_info_class(col, QuantityInfo)
    @masked.setter
    def masked(self, masked):
        raise Exception('Masked attribute is read-only (use t = Table(t, masked=True)'
                        ' to convert to a masked table)')
    @property
    def ColumnClass(self):
        if self._column_class is None:
            return self.Column
        else:
            return self._column_class