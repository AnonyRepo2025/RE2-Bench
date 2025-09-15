import fnmatch
import os
import re
import sys
import numpy as np
from astropy import log
from astropy.utils.console import Getch, color_print, conf, terminal_size
from astropy.utils.data_info import dtype_info_name
from astropy.utils.xml.writer import xml_escape
from astropy.utils.xml.writer import xml_escape

__all__ = []

class TableFormatter:
    @staticmethod
    def _get_pprint_size(max_lines=None, max_width=None):
        """Get the output size (number of lines and character width) for Column and
        Table pformat/pprint methods.

        If no value of ``max_lines`` is supplied then the height of the
        screen terminal is used to set ``max_lines``.  If the terminal
        height cannot be determined then the default will be determined
        using the ``astropy.table.conf.max_lines`` configuration item. If a
        negative value of ``max_lines`` is supplied then there is no line
        limit applied.

        The same applies for max_width except the configuration item is
        ``astropy.table.conf.max_width``.

        Parameters
        ----------
        max_lines : int or None
            Maximum lines of output (header + data rows)

        max_width : int or None
            Maximum width (characters) output

        Returns
        -------
        max_lines, max_width : int

        """
        # Declare to keep static type checker happy.
        lines = None
        width = None

        if max_lines is None:
            max_lines = conf.max_lines

        if max_width is None:
            max_width = conf.max_width

        if max_lines is None or max_width is None:
            lines, width = terminal_size()

        if max_lines is None:
            max_lines = lines
        elif max_lines < 0:
            max_lines = sys.maxsize
        if max_lines < 8:
            max_lines = 8

        if max_width is None:
            max_width = width
        elif max_width < 0:
            max_width = sys.maxsize
        if max_width < 10:
            max_width = 10

        return max_lines, max_width
    def _name_and_structure(self, name, dtype, sep=" "):
        """Format a column name, including a possible structure.

        Normally, just returns the name, but if it has a structured dtype,
        will add the parts in between square brackets.  E.g.,
        "name [f0, f1]" or "name [f0[sf0, sf1], f1]".
        """
        if dtype is None or dtype.names is None:
            return name

        structure = ', '.join([self._name_and_structure(name, dt, sep="")
                               for name, (dt, _) in dtype.fields.items()])
        return f"{name}{sep}[{structure}]"
    def _pformat_col_iter(self, col, max_lines, show_name, show_unit, outs,
                          show_dtype=False, show_length=None):
        """Iterator which yields formatted string representation of column values.

        Parameters
        ----------
        max_lines : int
            Maximum lines of output (header + data rows)

        show_name : bool
            Include column name. Default is True.

        show_unit : bool
            Include a header row for unit.  Default is to show a row
            for units only if one or more columns has a defined value
            for the unit.

        outs : dict
            Must be a dict which is used to pass back additional values
            defined within the iterator.

        show_dtype : bool
            Include column dtype. Default is False.

        show_length : bool
            Include column length at end.  Default is to show this only
            if the column is not shown completely.
        """
        max_lines, _ = self._get_pprint_size(max_lines, -1)
        dtype = getattr(col, 'dtype', None)
        multidims = getattr(col, 'shape', [0])[1:]
        if multidims:
            multidim0 = tuple(0 for n in multidims)
            multidim1 = tuple(n - 1 for n in multidims)
            multidims_all_ones = np.prod(multidims) == 1
            multidims_has_zero = 0 in multidims

        i_dashes = None
        i_centers = []  # Line indexes where content should be centered
        n_header = 0
        if show_name:
            i_centers.append(n_header)
            # Get column name (or 'None' if not set)
            col_name = str(col.info.name)
            n_header += 1
            yield self._name_and_structure(col_name, dtype)
        if show_unit:
            i_centers.append(n_header)
            n_header += 1
            yield str(col.info.unit or '')
        if show_dtype:
            i_centers.append(n_header)
            n_header += 1
            if dtype is not None:
                col_dtype = dtype_info_name((dtype, multidims))
            else:
                col_dtype = col.__class__.__qualname__ or 'object'
            yield col_dtype
        if show_unit or show_name or show_dtype:
            i_dashes = n_header
            n_header += 1
            yield '---'

        max_lines -= n_header
        n_print2 = max_lines // 2
        n_rows = len(col)

        # This block of code is responsible for producing the function that
        # will format values for this column.  The ``format_func`` function
        # takes two args (col_format, val) and returns the string-formatted
        # version.  Some points to understand:
        #
        # - col_format could itself be the formatting function, so it will
        #    actually end up being called with itself as the first arg.  In
        #    this case the function is expected to ignore its first arg.
        #
        # - auto_format_func is a function that gets called on the first
        #    column value that is being formatted.  It then determines an
        #    appropriate formatting function given the actual value to be
        #    formatted.  This might be deterministic or it might involve
        #    try/except.  The latter allows for different string formatting
        #    options like %f or {:5.3f}.  When auto_format_func is called it:

        #    1. Caches the function in the _format_funcs dict so for subsequent
        #       values the right function is called right away.
        #    2. Returns the formatted value.
        #
        # - possible_string_format_functions is a function that yields a
        #    succession of functions that might successfully format the
        #    value.  There is a default, but Mixin methods can override this.
        #    See Quantity for an example.
        #
        # - get_auto_format_func() returns a wrapped version of auto_format_func
        #    with the column id and possible_string_format_functions as
        #    enclosed variables.
        col_format = col.info.format or getattr(col.info, 'default_format',
                                                None)
        pssf = (getattr(col.info, 'possible_string_format_functions', None)
                or _possible_string_format_functions)
        auto_format_func = get_auto_format_func(col, pssf)
        format_func = col.info._format_funcs.get(col_format, auto_format_func)

        if len(col) > max_lines:
            if show_length is None:
                show_length = True
            i0 = n_print2 - (1 if show_length else 0)
            i1 = n_rows - n_print2 - max_lines % 2
            indices = np.concatenate([np.arange(0, i0 + 1),
                                      np.arange(i1 + 1, len(col))])
        else:
            i0 = -1
            indices = np.arange(len(col))

        def format_col_str(idx):
            if multidims:
                # Prevents columns like Column(data=[[(1,)],[(2,)]], name='a')
                # with shape (n,1,...,1) from being printed as if there was
                # more than one element in a row
                if multidims_all_ones:
                    return format_func(col_format, col[(idx,) + multidim0])
                elif multidims_has_zero:
                    # Any zero dimension means there is no data to print
                    return ""
                else:
                    left = format_func(col_format, col[(idx,) + multidim0])
                    right = format_func(col_format, col[(idx,) + multidim1])
                    return f'{left} .. {right}'
            else:
                return format_func(col_format, col[idx])

        # Add formatted values if within bounds allowed by max_lines
        for idx in indices:
            if idx == i0:
                yield '...'
            else:
                try:
                    yield format_col_str(idx)
                except ValueError:
                    raise ValueError(
                        'Unable to parse format string "{}" for entry "{}" '
                        'in column "{}"'.format(col_format, col[idx],
                                                col.info.name))

        outs['show_length'] = show_length
        outs['n_header'] = n_header
        outs['i_centers'] = i_centers
        outs['i_dashes'] = i_dashes