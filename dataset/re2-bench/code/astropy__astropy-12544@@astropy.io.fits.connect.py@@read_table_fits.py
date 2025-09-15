import os
import re
import warnings
from copy import deepcopy
import numpy as np
from astropy.io import registry as io_registry
from astropy import units as u
from astropy.table import Table, serialize, meta, Column, MaskedColumn
from astropy.time import Time
from astropy.utils.data_info import serialize_context_as
from astropy.utils.exceptions import (AstropyUserWarning,
                                      AstropyDeprecationWarning)
from astropy.utils.misc import NOT_OVERWRITING_MSG
from . import HDUList, TableHDU, BinTableHDU, GroupsHDU, append as fits_append
from .column import KEYWORD_NAMES, _fortran_to_python_format
from .convenience import table_to_hdu
from .hdu.hdulist import fitsopen as fits_open, FITS_SIGNATURE
from .util import first
from .fitstime import fits_to_time

REMOVE_KEYWORDS = ['XTENSION', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2',
                   'PCOUNT', 'GCOUNT', 'TFIELDS', 'THEAP']
COLUMN_KEYWORD_REGEXP = '(' + '|'.join(KEYWORD_NAMES) + ')[0-9]+'

def read_table_fits(input, hdu=None, astropy_native=False, memmap=False,
                    character_as_bytes=True, unit_parse_strict='warn',
                    mask_invalid=True):
    """
    Read a Table object from an FITS file

    If the ``astropy_native`` argument is ``True``, then input FITS columns
    which are representations of an astropy core object will be converted to
    that class and stored in the ``Table`` as "mixin columns".  Currently this
    is limited to FITS columns which adhere to the FITS Time standard, in which
    case they will be converted to a `~astropy.time.Time` column in the output
    table.

    Parameters
    ----------
    input : str or file-like or compatible `astropy.io.fits` HDU object
        If a string, the filename to read the table from. If a file object, or
        a compatible HDU object, the object to extract the table from. The
        following `astropy.io.fits` HDU objects can be used as input:
        - :class:`~astropy.io.fits.hdu.table.TableHDU`
        - :class:`~astropy.io.fits.hdu.table.BinTableHDU`
        - :class:`~astropy.io.fits.hdu.table.GroupsHDU`
        - :class:`~astropy.io.fits.hdu.hdulist.HDUList`
    hdu : int or str, optional
        The HDU to read the table from.
    astropy_native : bool, optional
        Read in FITS columns as native astropy objects where possible instead
        of standard Table Column objects. Default is False.
    memmap : bool, optional
        Whether to use memory mapping, which accesses data on disk as needed. If
        you are only accessing part of the data, this is often more efficient.
        If you want to access all the values in the table, and you are able to
        fit the table in memory, you may be better off leaving memory mapping
        off. However, if your table would not fit in memory, you should set this
        to `True`.
        When set to `True` then ``mask_invalid`` is set to `False` since the
        masking would cause loading the full data array.
    character_as_bytes : bool, optional
        If `True`, string columns are stored as Numpy byte arrays (dtype ``S``)
        and are converted on-the-fly to unicode strings when accessing
        individual elements. If you need to use Numpy unicode arrays (dtype
        ``U``) internally, you should set this to `False`, but note that this
        will use more memory. If set to `False`, string columns will not be
        memory-mapped even if ``memmap`` is `True`.
    unit_parse_strict : str, optional
        Behaviour when encountering invalid column units in the FITS header.
        Default is "warn", which will emit a ``UnitsWarning`` and create a
        :class:`~astropy.units.core.UnrecognizedUnit`.
        Values are the ones allowed by the ``parse_strict`` argument of
        :class:`~astropy.units.core.Unit`: ``raise``, ``warn`` and ``silent``.
    mask_invalid : bool, optional
        By default the code masks NaNs in float columns and empty strings in
        string columns. Set this parameter to `False` to avoid the performance
        penalty of doing this masking step. The masking is always deactivated
        when using ``memmap=True`` (see above).

    """

    if isinstance(input, HDUList):

        # Parse all table objects
        tables = dict()
        for ihdu, hdu_item in enumerate(input):
            if isinstance(hdu_item, (TableHDU, BinTableHDU, GroupsHDU)):
                tables[ihdu] = hdu_item

        if len(tables) > 1:
            if hdu is None:
                warnings.warn("hdu= was not specified but multiple tables"
                              " are present, reading in first available"
                              f" table (hdu={first(tables)})",
                              AstropyUserWarning)
                hdu = first(tables)

            # hdu might not be an integer, so we first need to convert it
            # to the correct HDU index
            hdu = input.index_of(hdu)

            if hdu in tables:
                table = tables[hdu]
            else:
                raise ValueError(f"No table found in hdu={hdu}")

        elif len(tables) == 1:
            if hdu is not None:
                msg = None
                try:
                    hdi = input.index_of(hdu)
                except KeyError:
                    msg = f"Specified hdu={hdu} not found"
                else:
                    if hdi >= len(input):
                        msg = f"Specified hdu={hdu} not found"
                    elif hdi not in tables:
                        msg = f"No table found in specified hdu={hdu}"
                if msg is not None:
                    warnings.warn(f"{msg}, reading in first available table "
                                  f"(hdu={first(tables)}) instead. This will"
                                  " result in an error in future versions!",
                                  AstropyDeprecationWarning)
            table = tables[first(tables)]

        else:
            raise ValueError("No table found")

    elif isinstance(input, (TableHDU, BinTableHDU, GroupsHDU)):

        table = input

    else:

        if memmap:
            # using memmap is not compatible with masking invalid value by
            # default so we deactivate the masking
            mask_invalid = False

        hdulist = fits_open(input, character_as_bytes=character_as_bytes,
                            memmap=memmap)

        try:
            return read_table_fits(
                hdulist, hdu=hdu,
                astropy_native=astropy_native,
                unit_parse_strict=unit_parse_strict,
                mask_invalid=mask_invalid,
            )
        finally:
            hdulist.close()

    # In the loop below we access the data using data[col.name] rather than
    # col.array to make sure that the data is scaled correctly if needed.
    data = table.data

    columns = []
    for col in data.columns:
        # Check if column is masked. Here, we make a guess based on the
        # presence of FITS mask values. For integer columns, this is simply
        # the null header, for float and complex, the presence of NaN, and for
        # string, empty strings.
        # Since Multi-element columns with dtypes such as '2f8' have a subdtype,
        # we should look up the type of column on that.
        masked = mask = False
        coltype = (col.dtype.subdtype[0].type if col.dtype.subdtype
                   else col.dtype.type)
        if col.null is not None:
            mask = data[col.name] == col.null
            # Return a MaskedColumn even if no elements are masked so
            # we roundtrip better.
            masked = True
        elif mask_invalid and issubclass(coltype, np.inexact):
            mask = np.isnan(data[col.name])
        elif mask_invalid and issubclass(coltype, np.character):
            mask = col.array == b''

        if masked or np.any(mask):
            column = MaskedColumn(data=data[col.name], name=col.name,
                                  mask=mask, copy=False)
        else:
            column = Column(data=data[col.name], name=col.name, copy=False)

        # Copy over units
        if col.unit is not None:
            column.unit = u.Unit(col.unit, format='fits', parse_strict=unit_parse_strict)

        # Copy over display format
        if col.disp is not None:
            column.format = _fortran_to_python_format(col.disp)

        columns.append(column)

    # Create Table object
    t = Table(columns, copy=False)

    # TODO: deal properly with unsigned integers

    hdr = table.header
    if astropy_native:
        # Avoid circular imports, and also only import if necessary.
        from .fitstime import fits_to_time
        hdr = fits_to_time(hdr, t)

    for key, value, comment in hdr.cards:

        if key in ['COMMENT', 'HISTORY']:
            # Convert to io.ascii format
            if key == 'COMMENT':
                key = 'comments'

            if key in t.meta:
                t.meta[key].append(value)
            else:
                t.meta[key] = [value]

        elif key in t.meta:  # key is duplicate

            if isinstance(t.meta[key], list):
                t.meta[key].append(value)
            else:
                t.meta[key] = [t.meta[key], value]

        elif is_column_keyword(key) or key in REMOVE_KEYWORDS:

            pass

        else:

            t.meta[key] = value

    # TODO: implement masking

    # Decode any mixin columns that have been stored as standard Columns.
    t = _decode_mixins(t)

    return t
