import copy
import numbers
import operator
import re
import sys
import warnings
import weakref
from collections import OrderedDict
from contextlib import suppress
from functools import reduce
import numpy as np
from numpy import char as chararray
from astropy.utils import indent, isiterable, lazyproperty
from astropy.utils.exceptions import AstropyUserWarning
from .card import CARD_LENGTH, Card
from .util import NotifierMixin, _convert_array, _is_int, cmp, encode_ascii, pairwise
from .verify import VerifyError, VerifyWarning
from .fitsrec import FITS_rec
from .hdu.table import _TableBaseHDU

__all__ = ["Column", "ColDefs", "Delayed"]
FITS2NUMPY = {
    "L": "i1",
    "B": "u1",
    "I": "i2",
    "J": "i4",
    "K": "i8",
    "E": "f4",
    "D": "f8",
    "C": "c8",
    "M": "c16",
    "A": "a",
}
NUMPY2FITS = {val: key for key, val in FITS2NUMPY.items()}
NUMPY2FITS["b1"] = "L"
NUMPY2FITS["u2"] = "I"
NUMPY2FITS["u4"] = "J"
NUMPY2FITS["u8"] = "K"
NUMPY2FITS["f2"] = "E"
FORMATORDER = ["L", "B", "I", "J", "K", "D", "M", "A"]
FITSUPCONVERTERS = {"E": "D", "C": "M"}
ASCII2NUMPY = {"A": "a", "I": "i4", "J": "i8", "F": "f8", "E": "f8", "D": "f8"}
ASCII2STR = {"A": "", "I": "d", "J": "d", "F": "f", "E": "E", "D": "E"}
ASCII_DEFAULT_WIDTHS = {
    "A": (1, 0),
    "I": (10, 0),
    "J": (15, 0),
    "E": (15, 7),
    "F": (16, 7),
    "D": (25, 17),
}
TDISP_RE_DICT = {}
TDISP_RE_DICT["F"] = re.compile(
    r"(?:(?P<formatc>[F])(?:(?P<width>[0-9]+)\.{1}" r"(?P<precision>[0-9])+)+)|"
)
TDISP_RE_DICT["A"] = TDISP_RE_DICT["L"] = re.compile(
    r"(?:(?P<formatc>[AL])(?P<width>[0-9]+)+)|"
)
TDISP_RE_DICT["I"] = TDISP_RE_DICT["B"] = TDISP_RE_DICT["O"] = TDISP_RE_DICT[
    "Z"
] = re.compile(
    r"(?:(?P<formatc>[IBOZ])(?:(?P<width>[0-9]+)"
    r"(?:\.{0,1}(?P<precision>[0-9]+))?))|"
)
TDISP_RE_DICT["E"] = TDISP_RE_DICT["G"] = TDISP_RE_DICT["D"] = re.compile(
    r"(?:(?P<formatc>[EGD])(?:(?P<width>[0-9]+)\."
    r"(?P<precision>[0-9]+))+)"
    r"(?:E{0,1}(?P<exponential>[0-9]+)?)|"
)
TDISP_RE_DICT["EN"] = TDISP_RE_DICT["ES"] = re.compile(
    r"(?:(?P<formatc>E[NS])(?:(?P<width>[0-9]+)\.{1}" r"(?P<precision>[0-9])+)+)"
)
TDISP_FMT_DICT = {
    "I": "{{:{width}d}}",
    "B": "{{:{width}b}}",
    "O": "{{:{width}o}}",
    "Z": "{{:{width}x}}",
    "F": "{{:{width}.{precision}f}}",
    "G": "{{:{width}.{precision}g}}",
}
TDISP_FMT_DICT["A"] = TDISP_FMT_DICT["L"] = "{{:>{width}}}"
TDISP_FMT_DICT["E"] = TDISP_FMT_DICT["D"] = TDISP_FMT_DICT["EN"] = TDISP_FMT_DICT[
    "ES"
] = "{{:{width}.{precision}e}}"
KEYWORD_NAMES = (
    "TTYPE",
    "TFORM",
    "TUNIT",
    "TNULL",
    "TSCAL",
    "TZERO",
    "TDISP",
    "TBCOL",
    "TDIM",
    "TCTYP",
    "TCUNI",
    "TCRPX",
    "TCRVL",
    "TCDLT",
    "TRPOS",
)
KEYWORD_ATTRIBUTES = (
    "name",
    "format",
    "unit",
    "null",
    "bscale",
    "bzero",
    "disp",
    "start",
    "dim",
    "coord_type",
    "coord_unit",
    "coord_ref_point",
    "coord_ref_value",
    "coord_inc",
    "time_ref_pos",
)
KEYWORD_TO_ATTRIBUTE = OrderedDict(zip(KEYWORD_NAMES, KEYWORD_ATTRIBUTES))
ATTRIBUTE_TO_KEYWORD = OrderedDict(zip(KEYWORD_ATTRIBUTES, KEYWORD_NAMES))
TFORMAT_RE = re.compile(
    r"(?P<repeat>^[0-9]*)(?P<format>[LXBIJKAEDCMPQ])" r"(?P<option>[!-~]*)", re.I
)
TFORMAT_ASCII_RE = re.compile(
    r"(?:(?P<format>[AIJ])(?P<width>[0-9]+)?)|"
    r"(?:(?P<formatf>[FED])"
    r"(?:(?P<widthf>[0-9]+)(?:\."
    r"(?P<precision>[0-9]+))?)?)"
)
TTYPE_RE = re.compile(r"[0-9a-zA-Z_]+")
TDEF_RE = re.compile(r"(?P<label>^T[A-Z]*)(?P<num>[1-9][0-9 ]*$)")
TDIM_RE = re.compile(r"\(\s*(?P<dims>(?:\d+\s*)(?:,\s*\d+\s*)*\s*)\)\s*")
ASCIITNULL = 0
DEFAULT_ASCII_TNULL = "---"

class Column(NotifierMixin):
    """
    Class which contains the definition of one column, e.g.  ``ttype``,
    ``tform``, etc. and the array containing values for the column.
    """

    def __init__(
        self,
        name=None,
        format=None,
        unit=None,
        null=None,
        bscale=None,
        bzero=None,
        disp=None,
        start=None,
        dim=None,
        array=None,
        ascii=None,
        coord_type=None,
        coord_unit=None,
        coord_ref_point=None,
        coord_ref_value=None,
        coord_inc=None,
        time_ref_pos=None,
    ):
        """
        Construct a `Column` by specifying attributes.  All attributes
        except ``format`` can be optional; see :ref:`astropy:column_creation`
        and :ref:`astropy:creating_ascii_table` for more information regarding
        ``TFORM`` keyword.

        Parameters
        ----------
        name : str, optional
            column name, corresponding to ``TTYPE`` keyword

        format : str
            column format, corresponding to ``TFORM`` keyword

        unit : str, optional
            column unit, corresponding to ``TUNIT`` keyword

        null : str, optional
            null value, corresponding to ``TNULL`` keyword

        bscale : int-like, optional
            bscale value, corresponding to ``TSCAL`` keyword

        bzero : int-like, optional
            bzero value, corresponding to ``TZERO`` keyword

        disp : str, optional
            display format, corresponding to ``TDISP`` keyword

        start : int, optional
            column starting position (ASCII table only), corresponding
            to ``TBCOL`` keyword

        dim : str, optional
            column dimension corresponding to ``TDIM`` keyword

        array : iterable, optional
            a `list`, `numpy.ndarray` (or other iterable that can be used to
            initialize an ndarray) providing initial data for this column.
            The array will be automatically converted, if possible, to the data
            format of the column.  In the case were non-trivial ``bscale``
            and/or ``bzero`` arguments are given, the values in the array must
            be the *physical* values--that is, the values of column as if the
            scaling has already been applied (the array stored on the column
            object will then be converted back to its storage values).

        ascii : bool, optional
            set `True` if this describes a column for an ASCII table; this
            may be required to disambiguate the column format

        coord_type : str, optional
            coordinate/axis type corresponding to ``TCTYP`` keyword

        coord_unit : str, optional
            coordinate/axis unit corresponding to ``TCUNI`` keyword

        coord_ref_point : int-like, optional
            pixel coordinate of the reference point corresponding to ``TCRPX``
            keyword

        coord_ref_value : int-like, optional
            coordinate value at reference point corresponding to ``TCRVL``
            keyword

        coord_inc : int-like, optional
            coordinate increment at reference point corresponding to ``TCDLT``
            keyword

        time_ref_pos : str, optional
            reference position for a time coordinate column corresponding to
            ``TRPOS`` keyword
        """

        if format is None:
            raise ValueError("Must specify format to construct Column.")

        # any of the input argument (except array) can be a Card or just
        # a number/string
        kwargs = {"ascii": ascii}
        for attr in KEYWORD_ATTRIBUTES:
            value = locals()[attr]  # get the argument's value

            if isinstance(value, Card):
                value = value.value

            kwargs[attr] = value

        valid_kwargs, invalid_kwargs = self._verify_keywords(**kwargs)

        if invalid_kwargs:
            msg = ["The following keyword arguments to Column were invalid:"]

            for val in invalid_kwargs.values():
                msg.append(indent(val[1]))

            raise VerifyError("\n".join(msg))

        for attr in KEYWORD_ATTRIBUTES:
            setattr(self, attr, valid_kwargs.get(attr))

        # TODO: Try to eliminate the following two special cases
        # for recformat and dim:
        # This is not actually stored as an attribute on columns for some
        # reason
        recformat = valid_kwargs["recformat"]

        # The 'dim' keyword's original value is stored in self.dim, while
        # *only* the tuple form is stored in self._dims.
        self._dims = self.dim
        self.dim = dim

        # Awful hack to use for now to keep track of whether the column holds
        # pseudo-unsigned int data
        self._pseudo_unsigned_ints = False

        # if the column data is not ndarray, make it to be one, i.e.
        # input arrays can be just list or tuple, not required to be ndarray
        # does not include Object array because there is no guarantee
        # the elements in the object array are consistent.
        if not isinstance(array, (np.ndarray, chararray.chararray, Delayed)):
            try:  # try to convert to a ndarray first
                if array is not None:
                    array = np.array(array)
            except Exception:
                try:  # then try to convert it to a strings array
                    itemsize = int(recformat[1:])
                    array = chararray.array(array, itemsize=itemsize)
                except ValueError:
                    # then try variable length array
                    # Note: This includes _FormatQ by inheritance
                    if isinstance(recformat, _FormatP):
                        array = _VLF(array, dtype=recformat.dtype)
                    else:
                        raise ValueError(
                            f"Data is inconsistent with the format `{format}`."
                        )

        array = self._convert_to_valid_data_type(array)

        # We have required (through documentation) that arrays passed in to
        # this constructor are already in their physical values, so we make
        # note of that here
        if isinstance(array, np.ndarray):
            self._physical_values = True
        else:
            self._physical_values = False

        self._parent_fits_rec = None
        self.array = array

    def __repr__(self):
        text = ""
        for attr in KEYWORD_ATTRIBUTES:
            value = getattr(self, attr)
            if value is not None:
                text += attr + " = " + repr(value) + "; "
        return text[:-2]

    def __eq__(self, other):
        """
        Two columns are equal if their name and format are the same.  Other
        attributes aren't taken into account at this time.
        """

        # According to the FITS standard column names must be case-insensitive
        a = (self.name.lower(), self.format)
        b = (other.name.lower(), other.format)
        return a == b

    def __hash__(self):
        """
        Like __eq__, the hash of a column should be based on the unique column
        name and format, and be case-insensitive with respect to the column
        name.
        """

        return hash((self.name.lower(), self.format))

    @property
    def array(self):
        """
        The Numpy `~numpy.ndarray` associated with this `Column`.

        If the column was instantiated with an array passed to the ``array``
        argument, this will return that array.  However, if the column is
        later added to a table, such as via `BinTableHDU.from_columns` as
        is typically the case, this attribute will be updated to reference
        the associated field in the table, which may no longer be the same
        array.
        """

        # Ideally the .array attribute never would have existed in the first
        # place, or would have been internal-only.  This is a legacy of the
        # older design from Astropy that needs to have continued support, for
        # now.

        # One of the main problems with this design was that it created a
        # reference cycle.  When the .array attribute was updated after
        # creating a FITS_rec from the column (as explained in the docstring) a
        # reference cycle was created.  This is because the code in BinTableHDU
        # (and a few other places) does essentially the following:
        #
        # data._coldefs = columns  # The ColDefs object holding this Column
        # for col in columns:
        #     col.array = data.field(col.name)
        #
        # This way each columns .array attribute now points to the field in the
        # table data.  It's actually a pretty confusing interface (since it
        # replaces the array originally pointed to by .array), but it's the way
        # things have been for a long, long time.
        #
        # However, this results, in *many* cases, in a reference cycle.
        # Because the array returned by data.field(col.name), while sometimes
        # an array that owns its own data, is usually like a slice of the
        # original data.  It has the original FITS_rec as the array .base.
        # This results in the following reference cycle (for the n-th column):
        #
        #    data -> data._coldefs -> data._coldefs[n] ->
        #     data._coldefs[n].array -> data._coldefs[n].array.base -> data
        #
        # Because ndarray objects do not handled by Python's garbage collector
        # the reference cycle cannot be broken.  Therefore the FITS_rec's
        # refcount never goes to zero, its __del__ is never called, and its
        # memory is never freed.  This didn't occur in *all* cases, but it did
        # occur in many cases.
        #
        # To get around this, Column.array is no longer a simple attribute
        # like it was previously.  Now each Column has a ._parent_fits_rec
        # attribute which is a weakref to a FITS_rec object.  Code that
        # previously assigned each col.array to field in a FITS_rec (as in
        # the example a few paragraphs above) is still used, however now
        # array.setter checks if a reference cycle will be created.  And if
        # so, instead of saving directly to the Column's __dict__, it creates
        # the ._prent_fits_rec weakref, and all lookups of the column's .array
        # go through that instead.
        #
        # This alone does not fully solve the problem.  Because
        # _parent_fits_rec is a weakref, if the user ever holds a reference to
        # the Column, but deletes all references to the underlying FITS_rec,
        # the .array attribute would suddenly start returning None instead of
        # the array data.  This problem is resolved on FITS_rec's end.  See the
        # note in the FITS_rec._coldefs property for the rest of the story.

        # If the Columns's array is not a reference to an existing FITS_rec,
        # then it is just stored in self.__dict__; otherwise check the
        # _parent_fits_rec reference if it 's still available.
        if "array" in self.__dict__:
            return self.__dict__["array"]
        elif self._parent_fits_rec is not None:
            parent = self._parent_fits_rec()
            if parent is not None:
                return parent[self.name]
        else:
            return None

    @array.setter
    def array(self, array):
        # The following looks over the bases of the given array to check if it
        # has a ._coldefs attribute (i.e. is a FITS_rec) and that that _coldefs
        # contains this Column itself, and would create a reference cycle if we
        # stored the array directly in self.__dict__.
        # In this case it instead sets up the _parent_fits_rec weakref to the
        # underlying FITS_rec, so that array.getter can return arrays through
        # self._parent_fits_rec().field(self.name), rather than storing a
        # hard reference to the field like it used to.
        base = array
        while True:
            if hasattr(base, "_coldefs") and isinstance(base._coldefs, ColDefs):
                for col in base._coldefs:
                    if col is self and self._parent_fits_rec is None:
                        self._parent_fits_rec = weakref.ref(base)

                        # Just in case the user already set .array to their own
                        # array.
                        if "array" in self.__dict__:
                            del self.__dict__["array"]
                        return

            if getattr(base, "base", None) is not None:
                base = base.base
            else:
                break

        self.__dict__["array"] = array

    @array.deleter
    def array(self):
        try:
            del self.__dict__["array"]
        except KeyError:
            pass

        self._parent_fits_rec = None

    @ColumnAttribute("TTYPE")
    def name(col, name):
        if name is None:
            # Allow None to indicate deleting the name, or to just indicate an
            # unspecified name (when creating a new Column).
            return

        # Check that the name meets the recommended standard--other column
        # names are *allowed*, but will be discouraged
        if isinstance(name, str) and not TTYPE_RE.match(name):
            warnings.warn(
                "It is strongly recommended that column names contain only "
                "upper and lower-case ASCII letters, digits, or underscores "
                "for maximum compatibility with other software "
                "(got {!r}).".format(name),
                VerifyWarning,
            )

        # This ensures that the new name can fit into a single FITS card
        # without any special extension like CONTINUE cards or the like.
        if not isinstance(name, str) or len(str(Card("TTYPE", name))) != CARD_LENGTH:
            raise AssertionError(
                "Column name must be a string able to fit in a single "
                "FITS card--typically this means a maximum of 68 "
                "characters, though it may be fewer if the string "
                "contains special characters like quotes."
            )

    @ColumnAttribute("TCTYP")
    def coord_type(col, coord_type):
        if coord_type is None:
            return

        if not isinstance(coord_type, str) or len(coord_type) > 8:
            raise AssertionError(
                "Coordinate/axis type must be a string of atmost 8 characters."
            )

    @ColumnAttribute("TCUNI")
    def coord_unit(col, coord_unit):
        if coord_unit is not None and not isinstance(coord_unit, str):
            raise AssertionError("Coordinate/axis unit must be a string.")

    @ColumnAttribute("TCRPX")
    def coord_ref_point(col, coord_ref_point):
        if coord_ref_point is not None and not isinstance(
            coord_ref_point, numbers.Real
        ):
            raise AssertionError(
                "Pixel coordinate of the reference point must be real floating type."
            )

    @ColumnAttribute("TCRVL")
    def coord_ref_value(col, coord_ref_value):
        if coord_ref_value is not None and not isinstance(
            coord_ref_value, numbers.Real
        ):
            raise AssertionError(
                "Coordinate value at reference point must be real floating type."
            )

    @ColumnAttribute("TCDLT")
    def coord_inc(col, coord_inc):
        if coord_inc is not None and not isinstance(coord_inc, numbers.Real):
            raise AssertionError("Coordinate increment must be real floating type.")

    @ColumnAttribute("TRPOS")
    def time_ref_pos(col, time_ref_pos):
        if time_ref_pos is not None and not isinstance(time_ref_pos, str):
            raise AssertionError("Time reference position must be a string.")

    format = ColumnAttribute("TFORM")
    unit = ColumnAttribute("TUNIT")
    null = ColumnAttribute("TNULL")
    bscale = ColumnAttribute("TSCAL")
    bzero = ColumnAttribute("TZERO")
    disp = ColumnAttribute("TDISP")
    start = ColumnAttribute("TBCOL")
    dim = ColumnAttribute("TDIM")

    @lazyproperty
    def ascii(self):
        """Whether this `Column` represents a column in an ASCII table."""

        return isinstance(self.format, _AsciiColumnFormat)

    @lazyproperty
    def dtype(self):
        return self.format.dtype

    def copy(self):
        """
        Return a copy of this `Column`.
        """
        tmp = Column(format="I")  # just use a throw-away format
        tmp.__dict__ = self.__dict__.copy()
        return tmp

    @staticmethod
    def _convert_format(format, cls):
        """The format argument to this class's initializer may come in many
        forms.  This uses the given column format class ``cls`` to convert
        to a format of that type.

        TODO: There should be an abc base class for column format classes
        """

        # Short circuit in case we're already a _BaseColumnFormat--there is at
        # least one case in which this can happen
        if isinstance(format, _BaseColumnFormat):
            return format, format.recformat

        if format in NUMPY2FITS:
            with suppress(VerifyError):
                # legit recarray format?
                recformat = format
                format = cls.from_recformat(format)

        try:
            # legit FITS format?
            format = cls(format)
            recformat = format.recformat
        except VerifyError:
            raise VerifyError(f"Illegal format `{format}`.")

        return format, recformat

    @classmethod
    def _verify_keywords(
        cls,
        name=None,
        format=None,
        unit=None,
        null=None,
        bscale=None,
        bzero=None,
        disp=None,
        start=None,
        dim=None,
        ascii=None,
        coord_type=None,
        coord_unit=None,
        coord_ref_point=None,
        coord_ref_value=None,
        coord_inc=None,
        time_ref_pos=None,
    ):
        """
        Given the keyword arguments used to initialize a Column, specifically
        those that typically read from a FITS header (so excluding array),
        verify that each keyword has a valid value.

        Returns a 2-tuple of dicts.  The first maps valid keywords to their
        values.  The second maps invalid keywords to a 2-tuple of their value,
        and a message explaining why they were found invalid.
        """

        valid = {}
        invalid = {}

        try:
            format, recformat = cls._determine_formats(format, start, dim, ascii)
            valid.update(format=format, recformat=recformat)
        except (ValueError, VerifyError) as err:
            msg = (
                f"Column format option (TFORMn) failed verification: {err!s} "
                "The invalid value will be ignored for the purpose of "
                "formatting the data in this column."
            )
            invalid["format"] = (format, msg)
        except AttributeError as err:
            msg = (
                "Column format option (TFORMn) must be a string with a valid "
                f"FITS table format (got {format!s}: {err!s}). "
                "The invalid value will be ignored for the purpose of "
                "formatting the data in this column."
            )
            invalid["format"] = (format, msg)

        # Currently we don't have any validation for name, unit, bscale, or
        # bzero so include those by default
        # TODO: Add validation for these keywords, obviously
        for k, v in [
            ("name", name),
            ("unit", unit),
            ("bscale", bscale),
            ("bzero", bzero),
        ]:
            if v is not None and v != "":
                valid[k] = v

        # Validate null option
        # Note: Enough code exists that thinks empty strings are sensible
        # inputs for these options that we need to treat '' as None
        if null is not None and null != "":
            msg = None
            if isinstance(format, _AsciiColumnFormat):
                null = str(null)
                if len(null) > format.width:
                    msg = (
                        "ASCII table null option (TNULLn) is longer than "
                        "the column's character width and will be truncated "
                        "(got {!r}).".format(null)
                    )
            else:
                tnull_formats = ("B", "I", "J", "K")

                if not _is_int(null):
                    # Make this an exception instead of a warning, since any
                    # non-int value is meaningless
                    msg = (
                        "Column null option (TNULLn) must be an integer for "
                        "binary table columns (got {!r}).  The invalid value "
                        "will be ignored for the purpose of formatting "
                        "the data in this column.".format(null)
                    )

                elif not (
                    format.format in tnull_formats
                    or (
                        format.format in ("P", "Q") and format.p_format in tnull_formats
                    )
                ):
                    # TODO: We should also check that TNULLn's integer value
                    # is in the range allowed by the column's format
                    msg = (
                        "Column null option (TNULLn) is invalid for binary "
                        "table columns of type {!r} (got {!r}).  The invalid "
                        "value will be ignored for the purpose of formatting "
                        "the data in this column.".format(format, null)
                    )

            if msg is None:
                valid["null"] = null
            else:
                invalid["null"] = (null, msg)

        # Validate the disp option
        # TODO: Add full parsing and validation of TDISPn keywords
        if disp is not None and disp != "":
            msg = None
            if not isinstance(disp, str):
                msg = (
                    "Column disp option (TDISPn) must be a string (got "
                    f"{disp!r}). The invalid value will be ignored for the "
                    "purpose of formatting the data in this column."
                )

            elif isinstance(format, _AsciiColumnFormat) and disp[0].upper() == "L":
                # disp is at least one character long and has the 'L' format
                # which is not recognized for ASCII tables
                msg = (
                    "Column disp option (TDISPn) may not use the 'L' format "
                    "with ASCII table columns.  The invalid value will be "
                    "ignored for the purpose of formatting the data in this "
                    "column."
                )

            if msg is None:
                try:
                    _parse_tdisp_format(disp)
                    valid["disp"] = disp
                except VerifyError as err:
                    msg = (
                        "Column disp option (TDISPn) failed verification: "
                        f"{err!s} The invalid value will be ignored for the "
                        "purpose of formatting the data in this column."
                    )
                    invalid["disp"] = (disp, msg)
            else:
                invalid["disp"] = (disp, msg)

        # Validate the start option
        if start is not None and start != "":
            msg = None
            if not isinstance(format, _AsciiColumnFormat):
                # The 'start' option only applies to ASCII columns
                msg = (
                    "Column start option (TBCOLn) is not allowed for binary "
                    "table columns (got {!r}).  The invalid keyword will be "
                    "ignored for the purpose of formatting the data in this "
                    "column.".format(start)
                )
            else:
                try:
                    start = int(start)
                except (TypeError, ValueError):
                    pass

                if not _is_int(start) or start < 1:
                    msg = (
                        "Column start option (TBCOLn) must be a positive integer "
                        "(got {!r}).  The invalid value will be ignored for the "
                        "purpose of formatting the data in this column.".format(start)
                    )

            if msg is None:
                valid["start"] = start
            else:
                invalid["start"] = (start, msg)

        # Process TDIMn options
        # ASCII table columns can't have a TDIMn keyword associated with it;
        # for now we just issue a warning and ignore it.
        # TODO: This should be checked by the FITS verification code
        if dim is not None and dim != "":
            msg = None
            dims_tuple = tuple()
            # NOTE: If valid, the dim keyword's value in the the valid dict is
            # a tuple, not the original string; if invalid just the original
            # string is returned
            if isinstance(format, _AsciiColumnFormat):
                msg = (
                    "Column dim option (TDIMn) is not allowed for ASCII table "
                    "columns (got {!r}).  The invalid keyword will be ignored "
                    "for the purpose of formatting this column.".format(dim)
                )

            elif isinstance(dim, str):
                dims_tuple = _parse_tdim(dim)
            elif isinstance(dim, tuple):
                dims_tuple = dim
            else:
                msg = (
                    "`dim` argument must be a string containing a valid value "
                    "for the TDIMn header keyword associated with this column, "
                    "or a tuple containing the C-order dimensions for the "
                    "column.  The invalid value will be ignored for the purpose "
                    "of formatting this column."
                )

            if dims_tuple:
                if isinstance(recformat, _FormatP):
                    # TDIMs have different meaning for VLA format,
                    # no warning should be thrown
                    msg = None
                elif reduce(operator.mul, dims_tuple) > format.repeat:
                    msg = (
                        "The repeat count of the column format {!r} for column {!r} "
                        "is fewer than the number of elements per the TDIM "
                        "argument {!r}.  The invalid TDIMn value will be ignored "
                        "for the purpose of formatting this column.".format(
                            name, format, dim
                        )
                    )

            if msg is None:
                valid["dim"] = dims_tuple
            else:
                invalid["dim"] = (dim, msg)

        if coord_type is not None and coord_type != "":
            msg = None
            if not isinstance(coord_type, str):
                msg = (
                    "Coordinate/axis type option (TCTYPn) must be a string "
                    "(got {!r}). The invalid keyword will be ignored for the "
                    "purpose of formatting this column.".format(coord_type)
                )
            elif len(coord_type) > 8:
                msg = (
                    "Coordinate/axis type option (TCTYPn) must be a string "
                    "of atmost 8 characters (got {!r}). The invalid keyword "
                    "will be ignored for the purpose of formatting this "
                    "column.".format(coord_type)
                )

            if msg is None:
                valid["coord_type"] = coord_type
            else:
                invalid["coord_type"] = (coord_type, msg)

        if coord_unit is not None and coord_unit != "":
            msg = None
            if not isinstance(coord_unit, str):
                msg = (
                    "Coordinate/axis unit option (TCUNIn) must be a string "
                    "(got {!r}). The invalid keyword will be ignored for the "
                    "purpose of formatting this column.".format(coord_unit)
                )

            if msg is None:
                valid["coord_unit"] = coord_unit
            else:
                invalid["coord_unit"] = (coord_unit, msg)

        for k, v in [
            ("coord_ref_point", coord_ref_point),
            ("coord_ref_value", coord_ref_value),
            ("coord_inc", coord_inc),
        ]:
            if v is not None and v != "":
                msg = None
                if not isinstance(v, numbers.Real):
                    msg = (
                        "Column {} option ({}n) must be a real floating type (got"
                        " {!r}). The invalid value will be ignored for the purpose of"
                        " formatting the data in this column.".format(
                            k, ATTRIBUTE_TO_KEYWORD[k], v
                        )
                    )

                if msg is None:
                    valid[k] = v
                else:
                    invalid[k] = (v, msg)

        if time_ref_pos is not None and time_ref_pos != "":
            msg = None
            if not isinstance(time_ref_pos, str):
                msg = (
                    "Time coordinate reference position option (TRPOSn) must be "
                    "a string (got {!r}). The invalid keyword will be ignored for "
                    "the purpose of formatting this column.".format(time_ref_pos)
                )

            if msg is None:
                valid["time_ref_pos"] = time_ref_pos
            else:
                invalid["time_ref_pos"] = (time_ref_pos, msg)

        return valid, invalid

    @classmethod
    def _determine_formats(cls, format, start, dim, ascii):
        """
        Given a format string and whether or not the Column is for an
        ASCII table (ascii=None means unspecified, but lean toward binary table
        where ambiguous) create an appropriate _BaseColumnFormat instance for
        the column's format, and determine the appropriate recarray format.

        The values of the start and dim keyword arguments are also useful, as
        the former is only valid for ASCII tables and the latter only for
        BINARY tables.
        """

        # If the given format string is unambiguously a Numpy dtype or one of
        # the Numpy record format type specifiers supported by Astropy then that
        # should take priority--otherwise assume it is a FITS format
        if isinstance(format, np.dtype):
            format, _, _ = _dtype_to_recformat(format)

        # check format
        if ascii is None and not isinstance(format, _BaseColumnFormat):
            # We're just give a string which could be either a Numpy format
            # code, or a format for a binary column array *or* a format for an
            # ASCII column array--there may be many ambiguities here.  Try our
            # best to guess what the user intended.
            format, recformat = cls._guess_format(format, start, dim)
        elif not ascii and not isinstance(format, _BaseColumnFormat):
            format, recformat = cls._convert_format(format, _ColumnFormat)
        elif ascii and not isinstance(format, _AsciiColumnFormat):
            format, recformat = cls._convert_format(format, _AsciiColumnFormat)
        else:
            # The format is already acceptable and unambiguous
            recformat = format.recformat

        return format, recformat

    @classmethod
    def _guess_format(cls, format, start, dim):
        if start and dim:
            # This is impossible; this can't be a valid FITS column
            raise ValueError(
                "Columns cannot have both a start (TCOLn) and dim "
                "(TDIMn) option, since the former is only applies to "
                "ASCII tables, and the latter is only valid for binary "
                "tables."
            )
        elif start:
            # Only ASCII table columns can have a 'start' option
            guess_format = _AsciiColumnFormat
        elif dim:
            # Only binary tables can have a dim option
            guess_format = _ColumnFormat
        else:
            # If the format is *technically* a valid binary column format
            # (i.e. it has a valid format code followed by arbitrary
            # "optional" codes), but it is also strictly a valid ASCII
            # table format, then assume an ASCII table column was being
            # requested (the more likely case, after all).
            with suppress(VerifyError):
                format = _AsciiColumnFormat(format, strict=True)

            # A safe guess which reflects the existing behavior of previous
            # Astropy versions
            guess_format = _ColumnFormat

        try:
            format, recformat = cls._convert_format(format, guess_format)
        except VerifyError:
            # For whatever reason our guess was wrong (for example if we got
            # just 'F' that's not a valid binary format, but it an ASCII format
            # code albeit with the width/precision omitted
            guess_format = (
                _AsciiColumnFormat if guess_format is _ColumnFormat else _ColumnFormat
            )
            # If this fails too we're out of options--it is truly an invalid
            # format, or at least not supported
            format, recformat = cls._convert_format(format, guess_format)

        return format, recformat

    def _convert_to_valid_data_type(self, array):
        # Convert the format to a type we understand
        if isinstance(array, Delayed):
            return array
        elif array is None:
            return array
        else:
            format = self.format
            dims = self._dims
            if dims and format.format not in "PQ":
                shape = dims[:-1] if "A" in format else dims
                shape = (len(array),) + shape
                array = array.reshape(shape)

            if "P" in format or "Q" in format:
                return array
            elif "A" in format:
                if array.dtype.char in "SU":
                    if dims:
                        # The 'last' dimension (first in the order given
                        # in the TDIMn keyword itself) is the number of
                        # characters in each string
                        fsize = dims[-1]
                    else:
                        fsize = np.dtype(format.recformat).itemsize
                    return chararray.array(array, itemsize=fsize, copy=False)
                else:
                    return _convert_array(array, np.dtype(format.recformat))
            elif "L" in format:
                # boolean needs to be scaled back to storage values ('T', 'F')
                if array.dtype == np.dtype("bool"):
                    return np.where(array == np.False_, ord("F"), ord("T"))
                else:
                    return np.where(array == 0, ord("F"), ord("T"))
            elif "X" in format:
                return _convert_array(array, np.dtype("uint8"))
            else:
                # Preserve byte order of the original array for now; see #77
                numpy_format = array.dtype.byteorder + format.recformat

                # Handle arrays passed in as unsigned ints as pseudo-unsigned
                # int arrays; blatantly tacked in here for now--we need columns
                # to have explicit knowledge of whether they treated as
                # pseudo-unsigned
                bzeros = {
                    2: np.uint16(2**15),
                    4: np.uint32(2**31),
                    8: np.uint64(2**63),
                }
                if (
                    array.dtype.kind == "u"
                    and array.dtype.itemsize in bzeros
                    and self.bscale in (1, None, "")
                    and self.bzero == bzeros[array.dtype.itemsize]
                ):
                    # Basically the array is uint, has scale == 1.0, and the
                    # bzero is the appropriate value for a pseudo-unsigned
                    # integer of the input dtype, then go ahead and assume that
                    # uint is assumed
                    numpy_format = numpy_format.replace("i", "u")
                    self._pseudo_unsigned_ints = True

                # The .base here means we're dropping the shape information,
                # which is only used to format recarray fields, and is not
                # useful for converting input arrays to the correct data type
                dtype = np.dtype(numpy_format).base

                return _convert_array(array, dtype)
