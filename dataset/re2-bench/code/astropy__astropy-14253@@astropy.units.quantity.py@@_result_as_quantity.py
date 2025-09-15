import numbers
import operator
import re
import warnings
from fractions import Fraction
import numpy as np
from astropy import config as _config
from astropy.utils.compat import NUMPY_LT_1_22
from astropy.utils.data_info import ParentDtypeInfo
from astropy.utils.exceptions import AstropyWarning
from astropy.utils.misc import isiterable
from .core import (
    Unit,
    UnitBase,
    UnitConversionError,
    UnitsError,
    UnitTypeError,
    dimensionless_unscaled,
    get_current_unit_registry,
)
from .format import Base, Latex
from .quantity_helper import can_have_arbitrary_unit, check_output, converters_and_unit
from .quantity_helper.function_helpers import (
    DISPATCHED_FUNCTIONS,
    FUNCTION_HELPERS,
    SUBCLASS_SAFE_FUNCTIONS,
    UNSUPPORTED_FUNCTIONS,
)
from .structured import StructuredUnit, _structured_unit_like_dtype
from .utils import is_effectively_unity
from ._typing import HAS_ANNOTATED, Annotated
from astropy.units.physical import get_physical_type

__all__ = [
    "Quantity",
    "SpecificTypeQuantity",
    "QuantityInfoBase",
    "QuantityInfo",
    "allclose",
    "isclose",
]
__doctest_skip__ = ["Quantity.*"]
_UNIT_NOT_INITIALISED = "(Unit not initialised)"
_UFUNCS_FILTER_WARNINGS = {np.arcsin, np.arccos, np.arccosh, np.arctanh}
conf = Conf()

class Quantity(ndarray):
    _equivalencies = []
    _default_unit = dimensionless_unscaled
    _unit = None
    __array_priority__ = 10000
    info = QuantityInfo()
    value = property(
        to_value,
        doc="""The numerical value of this instance.

    See also
    --------
    to_value : Get the numerical value in a given unit.
    """,
    )
    _include_easy_conversion_members = False
    def _result_as_quantity(self, result, unit, out):
        """Turn result into a quantity with the given unit.

        If no output is given, it will take a view of the array as a quantity,
        and set the unit.  If output is given, those should be quantity views
        of the result arrays, and the function will just set the unit.

        Parameters
        ----------
        result : ndarray or tuple thereof
            Array(s) which need to be turned into quantity.
        unit : `~astropy.units.Unit`
            Unit for the quantities to be returned (or `None` if the result
            should not be a quantity).  Should be tuple if result is a tuple.
        out : `~astropy.units.Quantity` or None
            Possible output quantity. Should be `None` or a tuple if result
            is a tuple.

        Returns
        -------
        out : `~astropy.units.Quantity`
           With units set.
        """
        if isinstance(result, (tuple, list)):
            if out is None:
                out = (None,) * len(result)
            return result.__class__(
                self._result_as_quantity(result_, unit_, out_)
                for (result_, unit_, out_) in zip(result, unit, out)
            )

        if out is None:
            # View the result array as a Quantity with the proper unit.
            return (
                result if unit is None else self._new_view(result, unit, finalize=False)
            )

        elif isinstance(out, Quantity):
            # For given Quantity output, just set the unit. We know the unit
            # is not None and the output is of the correct Quantity subclass,
            # as it was passed through check_output.
            # (We cannot do this unconditionally, though, since it is possible
            # for out to be ndarray and the unit to be dimensionless.)
            out._set_unit(unit)

        return out
    def _set_unit(self, unit):
        """Set the unit.

        This is used anywhere the unit is set or modified, i.e., in the
        initilizer, in ``__imul__`` and ``__itruediv__`` for in-place
        multiplication and division by another unit, as well as in
        ``__array_finalize__`` for wrapping up views.  For Quantity, it just
        sets the unit, but subclasses can override it to check that, e.g.,
        a unit is consistent.
        """
        if not isinstance(unit, UnitBase):
            if isinstance(self._unit, StructuredUnit) or isinstance(
                unit, StructuredUnit
            ):
                unit = StructuredUnit(unit, self.dtype)
            else:
                # Trying to go through a string ensures that, e.g., Magnitudes with
                # dimensionless physical unit become Quantity with units of mag.
                unit = Unit(str(unit), parse_strict="silent")
                if not isinstance(unit, (UnitBase, StructuredUnit)):
                    raise UnitTypeError(
                        f"{self.__class__.__name__} instances require normal units, "
                        f"not {unit.__class__} instances."
                    )

        self._unit = unit