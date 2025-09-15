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
    def __mul__(self, other):
        """Multiplication between `Quantity` objects and other objects."""

        if isinstance(other, (UnitBase, str)):
            try:
                return self._new_view(
                    self.value.copy(), other * self.unit, finalize=False
                )
            except UnitsError:  # let other try to deal with it
                return NotImplemented

        return super().__mul__(other)