

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
from .core import Unit, UnitBase, UnitConversionError, UnitsError, UnitTypeError, dimensionless_unscaled, get_current_unit_registry
from .format import Base, Latex
from .quantity_helper import can_have_arbitrary_unit, check_output, converters_and_unit
from .quantity_helper.function_helpers import DISPATCHED_FUNCTIONS, FUNCTION_HELPERS, SUBCLASS_SAFE_FUNCTIONS, UNSUPPORTED_FUNCTIONS
from .structured import StructuredUnit, _structured_unit_like_dtype
from .utils import is_effectively_unity
from ._typing import HAS_ANNOTATED, Annotated
from astropy.units.physical import get_physical_type
__all__ = ['Quantity', 'SpecificTypeQuantity', 'QuantityInfoBase', 'QuantityInfo', 'allclose', 'isclose']
__doctest_skip__ = ['Quantity.*']
_UNIT_NOT_INITIALISED = '(Unit not initialised)'
_UFUNCS_FILTER_WARNINGS = {np.arcsin, np.arccos, np.arccosh, np.arctanh}
conf = Conf()

class Quantity(ndarray):
    _equivalencies = []
    _default_unit = dimensionless_unscaled
    _unit = None
    __array_priority__ = 10000
    info = QuantityInfo()
    value = property(to_value, doc='The numerical value of this instance.\n\n    See also\n    --------\n    to_value : Get the numerical value in a given unit.\n    ')
    _include_easy_conversion_members = False

    def _result_as_quantity(self, result, unit, out):
        if isinstance(result, (tuple, list)):
            if out is None:
                out = (None,) * len(result)
            return result.__class__((self._result_as_quantity(result_, unit_, out_) for result_, unit_, out_ in zip(result, unit, out)))
        if out is None:
            return result if unit is None else self._new_view(result, unit, finalize=False)
        elif isinstance(out, Quantity):
            out._set_unit(unit)
        return out

    def _set_unit(self, unit):
        if not isinstance(unit, UnitBase):
            if isinstance(self._unit, StructuredUnit) or isinstance(unit, StructuredUnit):
                unit = StructuredUnit(unit, self.dtype)
            else:
                unit = Unit(str(unit), parse_strict='silent')
                if not isinstance(unit, (UnitBase, StructuredUnit)):
                    raise UnitTypeError(f'{self.__class__.__name__} instances require normal units, not {unit.__class__} instances.')
        self._unit = unit