import re
import numbers
from fractions import Fraction
import warnings
import numpy as np
from .core import (Unit, dimensionless_unscaled, get_current_unit_registry,
                   UnitBase, UnitsError, UnitConversionError, UnitTypeError)
from .utils import is_effectively_unity
from .format.latex import Latex
from astropy.utils.compat import NUMPY_LT_1_14, NUMPY_LT_1_16, NUMPY_LT_1_17
from astropy.utils.compat.misc import override__dir__
from astropy.utils.exceptions import AstropyDeprecationWarning, AstropyWarning
from astropy.utils.misc import isiterable, InheritDocstrings
from astropy.utils.data_info import ParentDtypeInfo
from astropy import config as _config
from .quantity_helper import (converters_and_unit, can_have_arbitrary_unit,
                              check_output)

__all__ = ["Quantity", "SpecificTypeQuantity",
           "QuantityInfoBase", "QuantityInfo", "allclose", "isclose"]
__doctest_skip__ = ['Quantity.*']
_UNIT_NOT_INITIALISED = "(Unit not initialised)"
_UFUNCS_FILTER_WARNINGS = {np.arcsin, np.arccos, np.arccosh, np.arctanh}
conf = Conf()

class Quantity(ndarray):
    _equivalencies = []
    _default_unit = dimensionless_unscaled
    _unit = None
    __array_priority__ = 10000
    info = QuantityInfo()
    value = property(to_value,
                     doc="""The numerical value of this instance.

    See also
    --------
    to_value : Get the numerical value in a given unit.
    """)
    _include_easy_conversion_members = False
    def __array_ufunc__(self, function, method, *inputs, **kwargs):
        """Wrap numpy ufuncs, taking care of units.

        Parameters
        ----------
        function : callable
            ufunc to wrap.
        method : str
            Ufunc method: ``__call__``, ``at``, ``reduce``, etc.
        inputs : tuple
            Input arrays.
        kwargs : keyword arguments
            As passed on, with ``out`` containing possible quantity output.

        Returns
        -------
        result : `~astropy.units.Quantity`
            Results of the ufunc, with the unit set properly.
        """
        # Determine required conversion functions -- to bring the unit of the
        # input to that expected (e.g., radian for np.sin), or to get
        # consistent units between two inputs (e.g., in np.add) --
        # and the unit of the result (or tuple of units for nout > 1).
        converters, unit = converters_and_unit(function, method, *inputs)

        out = kwargs.get('out', None)
        # Avoid loop back by turning any Quantity output into array views.
        if out is not None:
            # If pre-allocated output is used, check it is suitable.
            # This also returns array view, to ensure we don't loop back.
            if function.nout == 1:
                out = out[0]
            out_array = check_output(out, unit, inputs, function=function)
            # Ensure output argument remains a tuple.
            kwargs['out'] = (out_array,) if function.nout == 1 else out_array

        # Same for inputs, but here also convert if necessary.
        arrays = []
        for input_, converter in zip(inputs, converters):
            input_ = getattr(input_, 'value', input_)
            arrays.append(converter(input_) if converter else input_)

        # Call our superclass's __array_ufunc__
        result = super().__array_ufunc__(function, method, *arrays, **kwargs)
        # If unit is None, a plain array is expected (e.g., comparisons), which
        # means we're done.
        # We're also done if the result was None (for method 'at') or
        # NotImplemented, which can happen if other inputs/outputs override
        # __array_ufunc__; hopefully, they can then deal with us.
        if unit is None or result is None or result is NotImplemented:
            return result

        return self._result_as_quantity(result, unit, out)
    def to_value(self, unit=None, equivalencies=[]):
        """
        The numerical value, possibly in a different unit.

        Parameters
        ----------
        unit : `~astropy.units.UnitBase` instance or str, optional
            The unit in which the value should be given. If not given or `None`,
            use the current unit.

        equivalencies : list of equivalence pairs, optional
            A list of equivalence pairs to try if the units are not directly
            convertible (see :ref:`unit_equivalencies`). If not provided or
            ``[]``, class default equivalencies will be used (none for
            `~astropy.units.Quantity`, but may be set for subclasses).
            If `None`, no equivalencies will be applied at all, not even any
            set globally or within a context.

        Returns
        -------
        value : `~numpy.ndarray` or scalar
            The value in the units specified. For arrays, this will be a view
            of the data if no unit conversion was necessary.

        See also
        --------
        to : Get a new instance in a different unit.
        """
        if unit is None or unit is self.unit:
            value = self.view(np.ndarray)
        else:
            unit = Unit(unit)
            # We want a view if the unit does not change.  One could check
            # with "==", but that calculates the scale that we need anyway.
            # TODO: would be better for `unit.to` to have an in-place flag.
            try:
                scale = self.unit._to(unit)
            except Exception:
                # Short-cut failed; try default (maybe equivalencies help).
                value = self._to_value(unit, equivalencies)
            else:
                value = self.view(np.ndarray)
                if not is_effectively_unity(scale):
                    # not in-place!
                    value = value * scale

        return value if self.shape else (value[()] if self.dtype.fields
                                         else value.item())
    @property
    def unit(self):
        """
        A `~astropy.units.UnitBase` object representing the unit of this
        quantity.
        """

        return self._unit