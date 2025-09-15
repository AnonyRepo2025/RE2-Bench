from abc import ABCMeta, abstractmethod
import numpy as np
from astropy.units import (Unit, UnitBase, UnitsError, UnitTypeError, UnitConversionError,
                           dimensionless_unscaled, Quantity)

__all__ = ['FunctionUnitBase', 'FunctionQuantity']
SUPPORTED_UFUNCS = set(getattr(np.core.umath, ufunc) for ufunc in (
    'isfinite', 'isinf', 'isnan', 'sign', 'signbit',
    'rint', 'floor', 'ceil', 'trunc',
    '_ones_like', 'ones_like', 'positive') if hasattr(np.core.umath, ufunc))
SUPPORTED_FUNCTIONS = set(getattr(np, function) for function in
                          ('clip', 'trace', 'mean', 'min', 'max', 'round'))

class FunctionUnitBase:
    __array_priority__ = 30000
    __truediv__ = __div__
    __rtruediv__ = __rdiv__
    def to(self, other, value=1., equivalencies=[]):
        """
        Return the converted values in the specified unit.

        Parameters
        ----------
        other : `~astropy.units.Unit` object, `~astropy.units.function.FunctionUnitBase` object or string
            The unit to convert to.

        value : scalar int or float, or sequence convertible to array, optional
            Value(s) in the current unit to be converted to the specified unit.
            If not provided, defaults to 1.0.

        equivalencies : list of equivalence pairs, optional
            A list of equivalence pairs to try if the units are not
            directly convertible.  See :ref:`unit_equivalencies`.
            This list is in meant to treat only equivalencies between different
            physical units; the build-in equivalency between the function
            unit and the physical one is automatically taken into account.

        Returns
        -------
        values : scalar or array
            Converted value(s). Input value sequences are returned as
            numpy arrays.

        Raises
        ------
        UnitsError
            If units are inconsistent.
        """
        # conversion to one's own physical unit should be fastest
        if other is self.physical_unit:
            return self.to_physical(value)

        other_function_unit = getattr(other, 'function_unit', other)
        if self.function_unit.is_equivalent(other_function_unit):
            # when other is an equivalent function unit:
            # first convert physical units to other's physical units
            other_physical_unit = getattr(other, 'physical_unit',
                                          dimensionless_unscaled)
            if self.physical_unit != other_physical_unit:
                value_other_physical = self.physical_unit.to(
                    other_physical_unit, self.to_physical(value),
                    equivalencies)
                # make function unit again, in own system
                value = self.from_physical(value_other_physical)

            # convert possible difference in function unit (e.g., dex->dB)
            return self.function_unit.to(other_function_unit, value)

        else:
            try:
                # when other is not a function unit
                return self.physical_unit.to(other, self.to_physical(value),
                                             equivalencies)
            except UnitConversionError as e:
                if self.function_unit == Unit('mag'):
                    # One can get to raw magnitudes via math that strips the dimensions off.
                    # Include extra information in the exception to remind users of this.
                    msg = "Did you perhaps subtract magnitudes so the unit got lost?"
                    e.args += (msg,)
                    raise e
                else:
                    raise