

from abc import ABCMeta, abstractmethod
import numpy as np
from astropy.units import Unit, UnitBase, UnitsError, UnitTypeError, UnitConversionError, dimensionless_unscaled, Quantity
__all__ = ['FunctionUnitBase', 'FunctionQuantity']
SUPPORTED_UFUNCS = set((getattr(np.core.umath, ufunc) for ufunc in ('isfinite', 'isinf', 'isnan', 'sign', 'signbit', 'rint', 'floor', 'ceil', 'trunc', '_ones_like', 'ones_like', 'positive') if hasattr(np.core.umath, ufunc)))
SUPPORTED_FUNCTIONS = set((getattr(np, function) for function in ('clip', 'trace', 'mean', 'min', 'max', 'round')))

class FunctionUnitBase:
    __array_priority__ = 30000
    __truediv__ = __div__
    __rtruediv__ = __rdiv__

    def to(self, other, value=1.0, equivalencies=[]):
        if other is self.physical_unit:
            return self.to_physical(value)
        other_function_unit = getattr(other, 'function_unit', other)
        if self.function_unit.is_equivalent(other_function_unit):
            other_physical_unit = getattr(other, 'physical_unit', dimensionless_unscaled)
            if self.physical_unit != other_physical_unit:
                value_other_physical = self.physical_unit.to(other_physical_unit, self.to_physical(value), equivalencies)
                value = self.from_physical(value_other_physical)
            return self.function_unit.to(other_function_unit, value)
        else:
            try:
                return self.physical_unit.to(other, self.to_physical(value), equivalencies)
            except UnitConversionError as e:
                if self.function_unit == Unit('mag'):
                    msg = 'Did you perhaps subtract magnitudes so the unit got lost?'
                    e.args += (msg,)
                    raise e
                else:
                    raise