import inspect
import operator
import textwrap
import warnings
import numpy as np
from ..utils.decorators import lazyproperty
from ..utils.exceptions import AstropyWarning
from ..utils.misc import isiterable, InheritDocstrings
from .utils import (is_effectively_unity, sanitize_scale, validate_power,
                    resolve_fractions)
from . import format as unit_format
from . import si
from . import cgs
from . import physical
from .quantity import Quantity
from .quantity import Quantity
from .quantity import Quantity
from .quantity import Quantity
from .quantity import Quantity

__all__ = [
    'UnitsError', 'UnitsWarning', 'UnitConversionError', 'UnitTypeError',
    'UnitBase', 'NamedUnit', 'IrreducibleUnit', 'Unit', 'CompositeUnit',
    'PrefixUnit', 'UnrecognizedUnit', 'def_unit', 'get_current_unit_registry',
    'set_enabled_units', 'add_enabled_units',
    'set_enabled_equivalencies', 'add_enabled_equivalencies',
    'dimensionless_unscaled', 'one']
UNITY = 1.0
_unit_registries = [_UnitRegistry()]
si_prefixes = [
    (['Y'], ['yotta'], 1e24),
    (['Z'], ['zetta'], 1e21),
    (['E'], ['exa'], 1e18),
    (['P'], ['peta'], 1e15),
    (['T'], ['tera'], 1e12),
    (['G'], ['giga'], 1e9),
    (['M'], ['mega'], 1e6),
    (['k'], ['kilo'], 1e3),
    (['h'], ['hecto'], 1e2),
    (['da'], ['deka', 'deca'], 1e1),
    (['d'], ['deci'], 1e-1),
    (['c'], ['centi'], 1e-2),
    (['m'], ['milli'], 1e-3),
    (['u'], ['micro'], 1e-6),
    (['n'], ['nano'], 1e-9),
    (['p'], ['pico'], 1e-12),
    (['f'], ['femto'], 1e-15),
    (['a'], ['atto'], 1e-18),
    (['z'], ['zepto'], 1e-21),
    (['y'], ['yocto'], 1e-24)
]
binary_prefixes = [
    (['Ki'], ['kibi'], 2. ** 10),
    (['Mi'], ['mebi'], 2. ** 20),
    (['Gi'], ['gibi'], 2. ** 30),
    (['Ti'], ['tebi'], 2. ** 40),
    (['Pi'], ['pebi'], 2. ** 50),
    (['Ei'], ['exbi'], 2. ** 60)
]
dimensionless_unscaled = CompositeUnit(1, [], [], _error_check=False)
one = dimensionless_unscaled
unit_format.fits.UnitScaleError = UnitScaleError

class UnitBase:
    __array_priority__ = 1000
    __truediv__ = __div__
    __rtruediv__ = __rdiv__
    def __div__(self, m):
        if isinstance(m, (bytes, str)):
            m = Unit(m)

        if isinstance(m, UnitBase):
            if m.is_unity():
                return self
            return CompositeUnit(1, [self, m], [1, -1], _error_check=False)

        try:
            # Cannot handle this as Unit, re-try as Quantity
            from .quantity import Quantity
            return Quantity(1, self) / m
        except TypeError:
            return NotImplemented
    def is_unity(self):
        """
        Returns `True` if the unit is unscaled and dimensionless.
        """
        return False