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
    def __array_finalize__(self, obj):
        # Check whether super().__array_finalize should be called
        # (sadly, ndarray.__array_finalize__ is None; we cannot be sure
        # what is above us).
        super_array_finalize = super().__array_finalize__
        if super_array_finalize is not None:
            super_array_finalize(obj)

        # If we're a new object or viewing an ndarray, nothing has to be done.
        if obj is None or obj.__class__ is np.ndarray:
            return

        # If our unit is not set and obj has a valid one, use it.
        if self._unit is None:
            unit = getattr(obj, "_unit", None)
            if unit is not None:
                self._set_unit(unit)

        # Copy info if the original had `info` defined.  Because of the way the
        # DataInfo works, `'info' in obj.__dict__` is False until the
        # `info` attribute is accessed or set.
        if "info" in obj.__dict__:
            self.info = obj.info
    def __quantity_subclass__(self, unit):
        """
        Overridden by subclasses to change what kind of view is
        created based on the output unit of an operation.

        Parameters
        ----------
        unit : UnitBase
            The unit for which the appropriate class should be returned

        Returns
        -------
        tuple :
            - `~astropy.units.Quantity` subclass
            - bool: True if subclasses of the given class are ok
        """
        return Quantity, True
    def _new_view(self, obj=None, unit=None, finalize=True):
        """Create a Quantity view of some array-like input, and set the unit

        By default, return a view of ``obj`` of the same class as ``self`` and
        with the same unit.  Subclasses can override the type of class for a
        given unit using ``__quantity_subclass__``, and can ensure properties
        other than the unit are copied using ``__array_finalize__``.

        If the given unit defines a ``_quantity_class`` of which ``self``
        is not an instance, a view using this class is taken.

        Parameters
        ----------
        obj : ndarray or scalar, optional
            The array to create a view of.  If obj is a numpy or python scalar,
            it will be converted to an array scalar.  By default, ``self``
            is converted.

        unit : unit-like, optional
            The unit of the resulting object.  It is used to select a
            subclass, and explicitly assigned to the view if given.
            If not given, the subclass and unit will be that of ``self``.

        finalize : bool, optional
            Whether to call ``__array_finalize__`` to transfer properties from
            ``self`` to the new view of ``obj`` (e.g., ``info`` for all
            subclasses, or ``_wrap_angle`` for `~astropy.coordinates.Latitude`).
            Default: `True`, as appropriate for, e.g., unit conversions or slicing,
            where the nature of the object does not change.

        Returns
        -------
        view : `~astropy.units.Quantity` subclass

        """
        # Determine the unit and quantity subclass that we need for the view.
        if unit is None:
            unit = self.unit
            quantity_subclass = self.__class__
        elif unit is self.unit and self.__class__ is Quantity:
            # The second part is because we should not presume what other
            # classes want to do for the same unit.  E.g., Constant will
            # always want to fall back to Quantity, and relies on going
            # through `__quantity_subclass__`.
            quantity_subclass = Quantity
        else:
            unit = Unit(unit)
            quantity_subclass = getattr(unit, "_quantity_class", Quantity)
            if isinstance(self, quantity_subclass):
                quantity_subclass, subok = self.__quantity_subclass__(unit)
                if subok:
                    quantity_subclass = self.__class__

        # We only want to propagate information from ``self`` to our new view,
        # so obj should be a regular array.  By using ``np.array``, we also
        # convert python and numpy scalars, which cannot be viewed as arrays
        # and thus not as Quantity either, to zero-dimensional arrays.
        # (These are turned back into scalar in `.value`)
        # Note that for an ndarray input, the np.array call takes only double
        # ``obj.__class is np.ndarray``. So, not worth special-casing.
        if obj is None:
            obj = self.view(np.ndarray)
        else:
            obj = np.array(obj, copy=False, subok=True)

        # Take the view, set the unit, and update possible other properties
        # such as ``info``, ``wrap_angle`` in `Longitude`, etc.
        view = obj.view(quantity_subclass)
        view._set_unit(unit)
        if finalize:
            view.__array_finalize__(self)
        return view
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
    def to_value(self, unit=None, equivalencies=[]):
        """
        The numerical value, possibly in a different unit.

        Parameters
        ----------
        unit : unit-like, optional
            The unit in which the value should be given. If not given or `None`,
            use the current unit.

        equivalencies : list of tuple, optional
            A list of equivalence pairs to try if the units are not directly
            convertible (see :ref:`astropy:unit_equivalencies`). If not provided
            or ``[]``, class default equivalencies will be used (none for
            `~astropy.units.Quantity`, but may be set for subclasses).
            If `None`, no equivalencies will be applied at all, not even any
            set globally or within a context.

        Returns
        -------
        value : ndarray or scalar
            The value in the units specified. For arrays, this will be a view
            of the data if no unit conversion was necessary.

        See also
        --------
        to : Get a new instance in a different unit.
        """
        if unit is None or unit is self.unit:
            value = self.view(np.ndarray)
        elif not self.dtype.names:
            # For non-structured, we attempt a short-cut, where we just get
            # the scale.  If that is 1, we do not have to do anything.
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
        else:
            # For structured arrays, we go the default route.
            value = self._to_value(unit, equivalencies)

        # Index with empty tuple to decay array scalars in to numpy scalars.
        return value if value.shape else value[()]
    @property
    def unit(self):
        """
        A `~astropy.units.UnitBase` object representing the unit of this
        quantity.
        """

        return self._unit
    def __getattr__(self, attr):
        """
        Quantities are able to directly convert to other units that
        have the same physical type.
        """
        if not self._include_easy_conversion_members:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no '{attr}' member"
            )

        def get_virtual_unit_attribute():
            registry = get_current_unit_registry().registry
            to_unit = registry.get(attr, None)
            if to_unit is None:
                return None

            try:
                return self.unit.to(
                    to_unit, self.value, equivalencies=self.equivalencies
                )
            except UnitsError:
                return None

        value = get_virtual_unit_attribute()

        if value is None:
            raise AttributeError(
                f"{self.__class__.__name__} instance has no attribute '{attr}'"
            )
        else:
            return value