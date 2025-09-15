import warnings
from collections import namedtuple
import numpy as np
from . import angle_formats as form
from astropy import units as u
from astropy.utils import isiterable

__all__ = ['Angle', 'Latitude', 'Longitude']
hms_tuple = namedtuple('hms_tuple', ('h', 'm', 's'))
dms_tuple = namedtuple('dms_tuple', ('d', 'm', 's'))
signed_dms_tuple = namedtuple('signed_dms_tuple', ('sign', 'd', 'm', 's'))

class Angle(u.SpecificTypeQuantity):
    """
    One or more angular value(s) with units equivalent to radians or degrees.

    An angle can be specified either as an array, scalar, tuple (see
    below), string, `~astropy.units.Quantity` or another
    :class:`~astropy.coordinates.Angle`.

    The input parser is flexible and supports a variety of formats.
    The examples below illustrate common ways of initializing an `Angle`
    object. First some imports::

      >>> from astropy.coordinates import Angle
      >>> from astropy import units as u

    The angle values can now be provided::

      >>> Angle('10.2345d')
      <Angle 10.2345 deg>
      >>> Angle(['10.2345d', '-20d'])
      <Angle [ 10.2345, -20.    ] deg>
      >>> Angle('1:2:30.43 degrees')
      <Angle 1.04178611 deg>
      >>> Angle('1 2 0 hours')
      <Angle 1.03333333 hourangle>
      >>> Angle(np.arange(1, 8), unit=u.deg)
      <Angle [1., 2., 3., 4., 5., 6., 7.] deg>
      >>> Angle('1°2′3″')
      <Angle 1.03416667 deg>
      >>> Angle('1°2′3″N')
      <Angle 1.03416667 deg>
      >>> Angle('1d2m3.4s')
      <Angle 1.03427778 deg>
      >>> Angle('1d2m3.4sS')
      <Angle -1.03427778 deg>
      >>> Angle('-1h2m3s')
      <Angle -1.03416667 hourangle>
      >>> Angle('-1h2m3sE')
      <Angle -1.03416667 hourangle>
      >>> Angle('-1h2.5m')
      <Angle -1.04166667 hourangle>
      >>> Angle('-1h2.5mW')
      <Angle 1.04166667 hourangle>
      >>> Angle('-1:2.5', unit=u.deg)
      <Angle -1.04166667 deg>
      >>> Angle(10.2345 * u.deg)
      <Angle 10.2345 deg>
      >>> Angle(Angle(10.2345 * u.deg))
      <Angle 10.2345 deg>

    Parameters
    ----------
    angle : `~numpy.array`, scalar, `~astropy.units.Quantity`, :class:`~astropy.coordinates.Angle`
        The angle value. If a tuple, will be interpreted as ``(h, m,
        s)`` or ``(d, m, s)`` depending on ``unit``. If a string, it
        will be interpreted following the rules described above.

        If ``angle`` is a sequence or array of strings, the resulting
        values will be in the given ``unit``, or if `None` is provided,
        the unit will be taken from the first given value.

    unit : unit-like, optional
        The unit of the value specified for the angle.  This may be
        any string that `~astropy.units.Unit` understands, but it is
        better to give an actual unit object.  Must be an angular
        unit.

    dtype : `~numpy.dtype`, optional
        See `~astropy.units.Quantity`.

    copy : bool, optional
        See `~astropy.units.Quantity`.

    Raises
    ------
    `~astropy.units.UnitsError`
        If a unit is not provided or it is not an angular unit.
    """
    _equivalent_unit = u.radian
    _include_easy_conversion_members = True

    def __new__(cls, angle, unit=None, dtype=None, copy=True, **kwargs):

        if not isinstance(angle, u.Quantity):
            if unit is not None:
                unit = cls._convert_unit_to_angle_unit(u.Unit(unit))

            if isinstance(angle, tuple):
                angle = cls._tuple_to_float(angle, unit)

            elif isinstance(angle, str):
                angle, angle_unit = form.parse_angle(angle, unit)
                if angle_unit is None:
                    angle_unit = unit

                if isinstance(angle, tuple):
                    if angle_unit == u.hourangle:
                        form._check_hour_range(angle[0])
                    form._check_minute_range(angle[1])
                    a = np.abs(angle[0]) + angle[1] / 60.
                    if len(angle) == 3:
                        form._check_second_range(angle[2])
                        a += angle[2] / 3600.

                    angle = np.copysign(a, angle[0])

                if angle_unit is not unit:
                    # Possible conversion to `unit` will be done below.
                    angle = u.Quantity(angle, angle_unit, copy=False)

            elif (isiterable(angle) and
                  not (isinstance(angle, np.ndarray) and
                       angle.dtype.kind not in 'SUVO')):
                angle = [Angle(x, unit, copy=False) for x in angle]

        return super().__new__(cls, angle, unit, dtype=dtype, copy=copy,
                               **kwargs)

    @staticmethod
    def _tuple_to_float(angle, unit):
        """
        Converts an angle represented as a 3-tuple or 2-tuple into a floating
        point number in the given unit.
        """
        # TODO: Numpy array of tuples?
        if unit == u.hourangle:
            return form.hms_to_hours(*angle)
        elif unit == u.degree:
            return form.dms_to_degrees(*angle)
        else:
            raise u.UnitsError(f"Can not parse '{angle}' as unit '{unit}'")

    @staticmethod
    def _convert_unit_to_angle_unit(unit):
        return u.hourangle if unit is u.hour else unit

    def _set_unit(self, unit):
        super()._set_unit(self._convert_unit_to_angle_unit(unit))

    @property
    def hour(self):
        """
        The angle's value in hours (read-only property).
        """
        return self.hourangle

    @property
    def hms(self):
        """
        The angle's value in hours, as a named tuple with ``(h, m, s)``
        members.  (This is a read-only property.)
        """
        return hms_tuple(*form.hours_to_hms(self.hourangle))

    @property
    def dms(self):
        """
        The angle's value in degrees, as a named tuple with ``(d, m, s)``
        members.  (This is a read-only property.)
        """
        return dms_tuple(*form.degrees_to_dms(self.degree))

    @property
    def signed_dms(self):
        """
        The angle's value in degrees, as a named tuple with ``(sign, d, m, s)``
        members.  The ``d``, ``m``, ``s`` are thus always positive, and the sign of
        the angle is given by ``sign``. (This is a read-only property.)

        This is primarily intended for use with `dms` to generate string
        representations of coordinates that are correct for negative angles.
        """
        return signed_dms_tuple(np.sign(self.degree),
                                *form.degrees_to_dms(np.abs(self.degree)))

    def to_string(self, unit=None, decimal=False, sep='fromunit',
                  precision=None, alwayssign=False, pad=False,
                  fields=3, format=None):
        """ A string representation of the angle.

        Parameters
        ----------
        unit : `~astropy.units.UnitBase`, optional
            Specifies the unit.  Must be an angular unit.  If not
            provided, the unit used to initialize the angle will be
            used.

        decimal : bool, optional
            If `True`, a decimal representation will be used, otherwise
            the returned string will be in sexagesimal form.

        sep : str, optional
            The separator between numbers in a sexagesimal
            representation.  E.g., if it is ':', the result is
            ``'12:41:11.1241'``. Also accepts 2 or 3 separators. E.g.,
            ``sep='hms'`` would give the result ``'12h41m11.1241s'``, or
            sep='-:' would yield ``'11-21:17.124'``.  Alternatively, the
            special string 'fromunit' means 'dms' if the unit is
            degrees, or 'hms' if the unit is hours.

        precision : int, optional
            The level of decimal precision.  If ``decimal`` is `True`,
            this is the raw precision, otherwise it gives the
            precision of the last place of the sexagesimal
            representation (seconds).  If `None`, or not provided, the
            number of decimal places is determined by the value, and
            will be between 0-8 decimal places as required.

        alwayssign : bool, optional
            If `True`, include the sign no matter what.  If `False`,
            only include the sign if it is negative.

        pad : bool, optional
            If `True`, include leading zeros when needed to ensure a
            fixed number of characters for sexagesimal representation.

        fields : int, optional
            Specifies the number of fields to display when outputting
            sexagesimal notation.  For example:

                - fields == 1: ``'5d'``
                - fields == 2: ``'5d45m'``
                - fields == 3: ``'5d45m32.5s'``

            By default, all fields are displayed.

        format : str, optional
            The format of the result.  If not provided, an unadorned
            string is returned.  Supported values are:

            - 'latex': Return a LaTeX-formatted string

            - 'latex_inline': Return a LaTeX-formatted string which is the
              same as with ``format='latex'`` for |Angle| instances

            - 'unicode': Return a string containing non-ASCII unicode
              characters, such as the degree symbol

        Returns
        -------
        strrepr : str or array
            A string representation of the angle. If the angle is an array, this
            will be an array with a unicode dtype.

        """
        if unit is None:
            unit = self.unit
        else:
            unit = self._convert_unit_to_angle_unit(u.Unit(unit))

        separators = {
            None: {
                u.degree: 'dms',
                u.hourangle: 'hms'},
            'latex': {
                u.degree: [r'^\circ', r'{}^\prime', r'{}^{\prime\prime}'],
                u.hourangle: [r'^{\mathrm{h}}', r'^{\mathrm{m}}', r'^{\mathrm{s}}']},
            'unicode': {
                u.degree: '°′″',
                u.hourangle: 'ʰᵐˢ'}
        }
        # 'latex_inline' provides no functionality beyond what 'latex' offers,
        # but it should be implemented to avoid ValueErrors in user code.
        separators['latex_inline'] = separators['latex']

        if sep == 'fromunit':
            if format not in separators:
                raise ValueError(f"Unknown format '{format}'")
            seps = separators[format]
            if unit in seps:
                sep = seps[unit]

        # Create an iterator so we can format each element of what
        # might be an array.
        if unit is u.degree:
            if decimal:
                values = self.degree
                if precision is not None:
                    func = ("{0:0." + str(precision) + "f}").format
                else:
                    func = '{:g}'.format
            else:
                if sep == 'fromunit':
                    sep = 'dms'
                values = self.degree
                func = lambda x: form.degrees_to_string(
                    x, precision=precision, sep=sep, pad=pad,
                    fields=fields)

        elif unit is u.hourangle:
            if decimal:
                values = self.hour
                if precision is not None:
                    func = ("{0:0." + str(precision) + "f}").format
                else:
                    func = '{:g}'.format
            else:
                if sep == 'fromunit':
                    sep = 'hms'
                values = self.hour
                func = lambda x: form.hours_to_string(
                    x, precision=precision, sep=sep, pad=pad,
                    fields=fields)

        elif unit.is_equivalent(u.radian):
            if decimal:
                values = self.to_value(unit)
                if precision is not None:
                    func = ("{0:1." + str(precision) + "f}").format
                else:
                    func = "{:g}".format
            elif sep == 'fromunit':
                values = self.to_value(unit)
                unit_string = unit.to_string(format=format)
                if format == 'latex' or format == 'latex_inline':
                    unit_string = unit_string[1:-1]

                if precision is not None:
                    def plain_unit_format(val):
                        return ("{0:0." + str(precision) + "f}{1}").format(
                            val, unit_string)
                    func = plain_unit_format
                else:
                    def plain_unit_format(val):
                        return f"{val:g}{unit_string}"
                    func = plain_unit_format
            else:
                raise ValueError(
                    f"'{unit.name}' can not be represented in sexagesimal notation")

        else:
            raise u.UnitsError(
                "The unit value provided is not an angular unit.")

        def do_format(val):
            # Check if value is not nan to avoid ValueErrors when turning it into
            # a hexagesimal string.
            if not np.isnan(val):
                s = func(float(val))
                if alwayssign and not s.startswith('-'):
                    s = '+' + s
                if format == 'latex' or format == 'latex_inline':
                    s = f'${s}$'
                return s
            s = f"{val}"
            return s

        format_ufunc = np.vectorize(do_format, otypes=['U'])
        result = format_ufunc(values)

        if result.ndim == 0:
            result = result[()]
        return result

    def _wrap_at(self, wrap_angle):
        """
        Implementation that assumes ``angle`` is already validated
        and that wrapping is inplace.
        """
        # Convert the wrap angle and 360 degrees to the native unit of
        # this Angle, then do all the math on raw Numpy arrays rather
        # than Quantity objects for speed.
        a360 = u.degree.to(self.unit, 360.0)
        wrap_angle = wrap_angle.to_value(self.unit)
        wrap_angle_floor = wrap_angle - a360
        self_angle = self.view(np.ndarray)
        # Do the wrapping, but only if any angles need to be wrapped
        #
        # This invalid catch block is needed both for the floor division
        # and for the comparisons later on (latter not really needed
        # any more for >= 1.19 (NUMPY_LT_1_19), but former is).
        with np.errstate(invalid='ignore'):
            wraps = (self_angle - wrap_angle_floor) // a360
            np.nan_to_num(wraps, copy=False)
            if np.any(wraps != 0):
                self_angle -= wraps*a360
                # Rounding errors can cause problems.
                self_angle[self_angle >= wrap_angle] -= a360
                self_angle[self_angle < wrap_angle_floor] += a360

    def wrap_at(self, wrap_angle, inplace=False):
        """
        Wrap the `~astropy.coordinates.Angle` object at the given ``wrap_angle``.

        This method forces all the angle values to be within a contiguous
        360 degree range so that ``wrap_angle - 360d <= angle <
        wrap_angle``. By default a new Angle object is returned, but if the
        ``inplace`` argument is `True` then the `~astropy.coordinates.Angle`
        object is wrapped in place and nothing is returned.

        For instance::

          >>> from astropy.coordinates import Angle
          >>> import astropy.units as u
          >>> a = Angle([-20.0, 150.0, 350.0] * u.deg)

          >>> a.wrap_at(360 * u.deg).degree  # Wrap into range 0 to 360 degrees  # doctest: +FLOAT_CMP
          array([340., 150., 350.])

          >>> a.wrap_at('180d', inplace=True)  # Wrap into range -180 to 180 degrees  # doctest: +FLOAT_CMP
          >>> a.degree  # doctest: +FLOAT_CMP
          array([-20., 150., -10.])

        Parameters
        ----------
        wrap_angle : angle-like
            Specifies a single value for the wrap angle.  This can be any
            object that can initialize an `~astropy.coordinates.Angle` object,
            e.g. ``'180d'``, ``180 * u.deg``, or ``Angle(180, unit=u.deg)``.

        inplace : bool
            If `True` then wrap the object in place instead of returning
            a new `~astropy.coordinates.Angle`

        Returns
        -------
        out : Angle or None
            If ``inplace is False`` (default), return new
            `~astropy.coordinates.Angle` object with angles wrapped accordingly.
            Otherwise wrap in place and return `None`.
        """
        wrap_angle = Angle(wrap_angle, copy=False)  # Convert to an Angle
        if not inplace:
            self = self.copy()
        self._wrap_at(wrap_angle)
        return None if inplace else self

    def is_within_bounds(self, lower=None, upper=None):
        """
        Check if all angle(s) satisfy ``lower <= angle < upper``

        If ``lower`` is not specified (or `None`) then no lower bounds check is
        performed.  Likewise ``upper`` can be left unspecified.  For example::

          >>> from astropy.coordinates import Angle
          >>> import astropy.units as u
          >>> a = Angle([-20, 150, 350] * u.deg)
          >>> a.is_within_bounds('0d', '360d')
          False
          >>> a.is_within_bounds(None, '360d')
          True
          >>> a.is_within_bounds(-30 * u.deg, None)
          True

        Parameters
        ----------
        lower : angle-like or None
            Specifies lower bound for checking.  This can be any object
            that can initialize an `~astropy.coordinates.Angle` object, e.g. ``'180d'``,
            ``180 * u.deg``, or ``Angle(180, unit=u.deg)``.
        upper : angle-like or None
            Specifies upper bound for checking.  This can be any object
            that can initialize an `~astropy.coordinates.Angle` object, e.g. ``'180d'``,
            ``180 * u.deg``, or ``Angle(180, unit=u.deg)``.

        Returns
        -------
        is_within_bounds : bool
            `True` if all angles satisfy ``lower <= angle < upper``
        """
        ok = True
        if lower is not None:
            ok &= np.all(Angle(lower) <= self)
        if ok and upper is not None:
            ok &= np.all(self < Angle(upper))
        return bool(ok)

    def _str_helper(self, format=None):
        if self.isscalar:
            return self.to_string(format=format)

        def formatter(x):
            return x.to_string(format=format)

        return np.array2string(self, formatter={'all': formatter})

    def __str__(self):
        return self._str_helper()

    def _repr_latex_(self):
        return self._str_helper(format='latex')
