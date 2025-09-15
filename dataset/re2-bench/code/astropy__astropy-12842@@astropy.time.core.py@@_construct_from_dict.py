import os
import copy
import enum
import operator
import threading
from datetime import datetime, date, timedelta
from time import strftime
from warnings import warn
import numpy as np
import erfa
from astropy import units as u, constants as const
from astropy.units import UnitConversionError
from astropy.utils import ShapedLikeNDArray
from astropy.utils.compat.misc import override__dir__
from astropy.utils.data_info import MixinInfo, data_info_factory
from astropy.utils.exceptions import AstropyDeprecationWarning, AstropyWarning
from .utils import day_frac
from .formats import (TIME_FORMATS, TIME_DELTA_FORMATS,
                      TimeJD, TimeUnique, TimeAstropyTime, TimeDatetime)
from .formats import TimeFromEpoch  # noqa
from astropy.extern import _strptime
from astropy.coordinates import (UnitSphericalRepresentation, CartesianRepresentation,
                                         HCRS, ICRS, GCRS, solar_system_ephemeris)
from astropy.coordinates import Longitude, EarthLocation
from astropy.coordinates.builtin_frames.utils import get_polar_motion
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.utils import iers
from datetime import datetime
from astropy.coordinates import EarthLocation
from astropy.utils.iers import earth_orientation_table
from astropy.utils.iers import earth_orientation_table

__all__ = ['TimeBase', 'Time', 'TimeDelta', 'TimeInfo', 'TimeInfoBase', 'update_leap_seconds',
           'TIME_SCALES', 'STANDARD_TIME_SCALES', 'TIME_DELTA_SCALES',
           'ScaleValueError', 'OperandTypeError', 'TimeDeltaMissingUnitWarning']
STANDARD_TIME_SCALES = ('tai', 'tcb', 'tcg', 'tdb', 'tt', 'ut1', 'utc')
LOCAL_SCALES = ('local',)
TIME_TYPES = dict((scale, scales) for scales in (STANDARD_TIME_SCALES, LOCAL_SCALES)
                  for scale in scales)
TIME_SCALES = STANDARD_TIME_SCALES + LOCAL_SCALES
MULTI_HOPS = {('tai', 'tcb'): ('tt', 'tdb'),
              ('tai', 'tcg'): ('tt',),
              ('tai', 'ut1'): ('utc',),
              ('tai', 'tdb'): ('tt',),
              ('tcb', 'tcg'): ('tdb', 'tt'),
              ('tcb', 'tt'): ('tdb',),
              ('tcb', 'ut1'): ('tdb', 'tt', 'tai', 'utc'),
              ('tcb', 'utc'): ('tdb', 'tt', 'tai'),
              ('tcg', 'tdb'): ('tt',),
              ('tcg', 'ut1'): ('tt', 'tai', 'utc'),
              ('tcg', 'utc'): ('tt', 'tai'),
              ('tdb', 'ut1'): ('tt', 'tai', 'utc'),
              ('tdb', 'utc'): ('tt', 'tai'),
              ('tt', 'ut1'): ('tai', 'utc'),
              ('tt', 'utc'): ('tai',),
              }
GEOCENTRIC_SCALES = ('tai', 'tt', 'tcg')
BARYCENTRIC_SCALES = ('tcb', 'tdb')
ROTATIONAL_SCALES = ('ut1',)
TIME_DELTA_TYPES = dict((scale, scales)
                        for scales in (GEOCENTRIC_SCALES, BARYCENTRIC_SCALES,
                                       ROTATIONAL_SCALES, LOCAL_SCALES) for scale in scales)
TIME_DELTA_SCALES = GEOCENTRIC_SCALES + BARYCENTRIC_SCALES + ROTATIONAL_SCALES + LOCAL_SCALES
SCALE_OFFSETS = {('tt', 'tai'): None,
                 ('tai', 'tt'): None,
                 ('tcg', 'tt'): -erfa.ELG,
                 ('tt', 'tcg'): erfa.ELG / (1. - erfa.ELG),
                 ('tcg', 'tai'): -erfa.ELG,
                 ('tai', 'tcg'): erfa.ELG / (1. - erfa.ELG),
                 ('tcb', 'tdb'): -erfa.ELB,
                 ('tdb', 'tcb'): erfa.ELB / (1. - erfa.ELB)}
SIDEREAL_TIME_MODELS = {
    'mean': {
        'IAU2006': {'function': erfa.gmst06, 'scales': ('ut1', 'tt')},
        'IAU2000': {'function': erfa.gmst00, 'scales': ('ut1', 'tt')},
        'IAU1982': {'function': erfa.gmst82, 'scales': ('ut1',), 'include_tio': False}
    },
    'apparent': {
        'IAU2006A': {'function': erfa.gst06a, 'scales': ('ut1', 'tt')},
        'IAU2000A': {'function': erfa.gst00a, 'scales': ('ut1', 'tt')},
        'IAU2000B': {'function': erfa.gst00b, 'scales': ('ut1',)},
        'IAU1994': {'function': erfa.gst94, 'scales': ('ut1',), 'include_tio': False}
    }}
_LEAP_SECONDS_CHECK = _LeapSecondsCheck.NOT_STARTED
_LEAP_SECONDS_LOCK = threading.RLock()

class TimeInfoBase(MixinInfo):
    attr_names = MixinInfo.attr_names | {'serialize_method'}
    _supports_indexing = True
    _represent_as_dict_extra_attrs = ('format', 'scale', 'precision',
                                      'in_subfmt', 'out_subfmt', 'location',
                                      '_delta_ut1_utc', '_delta_tdb_tt')
    _represent_as_dict_primary_data = 'value'
    mask_val = np.ma.masked
    info_summary_stats = staticmethod(
        data_info_factory(names=MixinInfo._stats,
                          funcs=[getattr(np, stat) for stat in MixinInfo._stats]))
    def _construct_from_dict(self, map):
        if 'jd1' in map and 'jd2' in map:
            # Initialize as JD but revert to desired format and out_subfmt (if needed)
            format = map.pop('format')
            out_subfmt = map.pop('out_subfmt', None)
            map['format'] = 'jd'
            map['val'] = map.pop('jd1')
            map['val2'] = map.pop('jd2')
            out = self._parent_cls(**map)
            out.format = format
            if out_subfmt is not None:
                out.out_subfmt = out_subfmt

        else:
            map['val'] = map.pop('value')
            out = self._parent_cls(**map)

        return out