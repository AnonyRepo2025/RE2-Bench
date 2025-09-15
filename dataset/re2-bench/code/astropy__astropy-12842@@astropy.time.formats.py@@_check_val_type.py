import fnmatch
import time
import re
import datetime
import warnings
from decimal import Decimal
from collections import OrderedDict, defaultdict
import numpy as np
import erfa
from astropy.utils.decorators import lazyproperty, classproperty
from astropy.utils.exceptions import AstropyDeprecationWarning
import astropy.units as u
from . import _parse_times
from . import utils
from .utils import day_frac, quantity_day_frac, two_sum, two_product
from . import conf
from .core import Time, TIME_SCALES, TIME_DELTA_SCALES, ScaleValueError  # noqa
from matplotlib.dates import get_epoch
from erfa import ErfaWarning

__all__ = ['TimeFormat', 'TimeJD', 'TimeMJD', 'TimeFromEpoch', 'TimeUnix',
           'TimeUnixTai', 'TimeCxcSec', 'TimeGPS', 'TimeDecimalYear',
           'TimePlotDate', 'TimeUnique', 'TimeDatetime', 'TimeString',
           'TimeISO', 'TimeISOT', 'TimeFITS', 'TimeYearDayTime',
           'TimeEpochDate', 'TimeBesselianEpoch', 'TimeJulianEpoch',
           'TimeDeltaFormat', 'TimeDeltaSec', 'TimeDeltaJD',
           'TimeEpochDateString', 'TimeBesselianEpochString',
           'TimeJulianEpochString', 'TIME_FORMATS', 'TIME_DELTA_FORMATS',
           'TimezoneInfo', 'TimeDeltaDatetime', 'TimeDatetime64', 'TimeYMDHMS',
           'TimeNumeric', 'TimeDeltaNumeric']
__doctest_skip__ = ['TimePlotDate']
TIME_FORMATS = OrderedDict()
TIME_DELTA_FORMATS = OrderedDict()
FITS_DEPRECATED_SCALES = {'TDT': 'tt', 'ET': 'tt',
                          'GMT': 'utc', 'UT': 'utc', 'IAT': 'tai'}

class TimeBesselianEpoch(TimeEpochDate):
    name = 'byear'
    epoch_to_jd = 'epb2jd'
    jd_to_epoch = 'epb'
    def _check_val_type(self, val1, val2):
        """Input value validation, typically overridden by derived classes"""
        if hasattr(val1, 'to') and hasattr(val1, 'unit') and val1.unit is not None:
            raise ValueError("Cannot use Quantities for 'byear' format, "
                             "as the interpretation would be ambiguous. "
                             "Use float with Besselian year instead. ")
        # FIXME: is val2 really okay here?
        return super()._check_val_type(val1, val2)