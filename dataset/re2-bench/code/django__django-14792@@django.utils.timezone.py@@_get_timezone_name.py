import functools
from contextlib import ContextDecorator
from datetime import datetime, timedelta, timezone, tzinfo
import pytz
from asgiref.local import Local
from django.conf import settings

__all__ = [
    'utc', 'get_fixed_timezone',
    'get_default_timezone', 'get_default_timezone_name',
    'get_current_timezone', 'get_current_timezone_name',
    'activate', 'deactivate', 'override',
    'localtime', 'now',
    'is_aware', 'is_naive', 'make_aware', 'make_naive',
]
utc = pytz.utc
_PYTZ_BASE_CLASSES = (pytz.tzinfo.BaseTzInfo, pytz._FixedOffset)
_active = Local()

def _get_timezone_name(timezone):
    """
    Return the offset for fixed offset timezones, or the name of timezone if
    not set.
    """
    return timezone.tzname(None) or str(timezone)
