def typecast_timestamp(s):
    if not s:
        return None
    if ' ' not in s:
        return typecast_date(s)
    d, t = s.split()
    if '-' in t:
        t, _ = t.split('-', 1)
    elif '+' in t:
        t, _ = t.split('+', 1)
    dates = d.split('-')
    times = t.split(':')
    seconds = times[2]
    if '.' in seconds:
        seconds, microseconds = seconds.split('.')
    else:
        microseconds = '0'
    tzinfo = utc if settings.USE_TZ else None
    return datetime.datetime(int(dates[0]), int(dates[1]), int(dates[2]), int(times[0]), int(times[1]), int(seconds), int((microseconds + '000000')[:6]), tzinfo)

def typecast_date(s):
    return datetime.date(*map(int, s.split('-'))) if s else None

def localtime(value=None, timezone=None):
    if value is None:
        value = now()
    if timezone is None:
        timezone = get_current_timezone()
    if is_naive(value):
        raise ValueError('localtime() cannot be applied to a naive datetime')
    return value.astimezone(timezone)

def is_naive(value):
    return value.utcoffset() is None



import datetime
import decimal
import functools
import hashlib
import math
import operator
import re
import statistics
import warnings
from itertools import chain
from sqlite3 import dbapi2 as Database
import pytz
from django.core.exceptions import ImproperlyConfigured
from django.db import utils
from django.db.backends import utils as backend_utils
from django.db.backends.base.base import BaseDatabaseWrapper
from django.utils import timezone
from django.utils.dateparse import parse_datetime, parse_time
from django.utils.duration import duration_microseconds
from .client import DatabaseClient
from .creation import DatabaseCreation
from .features import DatabaseFeatures
from .introspection import DatabaseIntrospection
from .operations import DatabaseOperations
from .schema import DatabaseSchemaEditor
FORMAT_QMARK_REGEX = re.compile('(?<!%)%s')

def _sqlite_datetime_parse(dt, tzname=None, conn_tzname=None):
    if dt is None:
        return None
    try:
        dt = backend_utils.typecast_timestamp(dt)
    except (TypeError, ValueError):
        return None
    if conn_tzname:
        dt = dt.replace(tzinfo=pytz.timezone(conn_tzname))
    if tzname is not None and tzname != conn_tzname:
        sign_index = tzname.find('+') + tzname.find('-') + 1
        if sign_index > -1:
            sign = tzname[sign_index]
            tzname, offset = tzname.split(sign)
            if offset:
                hours, minutes = offset.split(':')
                offset_delta = datetime.timedelta(hours=int(hours), minutes=int(minutes))
                dt += offset_delta if sign == '+' else -offset_delta
        dt = timezone.localtime(dt, pytz.timezone(tzname))
    return dt