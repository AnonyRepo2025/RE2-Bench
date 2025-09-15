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
from .client import DatabaseClient                          # isort:skip
from .creation import DatabaseCreation                      # isort:skip
from .features import DatabaseFeatures                      # isort:skip
from .introspection import DatabaseIntrospection            # isort:skip
from .operations import DatabaseOperations                  # isort:skip
from .schema import DatabaseSchemaEditor                    # isort:skip

FORMAT_QMARK_REGEX = re.compile(r'(?<!%)%s')

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
