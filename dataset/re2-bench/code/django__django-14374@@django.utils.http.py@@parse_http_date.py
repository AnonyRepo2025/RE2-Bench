import base64
import datetime
import re
import unicodedata
from binascii import Error as BinasciiError
from email.utils import formatdate
from urllib.parse import (
    ParseResult, SplitResult, _coerce_args, _splitnetloc, _splitparams,
    scheme_chars, urlencode as original_urlencode, uses_params,
)
from django.utils.datastructures import MultiValueDict
from django.utils.regex_helper import _lazy_re_compile

ETAG_MATCH = _lazy_re_compile(r'''
    \A(      # start of string and capture group
    (?:W/)?  # optional weak indicator
    "        # opening quote
    [^"]*    # any sequence of non-quote characters
    "        # end quote
    )\Z      # end of string and capture group
''', re.X)
MONTHS = 'jan feb mar apr may jun jul aug sep oct nov dec'.split()
__D = r'(?P<day>\d{2})'
__D2 = r'(?P<day>[ \d]\d)'
__M = r'(?P<mon>\w{3})'
__Y = r'(?P<year>\d{4})'
__Y2 = r'(?P<year>\d{2})'
__T = r'(?P<hour>\d{2}):(?P<min>\d{2}):(?P<sec>\d{2})'
RFC1123_DATE = _lazy_re_compile(r'^\w{3}, %s %s %s %s GMT$' % (__D, __M, __Y, __T))
RFC850_DATE = _lazy_re_compile(r'^\w{6,9}, %s-%s-%s %s GMT$' % (__D, __M, __Y2, __T))
ASCTIME_DATE = _lazy_re_compile(r'^\w{3} %s %s %s %s$' % (__M, __D2, __T, __Y))
RFC3986_GENDELIMS = ":/?#[]@"
RFC3986_SUBDELIMS = "!$&'()*+,;="

def parse_http_date(date):
    """
    Parse a date format as specified by HTTP RFC7231 section 7.1.1.1.

    The three formats allowed by the RFC are accepted, even if only the first
    one is still in widespread use.

    Return an integer expressed in seconds since the epoch, in UTC.
    """
    # email.utils.parsedate() does the job for RFC1123 dates; unfortunately
    # RFC7231 makes it mandatory to support RFC850 dates too. So we roll
    # our own RFC-compliant parsing.
    for regex in RFC1123_DATE, RFC850_DATE, ASCTIME_DATE:
        m = regex.match(date)
        if m is not None:
            break
    else:
        raise ValueError("%r is not in a valid HTTP date format" % date)
    try:
        tz = datetime.timezone.utc
        year = int(m['year'])
        if year < 100:
            current_year = datetime.datetime.now(tz=tz).year
            current_century = current_year - (current_year % 100)
            if year - (current_year % 100) > 50:
                # year that appears to be more than 50 years in the future are
                # interpreted as representing the past.
                year += current_century - 100
            else:
                year += current_century
        month = MONTHS.index(m['mon'].lower()) + 1
        day = int(m['day'])
        hour = int(m['hour'])
        min = int(m['min'])
        sec = int(m['sec'])
        result = datetime.datetime(year, month, day, hour, min, sec, tzinfo=tz)
        return int(result.timestamp())
    except Exception as exc:
        raise ValueError("%r is not a valid date" % date) from exc
