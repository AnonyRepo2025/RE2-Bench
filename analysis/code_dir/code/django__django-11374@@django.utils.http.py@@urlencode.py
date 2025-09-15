import base64
import calendar
import datetime
import re
import unicodedata
import warnings
from binascii import Error as BinasciiError
from email.utils import formatdate
from urllib.parse import ParseResult, SplitResult, _coerce_args, _splitnetloc, _splitparams, quote, quote_plus, scheme_chars, unquote, unquote_plus, urlencode as original_urlencode, uses_params
from django.core.exceptions import TooManyFieldsSent
from django.utils.datastructures import MultiValueDict
from django.utils.deprecation import RemovedInDjango40Warning
from django.utils.functional import keep_lazy_text
ETAG_MATCH = re.compile('\n    \\A(      # start of string and capture group\n    (?:W/)?  # optional weak indicator\n    "        # opening quote\n    [^"]*    # any sequence of non-quote characters\n    "        # end quote\n    )\\Z      # end of string and capture group\n', re.X)
MONTHS = 'jan feb mar apr may jun jul aug sep oct nov dec'.split()
__D = '(?P<day>\\d{2})'
__D2 = '(?P<day>[ \\d]\\d)'
__M = '(?P<mon>\\w{3})'
__Y = '(?P<year>\\d{4})'
__Y2 = '(?P<year>\\d{2})'
__T = '(?P<hour>\\d{2}):(?P<min>\\d{2}):(?P<sec>\\d{2})'
RFC1123_DATE = re.compile('^\\w{3}, %s %s %s %s GMT$' % (__D, __M, __Y, __T))
RFC850_DATE = re.compile('^\\w{6,9}, %s-%s-%s %s GMT$' % (__D, __M, __Y2, __T))
ASCTIME_DATE = re.compile('^\\w{3} %s %s %s %s$' % (__M, __D2, __T, __Y))
RFC3986_GENDELIMS = ':/?#[]@'
RFC3986_SUBDELIMS = "!$&'()*+,;="
FIELDS_MATCH = re.compile('[&;]')

def urlencode(query, doseq=False):
    if isinstance(query, MultiValueDict):
        query = query.lists()
    elif hasattr(query, 'items'):
        query = query.items()
    query_params = []
    for key, value in query:
        if value is None:
            raise TypeError('Cannot encode None in a query string. Did you mean to pass an empty string or omit the value?')
        elif isinstance(value, (str, bytes)) or not doseq:
            query_val = value
        else:
            try:
                itr = iter(value)
            except TypeError:
                query_val = value
            else:
                query_val = []
                for item in itr:
                    if item is None:
                        raise TypeError('Cannot encode None in a query string. Did you mean to pass an empty string or omit the value?')
                    elif not isinstance(item, bytes):
                        item = str(item)
                    query_val.append(item)
        query_params.append((key, query_val))
    return original_urlencode(query_params, doseq)