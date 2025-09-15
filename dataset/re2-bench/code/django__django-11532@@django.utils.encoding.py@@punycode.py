import codecs
import datetime
import locale
import warnings
from decimal import Decimal
from urllib.parse import quote
from django.utils.deprecation import RemovedInDjango40Warning
from django.utils.functional import Promise

_PROTECTED_TYPES = (
    type(None), int, float, Decimal, datetime.datetime, datetime.date, datetime.time,
)
_ascii_ranges = [[45, 46, 95, 126], range(65, 91), range(97, 123)]
_hextobyte = {
    (fmt % char).encode(): bytes((char,))
    for ascii_range in _ascii_ranges
    for char in ascii_range
    for fmt in ['%02x', '%02X']
}
_hexdig = '0123456789ABCDEFabcdef'
DEFAULT_LOCALE_ENCODING = get_system_encoding()

def punycode(domain):
    """Return the Punycode of the given domain if it's non-ASCII."""
    return domain.encode('idna').decode('ascii')
