import cgi
import codecs
import collections
import io
import os
import platform
import re
import sys
from . import __version__
from . import certs
from .compat import parse_http_list as _parse_list_header
from .compat import (quote, urlparse, bytes, str, OrderedDict, unquote, is_py2,
                     builtin_str, getproxies, proxy_bypass)
from .cookies import RequestsCookieJar, cookiejar_from_dict
from .structures import CaseInsensitiveDict
from .exceptions import MissingSchema, InvalidURL
from netrc import netrc, NetrcParseError

_hush_pyflakes = (RequestsCookieJar,)
NETRC_FILES = ('.netrc', '_netrc')
DEFAULT_CA_BUNDLE_PATH = certs.where()
UNRESERVED_SET = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    + "0123456789-._~")
_null = '\x00'.encode('ascii')  # encoding to ASCII for Python 3
_null2 = _null * 2
_null3 = _null * 3

def get_auth_from_url(url):
    """Given a url with authentication components, extract them into a tuple of
    username,password."""
    if url:
        url = unquote(url)
        parsed = urlparse(url)
        return (parsed.username, parsed.password)
    else:
        return ('', '')
