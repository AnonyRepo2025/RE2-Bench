import cgi
import codecs
import collections
import io
import os
import platform
import re
import sys
import socket
import struct
import warnings
from . import __version__
from . import certs
from .compat import parse_http_list as _parse_list_header
from .compat import (quote, urlparse, bytes, str, OrderedDict, unquote, is_py2,
                     builtin_str, getproxies, proxy_bypass, urlunparse,
                     basestring, is_py3)
from .cookies import RequestsCookieJar, cookiejar_from_dict
from .structures import CaseInsensitiveDict
from .exceptions import InvalidURL, FileModeWarning
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

def unquote_unreserved(uri):
    """Un-escape any percent-escape sequences in a URI that are unreserved
    characters. This leaves all reserved, illegal and non-ASCII bytes encoded.
    """
    # This convert function is used to optionally convert the output of `chr`.
    # In Python 3, `chr` returns a unicode string, while in Python 2 it returns
    # a bytestring. Here we deal with that by optionally converting.
    def convert(is_bytes, c):
        if is_py2 and not is_bytes:
            return c.decode('ascii')
        elif is_py3 and is_bytes:
            return c.encode('ascii')
        else:
            return c

    # Handle both bytestrings and unicode strings.
    is_bytes = isinstance(uri, bytes)
    splitchar = u'%'
    base = u''
    if is_bytes:
        splitchar = splitchar.encode('ascii')
        base = base.encode('ascii')

    parts = uri.split(splitchar)
    for i in range(1, len(parts)):
        h = parts[i][0:2]
        if len(h) == 2 and h.isalnum():
            try:
                c = chr(int(h, 16))
            except ValueError:
                raise InvalidURL("Invalid percent-escape sequence: '%s'" % h)

            if c in UNRESERVED_SET:
                parts[i] = convert(is_bytes, c) + parts[i][2:]
            else:
                parts[i] = splitchar + parts[i]
        else:
            parts[i] = splitchar + parts[i]
    return base.join(parts)
