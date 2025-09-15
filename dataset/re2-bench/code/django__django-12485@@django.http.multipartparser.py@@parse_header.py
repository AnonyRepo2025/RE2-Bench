import base64
import binascii
import cgi
import collections
import html
from urllib.parse import unquote
from django.conf import settings
from django.core.exceptions import (
    RequestDataTooBig, SuspiciousMultipartForm, TooManyFieldsSent,
)
from django.core.files.uploadhandler import (
    SkipFile, StopFutureHandlers, StopUpload,
)
from django.utils.datastructures import MultiValueDict
from django.utils.encoding import force_str
from django.http import QueryDict

__all__ = ('MultiPartParser', 'MultiPartParserError', 'InputStreamExhausted')
RAW = "raw"
FILE = "file"
FIELD = "field"

def parse_header(line):
    """
    Parse the header into a key-value.

    Input (line): bytes, output: str for key/name, bytes for values which
    will be decoded later.
    """
    plist = _parse_header_params(b';' + line)
    key = plist.pop(0).lower().decode('ascii')
    pdict = {}
    for p in plist:
        i = p.find(b'=')
        if i >= 0:
            has_encoding = False
            name = p[:i].strip().lower().decode('ascii')
            if name.endswith('*'):
                # Lang/encoding embedded in the value (like "filename*=UTF-8''file.ext")
                # http://tools.ietf.org/html/rfc2231#section-4
                name = name[:-1]
                if p.count(b"'") == 2:
                    has_encoding = True
            value = p[i + 1:].strip()
            if len(value) >= 2 and value[:1] == value[-1:] == b'"':
                value = value[1:-1]
                value = value.replace(b'\\\\', b'\\').replace(b'\\"', b'"')
            if has_encoding:
                encoding, lang, value = value.split(b"'")
                value = unquote(value.decode(), encoding=encoding.decode())
            pdict[name] = value
    return key, pdict
