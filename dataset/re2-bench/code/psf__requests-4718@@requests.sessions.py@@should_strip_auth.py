import os
import sys
import time
from datetime import timedelta
from .auth import _basic_auth_str
from .compat import cookielib, is_py3, OrderedDict, urljoin, urlparse, Mapping
from .cookies import (
    cookiejar_from_dict, extract_cookies_to_jar, RequestsCookieJar, merge_cookies)
from .models import Request, PreparedRequest, DEFAULT_REDIRECT_LIMIT
from .hooks import default_hooks, dispatch_hook
from ._internal_utils import to_native_string
from .utils import to_key_val_list, default_headers
from .exceptions import (
    TooManyRedirects, InvalidSchema, ChunkedEncodingError, ContentDecodingError)
from .structures import CaseInsensitiveDict
from .adapters import HTTPAdapter
from .utils import (
    requote_uri, get_environ_proxies, get_netrc_auth, should_bypass_proxies,
    get_auth_from_url, rewind_body
)
from .status_codes import codes
from .models import REDIRECT_STATI



class SessionRedirectMixin(object):
    def should_strip_auth(self, old_url, new_url):
        """Decide whether Authorization header should be removed when redirecting"""
        old_parsed = urlparse(old_url)
        new_parsed = urlparse(new_url)
        if old_parsed.hostname != new_parsed.hostname:
            return True
        # Special case: allow http -> https redirect when using the standard
        # ports. This isn't specified by RFC 7235, but is kept to avoid
        # breaking backwards compatibility with older versions of requests
        # that allowed any redirects on the same host.
        if (old_parsed.scheme == 'http' and old_parsed.port in (80, None)
                and new_parsed.scheme == 'https' and new_parsed.port in (443, None)):
            return False
        # Standard case: root URI must match
        return old_parsed.port != new_parsed.port or old_parsed.scheme != new_parsed.scheme