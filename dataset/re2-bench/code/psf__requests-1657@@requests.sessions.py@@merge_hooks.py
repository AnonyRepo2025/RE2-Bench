import os
from collections import Mapping
from datetime import datetime
from .compat import cookielib, OrderedDict, urljoin, urlparse, urlunparse
from .cookies import cookiejar_from_dict, extract_cookies_to_jar, RequestsCookieJar
from .models import Request, PreparedRequest
from .hooks import default_hooks, dispatch_hook
from .utils import to_key_val_list, default_headers
from .exceptions import TooManyRedirects, InvalidSchema
from .structures import CaseInsensitiveDict
from .adapters import HTTPAdapter
from .utils import requote_uri, get_environ_proxies, get_netrc_auth
from .status_codes import codes

REDIRECT_STATI = (
    codes.moved, # 301
    codes.found, # 302
    codes.other, # 303
    codes.temporary_moved, # 307
)
DEFAULT_REDIRECT_LIMIT = 30

def merge_hooks(request_hooks, session_hooks, dict_class=OrderedDict):
    """
    Properly merges both requests and session hooks.

    This is necessary because when request_hooks == {'response': []}, the
    merge breaks Session hooks entirely.
    """
    if session_hooks is None or session_hooks.get('response') == []:
        return request_hooks

    if request_hooks is None or request_hooks.get('response') == []:
        return session_hooks

    return merge_setting(request_hooks, session_hooks, dict_class)
