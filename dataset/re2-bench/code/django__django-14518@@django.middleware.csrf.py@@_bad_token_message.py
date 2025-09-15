import logging
import string
from collections import defaultdict
from urllib.parse import urlparse
from django.conf import settings
from django.core.exceptions import DisallowedHost, ImproperlyConfigured
from django.http.request import HttpHeaders
from django.urls import get_callable
from django.utils.cache import patch_vary_headers
from django.utils.crypto import constant_time_compare, get_random_string
from django.utils.deprecation import MiddlewareMixin
from django.utils.functional import cached_property
from django.utils.http import is_same_domain
from django.utils.log import log_response
from django.utils.regex_helper import _lazy_re_compile

logger = logging.getLogger('django.security.csrf')
invalid_token_chars_re = _lazy_re_compile('[^a-zA-Z0-9]')
REASON_BAD_ORIGIN = "Origin checking failed - %s does not match any trusted origins."
REASON_NO_REFERER = "Referer checking failed - no Referer."
REASON_BAD_REFERER = "Referer checking failed - %s does not match any trusted origins."
REASON_NO_CSRF_COOKIE = "CSRF cookie not set."
REASON_CSRF_TOKEN_MISSING = 'CSRF token missing.'
REASON_MALFORMED_REFERER = "Referer checking failed - Referer is malformed."
REASON_INSECURE_REFERER = "Referer checking failed - Referer is insecure while host is secure."
REASON_INCORRECT_LENGTH = 'has incorrect length'
REASON_INVALID_CHARACTERS = 'has invalid characters'
CSRF_SECRET_LENGTH = 32
CSRF_TOKEN_LENGTH = 2 * CSRF_SECRET_LENGTH
CSRF_ALLOWED_CHARS = string.ascii_letters + string.digits
CSRF_SESSION_KEY = '_csrftoken'

class CsrfViewMiddleware(MiddlewareMixin):
    def _bad_token_message(self, reason, token_source):
        if token_source != 'POST':
            # Assume it is a settings.CSRF_HEADER_NAME value.
            header_name = HttpHeaders.parse_header_name(token_source)
            token_source = f'the {header_name!r} HTTP header'
        return f'CSRF token from {token_source} {reason}.'