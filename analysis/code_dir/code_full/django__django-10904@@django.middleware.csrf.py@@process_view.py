def is_secure(self):
    return self.scheme == 'https'

def scheme(self):
    if settings.SECURE_PROXY_SSL_HEADER:
        try:
            header, value = settings.SECURE_PROXY_SSL_HEADER
        except ValueError:
            raise ImproperlyConfigured('The SECURE_PROXY_SSL_HEADER setting must be a tuple containing two values.')
        if self.META.get(header) == value:
            return 'https'
    return self._get_scheme()

def _get_scheme(self):
    return self.environ.get('wsgi.url_scheme')

def get(self, key, default=None):
    try:
        val = self[key]
    except KeyError:
        return default
    if val == []:
        return default
    return val

def __getitem__(self, key):
    try:
        list_ = super().__getitem__(key)
    except KeyError:
        raise MultiValueDictKeyError(key)
    try:
        return list_[-1]
    except IndexError:
        return []

def _sanitize_token(token):
    if re.search('[^a-zA-Z0-9]', token):
        return _get_new_csrf_token()
    elif len(token) == CSRF_TOKEN_LENGTH:
        return token
    elif len(token) == CSRF_SECRET_LENGTH:
        return _salt_cipher_secret(token)
    return _get_new_csrf_token()

def _salt_cipher_secret(secret):
    salt = _get_new_csrf_string()
    chars = CSRF_ALLOWED_CHARS
    pairs = zip((chars.index(x) for x in secret), (chars.index(x) for x in salt))
    cipher = ''.join((chars[(x + y) % len(chars)] for x, y in pairs))
    return salt + cipher

def _get_new_csrf_string():
    return get_random_string(CSRF_SECRET_LENGTH, allowed_chars=CSRF_ALLOWED_CHARS)

def get_random_string(length=12, allowed_chars='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
    if not using_sysrandom:
        random.seed(hashlib.sha256(('%s%s%s' % (random.getstate(), time.time(), settings.SECRET_KEY)).encode()).digest())
    return ''.join((random.choice(allowed_chars) for i in range(length)))

def _compare_salted_tokens(request_csrf_token, csrf_token):
    return constant_time_compare(_unsalt_cipher_token(request_csrf_token), _unsalt_cipher_token(csrf_token))

def _unsalt_cipher_token(token):
    salt = token[:CSRF_SECRET_LENGTH]
    token = token[CSRF_SECRET_LENGTH:]
    chars = CSRF_ALLOWED_CHARS
    pairs = zip((chars.index(x) for x in token), (chars.index(x) for x in salt))
    secret = ''.join((chars[x - y] for x, y in pairs))
    return secret

def constant_time_compare(val1, val2):
    return hmac.compare_digest(force_bytes(val1), force_bytes(val2))

def force_bytes(s, encoding='utf-8', strings_only=False, errors='strict'):
    if isinstance(s, bytes):
        if encoding == 'utf-8':
            return s
        else:
            return s.decode('utf-8', errors).encode(encoding, errors)
    if strings_only and is_protected_type(s):
        return s
    if isinstance(s, memoryview):
        return bytes(s)
    return str(s).encode(encoding, errors)

def __getattr__(self, name):
    if self._wrapped is empty:
        self._setup(name)
    val = getattr(self._wrapped, name)
    self.__dict__[name] = val
    return val

def _get_new_csrf_token():
    return _salt_cipher_secret(_get_new_csrf_string())



import logging
import re
import string
from urllib.parse import urlparse
from django.conf import settings
from django.core.exceptions import DisallowedHost, ImproperlyConfigured
from django.urls import get_callable
from django.utils.cache import patch_vary_headers
from django.utils.crypto import constant_time_compare, get_random_string
from django.utils.deprecation import MiddlewareMixin
from django.utils.http import is_same_domain
from django.utils.log import log_response
logger = logging.getLogger('django.security.csrf')
REASON_NO_REFERER = 'Referer checking failed - no Referer.'
REASON_BAD_REFERER = 'Referer checking failed - %s does not match any trusted origins.'
REASON_NO_CSRF_COOKIE = 'CSRF cookie not set.'
REASON_BAD_TOKEN = 'CSRF token missing or incorrect.'
REASON_MALFORMED_REFERER = 'Referer checking failed - Referer is malformed.'
REASON_INSECURE_REFERER = 'Referer checking failed - Referer is insecure while host is secure.'
CSRF_SECRET_LENGTH = 32
CSRF_TOKEN_LENGTH = 2 * CSRF_SECRET_LENGTH
CSRF_ALLOWED_CHARS = string.ascii_letters + string.digits
CSRF_SESSION_KEY = '_csrftoken'

class CsrfViewMiddleware(MiddlewareMixin):

    def process_view(self, request, callback, callback_args, callback_kwargs):
        if getattr(request, 'csrf_processing_done', False):
            return None
        if getattr(callback, 'csrf_exempt', False):
            return None
        if request.method not in ('GET', 'HEAD', 'OPTIONS', 'TRACE'):
            if getattr(request, '_dont_enforce_csrf_checks', False):
                return self._accept(request)
            if request.is_secure():
                referer = request.META.get('HTTP_REFERER')
                if referer is None:
                    return self._reject(request, REASON_NO_REFERER)
                referer = urlparse(referer)
                if '' in (referer.scheme, referer.netloc):
                    return self._reject(request, REASON_MALFORMED_REFERER)
                if referer.scheme != 'https':
                    return self._reject(request, REASON_INSECURE_REFERER)
                good_referer = settings.SESSION_COOKIE_DOMAIN if settings.CSRF_USE_SESSIONS else settings.CSRF_COOKIE_DOMAIN
                if good_referer is not None:
                    server_port = request.get_port()
                    if server_port not in ('443', '80'):
                        good_referer = '%s:%s' % (good_referer, server_port)
                else:
                    try:
                        good_referer = request.get_host()
                    except DisallowedHost:
                        pass
                good_hosts = list(settings.CSRF_TRUSTED_ORIGINS)
                if good_referer is not None:
                    good_hosts.append(good_referer)
                if not any((is_same_domain(referer.netloc, host) for host in good_hosts)):
                    reason = REASON_BAD_REFERER % referer.geturl()
                    return self._reject(request, reason)
            csrf_token = request.META.get('CSRF_COOKIE')
            if csrf_token is None:
                return self._reject(request, REASON_NO_CSRF_COOKIE)
            request_csrf_token = ''
            if request.method == 'POST':
                try:
                    request_csrf_token = request.POST.get('csrfmiddlewaretoken', '')
                except OSError:
                    pass
            if request_csrf_token == '':
                request_csrf_token = request.META.get(settings.CSRF_HEADER_NAME, '')
            request_csrf_token = _sanitize_token(request_csrf_token)
            if not _compare_salted_tokens(request_csrf_token, csrf_token):
                return self._reject(request, REASON_BAD_TOKEN)
        return self._accept(request)