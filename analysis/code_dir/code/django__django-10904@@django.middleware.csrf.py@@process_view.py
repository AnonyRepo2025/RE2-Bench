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