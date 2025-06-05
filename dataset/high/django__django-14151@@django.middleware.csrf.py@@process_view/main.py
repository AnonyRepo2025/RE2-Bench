import logging
import re
import string
from collections import defaultdict
from urllib.parse import urlparse
from django.conf import settings
from django.core.exceptions import DisallowedHost, ImproperlyConfigured
from django.urls import get_callable
from django.utils.cache import patch_vary_headers
from django.utils.crypto import constant_time_compare, get_random_string
from django.utils.deprecation import MiddlewareMixin
from django.utils.functional import cached_property
from django.utils.http import is_same_domain
from django.utils.log import log_response

logger = logging.getLogger('django.security.csrf')
REASON_BAD_ORIGIN = "Origin checking failed - %s does not match any trusted origins."
REASON_NO_REFERER = "Referer checking failed - no Referer."
REASON_BAD_REFERER = "Referer checking failed - %s does not match any trusted origins."
REASON_NO_CSRF_COOKIE = "CSRF cookie not set."
REASON_BAD_TOKEN = "CSRF token missing or incorrect."
REASON_MALFORMED_REFERER = "Referer checking failed - Referer is malformed."
REASON_INSECURE_REFERER = "Referer checking failed - Referer is insecure while host is secure."
CSRF_SECRET_LENGTH = 32
CSRF_TOKEN_LENGTH = 2 * CSRF_SECRET_LENGTH
CSRF_ALLOWED_CHARS = string.ascii_letters + string.digits
CSRF_SESSION_KEY = '_csrftoken'

class CsrfViewMiddleware(MiddlewareMixin):
    @cached_property
    def csrf_trusted_origins_hosts(self):
        return [
            urlparse(origin).netloc.lstrip('*')
            for origin in settings.CSRF_TRUSTED_ORIGINS
        ]
    def _get_token(self, request):
        if settings.CSRF_USE_SESSIONS:
            try:
                return request.session.get(CSRF_SESSION_KEY)
            except AttributeError:
                raise ImproperlyConfigured(
                    'CSRF_USE_SESSIONS is enabled, but request.session is not '
                    'set. SessionMiddleware must appear before CsrfViewMiddleware '
                    'in MIDDLEWARE.'
                )
        else:
            try:
                cookie_token = request.COOKIES[settings.CSRF_COOKIE_NAME]
            except KeyError:
                return None

            csrf_token = _sanitize_token(cookie_token)
            if csrf_token != cookie_token:
                request.csrf_cookie_needs_reset = True
            return csrf_token
    def _origin_verified(self, request):
        request_origin = request.META['HTTP_ORIGIN']
        good_origin = '%s://%s' % (
            'https' if request.is_secure() else 'http',
            request.get_host(),
        )
        if request_origin == good_origin:
            return True
        if request_origin in self.allowed_origins_exact:
            return True
        try:
            parsed_origin = urlparse(request_origin)
        except ValueError:
            return False
        request_scheme = parsed_origin.scheme
        request_netloc = parsed_origin.netloc
        return any(
            is_same_domain(request_netloc, host)
            for host in self.allowed_origin_subdomains.get(request_scheme, ())
        )
    def process_view(self, request, callback, callback_args, callback_kwargs):
        if getattr(request, 'csrf_processing_done', False):
            return None

        if getattr(callback, 'csrf_exempt', False):
            return None

        if request.method not in ('GET', 'HEAD', 'OPTIONS', 'TRACE'):
            if getattr(request, '_dont_enforce_csrf_checks', False):
                return self._accept(request)

            if 'HTTP_ORIGIN' in request.META:
                if not self._origin_verified(request):
                    return self._reject(request, REASON_BAD_ORIGIN % request.META['HTTP_ORIGIN'])
            elif request.is_secure():
                referer = request.META.get('HTTP_REFERER')
                if referer is None:
                    return self._reject(request, REASON_NO_REFERER)

                try:
                    referer = urlparse(referer)
                except ValueError:
                    return self._reject(request, REASON_MALFORMED_REFERER)

                if '' in (referer.scheme, referer.netloc):
                    return self._reject(request, REASON_MALFORMED_REFERER)

                if referer.scheme != 'https':
                    return self._reject(request, REASON_INSECURE_REFERER)

                good_referer = (
                    settings.SESSION_COOKIE_DOMAIN
                    if settings.CSRF_USE_SESSIONS
                    else settings.CSRF_COOKIE_DOMAIN
                )
                if good_referer is not None:
                    server_port = request.get_port()
                    if server_port not in ('443', '80'):
                        good_referer = '%s:%s' % (good_referer, server_port)
                else:
                    try:
                        good_referer = request.get_host()
                    except DisallowedHost:
                        pass

                good_hosts = list(self.csrf_trusted_origins_hosts)
                if good_referer is not None:
                    good_hosts.append(good_referer)

                if not any(is_same_domain(referer.netloc, host) for host in good_hosts):
                    reason = REASON_BAD_REFERER % referer.geturl()
                    return self._reject(request, reason)

            csrf_token = self._get_token(request)
            if csrf_token is None:
                return self._reject(request, REASON_NO_CSRF_COOKIE)

            request_csrf_token = ""
            if request.method == "POST":
                try:
                    request_csrf_token = request.POST.get('csrfmiddlewaretoken', '')
                except OSError:
                    pass

            if request_csrf_token == "":
                request_csrf_token = request.META.get(settings.CSRF_HEADER_NAME, '')

            request_csrf_token = _sanitize_token(request_csrf_token)
            if not _compare_masked_tokens(request_csrf_token, csrf_token):
                return self._reject(request, REASON_BAD_TOKEN)

        return self._accept(request)