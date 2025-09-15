import cgi
import codecs
import copy
from io import BytesIO
from itertools import chain
from urllib.parse import quote, urlencode, urljoin, urlsplit
from django.conf import settings
from django.core import signing
from django.core.exceptions import (
    DisallowedHost, ImproperlyConfigured, RequestDataTooBig,
)
from django.core.files import uploadhandler
from django.http.multipartparser import MultiPartParser, MultiPartParserError
from django.utils.datastructures import (
    CaseInsensitiveMapping, ImmutableList, MultiValueDict,
)
from django.utils.encoding import escape_uri_path, iri_to_uri
from django.utils.functional import cached_property
from django.utils.http import is_same_domain, limited_parse_qsl
from django.utils.regex_helper import _lazy_re_compile

RAISE_ERROR = object()
host_validation_re = _lazy_re_compile(r"^([a-z0-9.-]+|\[[a-f0-9]*:[a-f0-9\.:]+\])(:\d+)?$")

class HttpRequest:
    _encoding = None
    _upload_handlers = []
    def _get_raw_host(self):
        """
        Return the HTTP host using the environment or request headers. Skip
        allowed hosts protection, so may return an insecure host.
        """
        # We try three options, in order of decreasing preference.
        if settings.USE_X_FORWARDED_HOST and (
                'HTTP_X_FORWARDED_HOST' in self.META):
            host = self.META['HTTP_X_FORWARDED_HOST']
        elif 'HTTP_HOST' in self.META:
            host = self.META['HTTP_HOST']
        else:
            # Reconstruct the host using the algorithm from PEP 333.
            host = self.META['SERVER_NAME']
            server_port = self.get_port()
            if server_port != ('443' if self.is_secure() else '80'):
                host = '%s:%s' % (host, server_port)
        return host
    def get_host(self):
        """Return the HTTP host using the environment or request headers."""
        host = self._get_raw_host()

        # Allow variants of localhost if ALLOWED_HOSTS is empty and DEBUG=True.
        allowed_hosts = settings.ALLOWED_HOSTS
        if settings.DEBUG and not allowed_hosts:
            allowed_hosts = ['.localhost', '127.0.0.1', '[::1]']

        domain, port = split_domain_port(host)
        if domain and validate_host(domain, allowed_hosts):
            return host
        else:
            msg = "Invalid HTTP_HOST header: %r." % host
            if domain:
                msg += " You may need to add %r to ALLOWED_HOSTS." % domain
            else:
                msg += " The domain name provided is not valid according to RFC 1034/1035."
            raise DisallowedHost(msg)
    def get_port(self):
        """Return the port number for the request as a string."""
        if settings.USE_X_FORWARDED_PORT and 'HTTP_X_FORWARDED_PORT' in self.META:
            port = self.META['HTTP_X_FORWARDED_PORT']
        else:
            port = self.META['SERVER_PORT']
        return str(port)
    def _get_scheme(self):
        """
        Hook for subclasses like WSGIRequest to implement. Return 'http' by
        default.
        """
        return 'http'
    @property
    def scheme(self):
        if settings.SECURE_PROXY_SSL_HEADER:
            try:
                header, secure_value = settings.SECURE_PROXY_SSL_HEADER
            except ValueError:
                raise ImproperlyConfigured(
                    'The SECURE_PROXY_SSL_HEADER setting must be a tuple containing two values.'
                )
            header_value = self.META.get(header)
            if header_value is not None:
                return 'https' if header_value == secure_value else 'http'
        return self._get_scheme()
    def is_secure(self):
        return self.scheme == 'https'