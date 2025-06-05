import cgi
import codecs
import copy
from io import BytesIO
from itertools import chain
from urllib.parse import parse_qsl, quote, urlencode, urljoin, urlsplit
from django.conf import settings
from django.core import signing
from django.core.exceptions import (
    DisallowedHost,
    ImproperlyConfigured,
    RequestDataTooBig,
    TooManyFieldsSent,
)
from django.core.files import uploadhandler
from django.http.multipartparser import MultiPartParser, MultiPartParserError
from django.utils.datastructures import (
    CaseInsensitiveMapping,
    ImmutableList,
    MultiValueDict,
)
from django.utils.encoding import escape_uri_path, iri_to_uri
from django.utils.functional import cached_property
from django.utils.http import is_same_domain
from django.utils.regex_helper import _lazy_re_compile
from .multipartparser import parse_header

RAISE_ERROR = object()
host_validation_re = _lazy_re_compile(
    r"^([a-z0-9.-]+|\[[a-f0-9]*:[a-f0-9\.:]+\])(:[0-9]+)?$"
)

class HttpRequest:
    _encoding = None
    _upload_handlers = []
    @property
    def scheme(self):
        if settings.SECURE_PROXY_SSL_HEADER:
            try:
                header, secure_value = settings.SECURE_PROXY_SSL_HEADER
            except ValueError:
                raise ImproperlyConfigured(
                    "The SECURE_PROXY_SSL_HEADER setting must be a tuple containing "
                    "two values."
                )
            header_value = self.META.get(header)
            if header_value is not None:
                header_value, *_ = header_value.split(",", 1)
                return "https" if header_value.strip() == secure_value else "http"
        return self._get_scheme()