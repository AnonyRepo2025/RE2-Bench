import cgi
import codecs
import copy
import warnings
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
from django.utils.deprecation import RemovedInDjango40Warning
from django.utils.encoding import escape_uri_path, iri_to_uri
from django.utils.functional import cached_property
from django.utils.http import is_same_domain, limited_parse_qsl
from django.utils.regex_helper import _lazy_re_compile
from .multipartparser import parse_header

RAISE_ERROR = object()
host_validation_re = _lazy_re_compile(r"^([a-z0-9.-]+|\[[a-f0-9]*:[a-f0-9\.:]+\])(:\d+)?$")

class HttpRequest:
    _encoding = None
    _upload_handlers = []
    def is_ajax(self):
        warnings.warn(
            'request.is_ajax() is deprecated. See Django 3.1 release notes '
            'for more details about this deprecation.',
            RemovedInDjango40Warning,
        )
        return self.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'