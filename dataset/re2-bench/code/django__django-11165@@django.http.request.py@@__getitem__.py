import copy
import re
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

RAISE_ERROR = object()
host_validation_re = re.compile(r"^([a-z0-9.-]+|\[[a-f0-9]*:[a-f0-9\.:]+\])(:\d+)?$")

class HttpHeaders(CaseInsensitiveMapping):
    HTTP_PREFIX = 'HTTP_'
    UNPREFIXED_HEADERS = {'CONTENT_TYPE', 'CONTENT_LENGTH'}
    def __getitem__(self, key):
        """Allow header lookup using underscores in place of hyphens."""
        return super().__getitem__(key.replace('_', '-'))