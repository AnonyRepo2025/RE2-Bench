import functools
import inspect
import re
from importlib import import_module
from urllib.parse import quote
from asgiref.local import Local
from django.conf import settings
from django.core.checks import Error, Warning
from django.core.checks.urls import check_resolver
from django.core.exceptions import ImproperlyConfigured, ViewDoesNotExist
from django.utils.datastructures import MultiValueDict
from django.utils.functional import cached_property
from django.utils.http import RFC3986_SUBDELIMS, escape_leading_slashes
from django.utils.regex_helper import normalize
from django.utils.translation import get_language
from .converters import get_converter
from .exceptions import NoReverseMatch, Resolver404
from .utils import get_callable
from django.conf import urls

_PATH_PARAMETER_COMPONENT_RE = re.compile(
    r'<(?:(?P<converter>[^>:]+):)?(?P<parameter>\w+)>'
)

class RegexPattern(CheckURLMixin):
    regex = LocaleRegexDescriptor('_regex')
    def match(self, path):
        match = self.regex.search(path)
        if match:
            # If there are any named groups, use those as kwargs, ignoring
            # non-named groups. Otherwise, pass all non-named arguments as
            # positional arguments.
            kwargs = {k: v for k, v in match.groupdict().items() if v is not None}
            args = () if kwargs else match.groups()
            return path[match.end():], args, kwargs
        return None