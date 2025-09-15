import functools
import inspect
import re
import string
from importlib import import_module
from pickle import PicklingError
from urllib.parse import quote
from asgiref.local import Local
from django.conf import settings
from django.core.checks import Error, Warning
from django.core.checks.urls import check_resolver
from django.core.exceptions import ImproperlyConfigured, ViewDoesNotExist
from django.utils.datastructures import MultiValueDict
from django.utils.functional import cached_property
from django.utils.http import RFC3986_SUBDELIMS, escape_leading_slashes
from django.utils.regex_helper import _lazy_re_compile, normalize
from django.utils.translation import get_language, get_supported_language_variant
from .converters import get_converter
from .exceptions import NoReverseMatch, Resolver404
from .utils import get_callable
from django.views import View
from django.conf import urls

_PATH_PARAMETER_COMPONENT_RE = _lazy_re_compile(
    r"<(?:(?P<converter>[^>:]+):)?(?P<parameter>[^>]+)>"
)

class LocalePrefixPattern:
    @property
    def language_prefix(self):
        language_code = get_language() or settings.LANGUAGE_CODE
        default_language = get_supported_language_variant(settings.LANGUAGE_CODE)
        if language_code == default_language and not self.prefix_default_language:
            return ""
        else:
            return "%s/" % language_code