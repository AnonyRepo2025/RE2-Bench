import functools
import gettext as gettext_module
import os
import re
import sys
import warnings
from asgiref.local import Local
from django.apps import apps
from django.conf import settings
from django.conf.locale import LANG_INFO
from django.core.exceptions import AppRegistryNotReady
from django.core.signals import setting_changed
from django.dispatch import receiver
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import SafeData, mark_safe
from . import to_language, to_locale

_translations = {}
_active = Local()
_default = None
CONTEXT_SEPARATOR = "\x04"
accept_language_re = _lazy_re_compile(
    r"""
        # "en", "en-au", "x-y-z", "es-419", "*"
        ([A-Za-z]{1,8}(?:-[A-Za-z0-9]{1,8})*|\*)
        # Optional "q=1.00", "q=0.8"
        (?:\s*;\s*q=(0(?:\.[0-9]{,3})?|1(?:\.0{,3})?))?
        # Multiple accepts per header.
        (?:\s*,\s*|$)
    """,
    re.VERBOSE,
)
language_code_re = _lazy_re_compile(
    r"^[a-z]{1,8}(?:-[a-z0-9]{1,8})*(?:@[a-z0-9]{1,20})?$", re.IGNORECASE
)
language_code_prefix_re = _lazy_re_compile(r"^/(\w+([@-]\w+){0,2})(/|$)")

def get_language_from_path(path, strict=False):
    """
    Return the language code if there's a valid language code found in `path`.

    If `strict` is False (the default), look for a country-specific variant
    when neither the language code nor its generic variant is found.
    """
    regex_match = language_code_prefix_re.match(path)
    if not regex_match:
        return None
    lang_code = regex_match[1]
    try:
        return get_supported_language_variant(lang_code, strict=strict)
    except LookupError:
        return None
