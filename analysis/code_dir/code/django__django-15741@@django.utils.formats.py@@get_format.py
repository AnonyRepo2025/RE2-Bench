import datetime
import decimal
import functools
import re
import unicodedata
from importlib import import_module
from django.conf import settings
from django.utils import dateformat, numberformat
from django.utils.functional import lazy
from django.utils.translation import check_for_language, get_language, to_locale
_format_cache = {}
_format_modules_cache = {}
ISO_INPUT_FORMATS = {'DATE_INPUT_FORMATS': ['%Y-%m-%d'], 'TIME_INPUT_FORMATS': ['%H:%M:%S', '%H:%M:%S.%f', '%H:%M'], 'DATETIME_INPUT_FORMATS': ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M', '%Y-%m-%d']}
FORMAT_SETTINGS = frozenset(['DECIMAL_SEPARATOR', 'THOUSAND_SEPARATOR', 'NUMBER_GROUPING', 'FIRST_DAY_OF_WEEK', 'MONTH_DAY_FORMAT', 'TIME_FORMAT', 'DATE_FORMAT', 'DATETIME_FORMAT', 'SHORT_DATE_FORMAT', 'SHORT_DATETIME_FORMAT', 'YEAR_MONTH_FORMAT', 'DATE_INPUT_FORMATS', 'TIME_INPUT_FORMATS', 'DATETIME_INPUT_FORMATS'])
get_format_lazy = lazy(get_format, str, list, tuple)

def get_format(format_type, lang=None, use_l10n=None):
    if use_l10n is None:
        try:
            use_l10n = settings._USE_L10N_INTERNAL
        except AttributeError:
            use_l10n = settings.USE_L10N
    if use_l10n and lang is None:
        lang = get_language()
    format_type = str(format_type)
    cache_key = (format_type, lang)
    try:
        return _format_cache[cache_key]
    except KeyError:
        pass
    val = None
    if use_l10n:
        for module in get_format_modules(lang):
            val = getattr(module, format_type, None)
            if val is not None:
                break
    if val is None:
        if format_type not in FORMAT_SETTINGS:
            return format_type
        val = getattr(settings, format_type)
    elif format_type in ISO_INPUT_FORMATS:
        val = list(val)
        for iso_input in ISO_INPUT_FORMATS.get(format_type, ()):
            if iso_input not in val:
                val.append(iso_input)
    _format_cache[cache_key] = val
    return val