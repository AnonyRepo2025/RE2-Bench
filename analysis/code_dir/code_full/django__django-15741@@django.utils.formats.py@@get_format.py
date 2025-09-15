def __getattribute__(self, name):
    if name == '_wrapped':
        return super().__getattribute__(name)
    value = super().__getattribute__(name)
    if not getattr(value, '_mask_wrapped', True):
        raise AttributeError
    return value

def _USE_L10N_INTERNAL(self):
    return self.__getattr__('USE_L10N')

def __getattr__(self, name):
    if (_wrapped := self._wrapped) is empty:
        self._setup(name)
        _wrapped = self._wrapped
    val = getattr(_wrapped, name)
    if name in {'MEDIA_URL', 'STATIC_URL'} and val is not None:
        val = self._add_script_prefix(val)
    elif name == 'SECRET_KEY' and (not val):
        raise ImproperlyConfigured('The SECRET_KEY setting must not be empty.')
    self.__dict__[name] = val
    return val

def get_language():
    return _trans.get_language()

def get_language():
    t = getattr(_active, 'value', None)
    if t is not None:
        try:
            return t.to_language()
        except AttributeError:
            pass
    return settings.LANGUAGE_CODE

def to_language(self):
    return self.__to_language

def __getattr__(self, name):
    if not name.isupper() or name in self._deleted:
        raise AttributeError
    return getattr(self.default_settings, name)

def gettext(message):
    return _trans.gettext(message)

def gettext(message):
    global _default
    eol_message = message.replace('\r\n', '\n').replace('\r', '\n')
    if eol_message:
        _default = _default or translation(settings.LANGUAGE_CODE)
        translation_object = getattr(_active, 'value', _default)
        result = translation_object.gettext(eol_message)
    else:
        result = type(message)('')
    if isinstance(message, SafeData):
        return mark_safe(result)
    return result

def get(self, key, default=None):
    missing = object()
    for cat in self._catalogs:
        result = cat.get(key, missing)
        if result is not missing:
            return result
    return default

def __str__(self):
    return self

def mark_safe(s):
    if hasattr(s, '__html__'):
        return s
    if callable(s):
        return _safety_decorator(mark_safe, s)
    return SafeString(s)

def wrapper(*args, **kwargs):
    if any((isinstance(arg, Promise) for arg in itertools.chain(args, kwargs.values()))):
        return lazy_func(*args, **kwargs)
    return func(*args, **kwargs)



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