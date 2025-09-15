from contextlib import ContextDecorator
from decimal import ROUND_UP, Decimal
from django.utils.autoreload import autoreload_started, file_changed
from django.utils.functional import lazy
from django.utils.regex_helper import _lazy_re_compile
from .template import templatize
from django.conf.locale import LANG_INFO
from django.conf import settings
from django.utils.translation import trans_real as trans
from django.utils.translation.reloader import (
                translation_file_changed, watch_for_translation_changes,
            )
from django.utils.translation import trans_null as trans

__all__ = [
    'activate', 'deactivate', 'override', 'deactivate_all',
    'get_language', 'get_language_from_request',
    'get_language_info', 'get_language_bidi',
    'check_for_language', 'to_language', 'to_locale', 'templatize',
    'gettext', 'gettext_lazy', 'gettext_noop',
    'ngettext', 'ngettext_lazy',
    'pgettext', 'pgettext_lazy',
    'npgettext', 'npgettext_lazy',
]
_trans = Trans()
gettext_lazy = lazy(gettext, str)
pgettext_lazy = lazy(pgettext, str)
trim_whitespace_re = _lazy_re_compile(r'\s*\n\s*')

def to_locale(language):
    """Turn a language name (en-us) into a locale name (en_US)."""
    lang, _, country = language.lower().partition('-')
    if not country:
        return language[:3].lower() + language[3:]
    # A language with > 2 characters after the dash only has its first
    # character after the dash capitalized; e.g. sr-latn becomes sr_Latn.
    # A language with 2 characters after the dash has both characters
    # capitalized; e.g. en-us becomes en_US.
    country, _, tail = country.partition('-')
    country = country.title() if len(country) > 2 else country.upper()
    if tail:
        country += '-' + tail
    return lang + '_' + country
