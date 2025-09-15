import re
import warnings
from contextlib import ContextDecorator
from django.utils.autoreload import autoreload_started, file_changed
from django.utils.deprecation import RemovedInDjango40Warning
from django.utils.functional import lazy
from .template import templatize
from django.conf.locale import LANG_INFO
from django.conf import settings
from django.utils.translation import trans_real as trans
from django.utils.translation.reloader import watch_for_translation_changes, translation_file_changed
from django.utils.translation import trans_null as trans

__all__ = [
    'activate', 'deactivate', 'override', 'deactivate_all',
    'get_language', 'get_language_from_request',
    'get_language_info', 'get_language_bidi',
    'check_for_language', 'to_language', 'to_locale', 'templatize',
    'gettext', 'gettext_lazy', 'gettext_noop',
    'ugettext', 'ugettext_lazy', 'ugettext_noop',
    'ngettext', 'ngettext_lazy',
    'ungettext', 'ungettext_lazy',
    'pgettext', 'pgettext_lazy',
    'npgettext', 'npgettext_lazy',
    'LANGUAGE_SESSION_KEY',
]
LANGUAGE_SESSION_KEY = '_language'
_trans = Trans()
gettext_lazy = lazy(gettext, str)
pgettext_lazy = lazy(pgettext, str)
trim_whitespace_re = re.compile(r'\s*\n\s*')

def ungettext_lazy(singular, plural, number=None):
    """
    A legacy compatibility wrapper for Unicode handling on Python 2.
    An alias of ungettext_lazy() since Django 2.0.
    """
    warnings.warn(
        'django.utils.translation.ungettext_lazy() is deprecated in favor of '
        'django.utils.translation.ngettext_lazy().',
        RemovedInDjango40Warning, stacklevel=2,
    )
    return ngettext_lazy(singular, plural, number)
