import datetime
import decimal
import unicodedata
from importlib import import_module
from django.conf import settings
from django.utils import dateformat, datetime_safe, numberformat
from django.utils.functional import lazy
from django.utils.translation import (
    check_for_language, get_language, to_locale,
)

_format_cache = {}
_format_modules_cache = {}
ISO_INPUT_FORMATS = {
    'DATE_INPUT_FORMATS': ['%Y-%m-%d'],
    'TIME_INPUT_FORMATS': ['%H:%M:%S', '%H:%M:%S.%f', '%H:%M'],
    'DATETIME_INPUT_FORMATS': [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M',
        '%Y-%m-%d'
    ],
}
FORMAT_SETTINGS = frozenset([
    'DECIMAL_SEPARATOR',
    'THOUSAND_SEPARATOR',
    'NUMBER_GROUPING',
    'FIRST_DAY_OF_WEEK',
    'MONTH_DAY_FORMAT',
    'TIME_FORMAT',
    'DATE_FORMAT',
    'DATETIME_FORMAT',
    'SHORT_DATE_FORMAT',
    'SHORT_DATETIME_FORMAT',
    'YEAR_MONTH_FORMAT',
    'DATE_INPUT_FORMATS',
    'TIME_INPUT_FORMATS',
    'DATETIME_INPUT_FORMATS',
])
get_format_lazy = lazy(get_format, str, list, tuple)

def localize(value, use_l10n=None):
    """
    Check if value is a localizable type (date, number...) and return it
    formatted as a string using current locale format.

    If use_l10n is provided and is not None, it forces the value to
    be localized (or not), overriding the value of settings.USE_L10N.
    """
    if isinstance(value, str):  # Handle strings first for performance reasons.
        return value
    elif isinstance(value, bool):  # Make sure booleans don't get treated as numbers
        return str(value)
    elif isinstance(value, (decimal.Decimal, float, int)):
        if use_l10n is False:
            return str(value)
        return number_format(value, use_l10n=use_l10n)
    elif isinstance(value, datetime.datetime):
        return date_format(value, 'DATETIME_FORMAT', use_l10n=use_l10n)
    elif isinstance(value, datetime.date):
        return date_format(value, use_l10n=use_l10n)
    elif isinstance(value, datetime.time):
        return time_format(value, 'TIME_FORMAT', use_l10n=use_l10n)
    return value
