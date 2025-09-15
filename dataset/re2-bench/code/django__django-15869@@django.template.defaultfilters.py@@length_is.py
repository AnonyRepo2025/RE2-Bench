import random as random_module
import re
import types
import warnings
from decimal import ROUND_HALF_UP, Context, Decimal, InvalidOperation
from functools import wraps
from inspect import unwrap
from operator import itemgetter
from pprint import pformat
from urllib.parse import quote
from django.utils import formats
from django.utils.dateformat import format, time_format
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.encoding import iri_to_uri
from django.utils.html import avoid_wrapping, conditional_escape, escape, escapejs
from django.utils.html import json_script as _json_script
from django.utils.html import linebreaks, strip_tags
from django.utils.html import urlize as _urlize
from django.utils.safestring import SafeData, mark_safe
from django.utils.text import Truncator, normalize_newlines, phone2numeric
from django.utils.text import slugify as _slugify
from django.utils.text import wrap
from django.utils.timesince import timesince, timeuntil
from django.utils.translation import gettext, ngettext
from .base import VARIABLE_ATTRIBUTE_SEPARATOR
from .library import Library

register = Library()

def length_is(value, arg):
    """Return a boolean of whether the value's length is the argument."""
    warnings.warn(
        "The length_is template filter is deprecated in favor of the length template "
        "filter and the == operator within an {% if %} tag.",
        RemovedInDjango51Warning,
    )
    try:
        return len(value) == int(arg)
    except (ValueError, TypeError):
        return ""
