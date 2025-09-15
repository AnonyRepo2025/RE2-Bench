import copy
import datetime
import json
import math
import operator
import os
import re
import uuid
import warnings
from decimal import Decimal, DecimalException
from io import BytesIO
from urllib.parse import urlsplit, urlunsplit
from django.core import validators
from django.core.exceptions import ValidationError
from django.db.models.enums import ChoicesMeta
from django.forms.boundfield import BoundField
from django.forms.utils import from_current_timezone, to_current_timezone
from django.forms.widgets import (
    FILE_INPUT_CONTRADICTION,
    CheckboxInput,
    ClearableFileInput,
    DateInput,
    DateTimeInput,
    EmailInput,
    FileInput,
    HiddenInput,
    MultipleHiddenInput,
    NullBooleanSelect,
    NumberInput,
    Select,
    SelectMultiple,
    SplitDateTimeWidget,
    SplitHiddenDateTimeWidget,
    Textarea,
    TextInput,
    TimeInput,
    URLInput,
)
from django.utils import formats
from django.utils.dateparse import parse_datetime, parse_duration
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.duration import duration_string
from django.utils.ipv6 import clean_ipv6_address
from django.utils.regex_helper import _lazy_re_compile
from django.utils.translation import gettext_lazy as _
from django.utils.translation import ngettext_lazy
from PIL import Image

__all__ = (
    "Field",
    "CharField",
    "IntegerField",
    "DateField",
    "TimeField",
    "DateTimeField",
    "DurationField",
    "RegexField",
    "EmailField",
    "FileField",
    "ImageField",
    "URLField",
    "BooleanField",
    "NullBooleanField",
    "ChoiceField",
    "MultipleChoiceField",
    "ComboField",
    "MultiValueField",
    "FloatField",
    "DecimalField",
    "SplitDateTimeField",
    "GenericIPAddressField",
    "FilePathField",
    "JSONField",
    "SlugField",
    "TypedChoiceField",
    "TypedMultipleChoiceField",
    "UUIDField",
)

class URLField(CharField):
    widget = URLInput
    default_error_messages = {
        "invalid": _("Enter a valid URL."),
    }
    default_validators = [validators.URLValidator()]
    def to_python(self, value):
        def split_url(url):
            """
            Return a list of url parts via urlparse.urlsplit(), or raise
            ValidationError for some malformed URLs.
            """
            try:
                return list(urlsplit(url))
            except ValueError:
                # urlparse.urlsplit can raise a ValueError with some
                # misformatted URLs.
                raise ValidationError(self.error_messages["invalid"], code="invalid")

        value = super().to_python(value)
        if value:
            url_fields = split_url(value)
            if not url_fields[0]:
                # If no URL scheme given, add a scheme.
                url_fields[0] = self.assume_scheme
            if not url_fields[1]:
                # Assume that if no domain is provided, that the path segment
                # contains the domain.
                url_fields[1] = url_fields[2]
                url_fields[2] = ""
                # Rebuild the url_fields list, since the domain segment may now
                # contain the path too.
                url_fields = split_url(urlunsplit(url_fields))
            value = urlunsplit(url_fields)
        return value