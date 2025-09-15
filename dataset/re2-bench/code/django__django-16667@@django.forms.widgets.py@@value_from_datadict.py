import copy
import datetime
import warnings
from collections import defaultdict
from graphlib import CycleError, TopologicalSorter
from itertools import chain
from django.forms.utils import to_current_timezone
from django.templatetags.static import static
from django.utils import formats
from django.utils.dates import MONTHS
from django.utils.formats import get_format
from django.utils.html import format_html, html_safe
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import mark_safe
from django.utils.translation import gettext_lazy as _
from .renderers import get_default_renderer

__all__ = (
    "Media",
    "MediaDefiningClass",
    "Widget",
    "TextInput",
    "NumberInput",
    "EmailInput",
    "URLInput",
    "PasswordInput",
    "HiddenInput",
    "MultipleHiddenInput",
    "FileInput",
    "ClearableFileInput",
    "Textarea",
    "DateInput",
    "DateTimeInput",
    "TimeInput",
    "CheckboxInput",
    "Select",
    "NullBooleanSelect",
    "SelectMultiple",
    "RadioSelect",
    "CheckboxSelectMultiple",
    "MultiWidget",
    "SplitDateTimeWidget",
    "SplitHiddenDateTimeWidget",
    "SelectDateWidget",
)
MEDIA_TYPES = ("css", "js")
FILE_INPUT_CONTRADICTION = object()

class SelectDateWidget(Widget):
    none_value = ("", "---")
    month_field = "%s_month"
    day_field = "%s_day"
    year_field = "%s_year"
    template_name = "django/forms/widgets/select_date.html"
    input_type = "select"
    select_widget = Select
    date_re = _lazy_re_compile(r"(\d{4}|0)-(\d\d?)-(\d\d?)$")
    use_fieldset = True
    def value_from_datadict(self, data, files, name):
        y = data.get(self.year_field % name)
        m = data.get(self.month_field % name)
        d = data.get(self.day_field % name)
        if y == m == d == "":
            return None
        if y is not None and m is not None and d is not None:
            input_format = get_format("DATE_INPUT_FORMATS")[0]
            input_format = formats.sanitize_strftime_format(input_format)
            try:
                date_value = datetime.date(int(y), int(m), int(d))
            except ValueError:
                # Return pseudo-ISO dates with zeros for any unselected values,
                # e.g. '2017-0-23'.
                return "%s-%s-%s" % (y or 0, m or 0, d or 0)
            except OverflowError:
                return "0-0-0"
            return date_value.strftime(input_format)
        return data.get(name)