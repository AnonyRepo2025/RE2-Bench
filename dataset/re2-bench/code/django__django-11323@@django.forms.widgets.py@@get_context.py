import copy
import datetime
import re
import warnings
from collections import defaultdict
from itertools import chain
from django.conf import settings
from django.forms.utils import to_current_timezone
from django.templatetags.static import static
from django.utils import datetime_safe, formats
from django.utils.datastructures import OrderedSet
from django.utils.dates import MONTHS
from django.utils.formats import get_format
from django.utils.html import format_html, html_safe
from django.utils.safestring import mark_safe
from django.utils.topological_sort import (
    CyclicDependencyError, stable_topological_sort,
)
from django.utils.translation import gettext_lazy as _
from .renderers import get_default_renderer

__all__ = (
    'Media', 'MediaDefiningClass', 'Widget', 'TextInput', 'NumberInput',
    'EmailInput', 'URLInput', 'PasswordInput', 'HiddenInput',
    'MultipleHiddenInput', 'FileInput', 'ClearableFileInput', 'Textarea',
    'DateInput', 'DateTimeInput', 'TimeInput', 'CheckboxInput', 'Select',
    'NullBooleanSelect', 'SelectMultiple', 'RadioSelect',
    'CheckboxSelectMultiple', 'MultiWidget', 'SplitDateTimeWidget',
    'SplitHiddenDateTimeWidget', 'SelectDateWidget',
)
MEDIA_TYPES = ('css', 'js')
FILE_INPUT_CONTRADICTION = object()

class SelectDateWidget(Widget):
    none_value = ('', '---')
    month_field = '%s_month'
    day_field = '%s_day'
    year_field = '%s_year'
    template_name = 'django/forms/widgets/select_date.html'
    input_type = 'select'
    select_widget = Select
    date_re = re.compile(r'(\d{4}|0)-(\d\d?)-(\d\d?)$')
    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        date_context = {}
        year_choices = [(i, str(i)) for i in self.years]
        if not self.is_required:
            year_choices.insert(0, self.year_none_value)
        year_name = self.year_field % name
        date_context['year'] = self.select_widget(attrs, choices=year_choices).get_context(
            name=year_name,
            value=context['widget']['value']['year'],
            attrs={
                **context['widget']['attrs'],
                'id': 'id_%s' % year_name,
                'placeholder': _('Year') if self.is_required else False,
            },
        )
        month_choices = list(self.months.items())
        if not self.is_required:
            month_choices.insert(0, self.month_none_value)
        month_name = self.month_field % name
        date_context['month'] = self.select_widget(attrs, choices=month_choices).get_context(
            name=month_name,
            value=context['widget']['value']['month'],
            attrs={
                **context['widget']['attrs'],
                'id': 'id_%s' % month_name,
                'placeholder': _('Month') if self.is_required else False,
            },
        )
        day_choices = [(i, i) for i in range(1, 32)]
        if not self.is_required:
            day_choices.insert(0, self.day_none_value)
        day_name = self.day_field % name
        date_context['day'] = self.select_widget(attrs, choices=day_choices,).get_context(
            name=day_name,
            value=context['widget']['value']['day'],
            attrs={
                **context['widget']['attrs'],
                'id': 'id_%s' % day_name,
                'placeholder': _('Day') if self.is_required else False,
            },
        )
        subwidgets = []
        for field in self._parse_date_fmt():
            subwidgets.append(date_context[field]['widget'])
        context['widget']['subwidgets'] = subwidgets
        return context
    def format_value(self, value):
        """
        Return a dict containing the year, month, and day of the current value.
        Use dict instead of a datetime to allow invalid dates such as February
        31 to display correctly.
        """
        year, month, day = None, None, None
        if isinstance(value, (datetime.date, datetime.datetime)):
            year, month, day = value.year, value.month, value.day
        elif isinstance(value, str):
            match = self.date_re.match(value)
            if match:
                # Convert any zeros in the date to empty strings to match the
                # empty option value.
                year, month, day = [int(val) or '' for val in match.groups()]
            elif settings.USE_L10N:
                input_format = get_format('DATE_INPUT_FORMATS')[0]
                try:
                    d = datetime.datetime.strptime(value, input_format)
                except ValueError:
                    pass
                else:
                    year, month, day = d.year, d.month, d.day
        return {'year': year, 'month': month, 'day': day}