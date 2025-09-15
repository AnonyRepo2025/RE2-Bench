def get_context(self, name, value, attrs):
    context = {}
    context['widget'] = {'name': name, 'is_hidden': self.is_hidden, 'required': self.is_required, 'value': self.format_value(value), 'attrs': self.build_attrs(self.attrs, attrs), 'template_name': self.template_name}
    return context

def is_hidden(self):
    return self.input_type == 'hidden' if hasattr(self, 'input_type') else False

def __getattr__(self, name):
    if self._wrapped is empty:
        self._setup(name)
    val = getattr(self._wrapped, name)
    self.__dict__[name] = val
    return val

def build_attrs(self, base_attrs, extra_attrs=None):
    return {**base_attrs, **(extra_attrs or {})}

def get_format(format_type, lang=None, use_l10n=None):
    use_l10n = use_l10n or (use_l10n is None and settings.USE_L10N)
    if use_l10n and lang is None:
        lang = get_language()
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

def get_format_modules(lang=None, reverse=False):
    if lang is None:
        lang = get_language()
    if lang not in _format_modules_cache:
        _format_modules_cache[lang] = list(iter_format_modules(lang, settings.FORMAT_MODULE_PATH))
    modules = _format_modules_cache[lang]
    if reverse:
        return list(reversed(modules))
    return modules

def __getattr__(self, name):
    if not name.isupper() or name in self._deleted:
        raise AttributeError
    return getattr(self.default_settings, name)

def iter_format_modules(lang, format_module_path=None):
    if not check_for_language(lang):
        return
    if format_module_path is None:
        format_module_path = settings.FORMAT_MODULE_PATH
    format_locations = []
    if format_module_path:
        if isinstance(format_module_path, str):
            format_module_path = [format_module_path]
        for path in format_module_path:
            format_locations.append(path + '.%s')
    format_locations.append('django.conf.locale.%s')
    locale = to_locale(lang)
    locales = [locale]
    if '_' in locale:
        locales.append(locale.split('_')[0])
    for location in format_locations:
        for loc in locales:
            try:
                yield import_module('%s.formats' % (location % loc))
            except ImportError:
                pass

def check_for_language(lang_code):
    return _trans.check_for_language(lang_code)

def check_for_language(lang_code):
    if lang_code is None or not language_code_re.search(lang_code):
        return False
    return any((gettext_module.find('django', path, [to_locale(lang_code)]) is not None for path in all_locale_paths()))

def all_locale_paths():
    globalpath = os.path.join(os.path.dirname(sys.modules[settings.__module__].__file__), 'locale')
    app_paths = []
    for app_config in apps.get_app_configs():
        locale_path = os.path.join(app_config.path, 'locale')
        if os.path.exists(locale_path):
            app_paths.append(locale_path)
    return [globalpath, *settings.LOCALE_PATHS, *app_paths]

def get_app_configs(self):
    self.check_apps_ready()
    return self.app_configs.values()



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
from django.utils.topological_sort import CyclicDependencyError, stable_topological_sort
from django.utils.translation import gettext_lazy as _
from .renderers import get_default_renderer
__all__ = ('Media', 'MediaDefiningClass', 'Widget', 'TextInput', 'NumberInput', 'EmailInput', 'URLInput', 'PasswordInput', 'HiddenInput', 'MultipleHiddenInput', 'FileInput', 'ClearableFileInput', 'Textarea', 'DateInput', 'DateTimeInput', 'TimeInput', 'CheckboxInput', 'Select', 'NullBooleanSelect', 'SelectMultiple', 'RadioSelect', 'CheckboxSelectMultiple', 'MultiWidget', 'SplitDateTimeWidget', 'SplitHiddenDateTimeWidget', 'SelectDateWidget')
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
    date_re = re.compile('(\\d{4}|0)-(\\d\\d?)-(\\d\\d?)$')

    def get_context(self, name, value, attrs):
        context = super().get_context(name, value, attrs)
        date_context = {}
        year_choices = [(i, str(i)) for i in self.years]
        if not self.is_required:
            year_choices.insert(0, self.year_none_value)
        year_name = self.year_field % name
        date_context['year'] = self.select_widget(attrs, choices=year_choices).get_context(name=year_name, value=context['widget']['value']['year'], attrs={**context['widget']['attrs'], 'id': 'id_%s' % year_name, 'placeholder': _('Year') if self.is_required else False})
        month_choices = list(self.months.items())
        if not self.is_required:
            month_choices.insert(0, self.month_none_value)
        month_name = self.month_field % name
        date_context['month'] = self.select_widget(attrs, choices=month_choices).get_context(name=month_name, value=context['widget']['value']['month'], attrs={**context['widget']['attrs'], 'id': 'id_%s' % month_name, 'placeholder': _('Month') if self.is_required else False})
        day_choices = [(i, i) for i in range(1, 32)]
        if not self.is_required:
            day_choices.insert(0, self.day_none_value)
        day_name = self.day_field % name
        date_context['day'] = self.select_widget(attrs, choices=day_choices).get_context(name=day_name, value=context['widget']['value']['day'], attrs={**context['widget']['attrs'], 'id': 'id_%s' % day_name, 'placeholder': _('Day') if self.is_required else False})
        subwidgets = []
        for field in self._parse_date_fmt():
            subwidgets.append(date_context[field]['widget'])
        context['widget']['subwidgets'] = subwidgets
        return context

    def format_value(self, value):
        year, month, day = (None, None, None)
        if isinstance(value, (datetime.date, datetime.datetime)):
            year, month, day = (value.year, value.month, value.day)
        elif isinstance(value, str):
            match = self.date_re.match(value)
            if match:
                year, month, day = [int(val) or '' for val in match.groups()]
            elif settings.USE_L10N:
                input_format = get_format('DATE_INPUT_FORMATS')[0]
                try:
                    d = datetime.datetime.strptime(value, input_format)
                except ValueError:
                    pass
                else:
                    year, month, day = (d.year, d.month, d.day)
        return {'year': year, 'month': month, 'day': day}