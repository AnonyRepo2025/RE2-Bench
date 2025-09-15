import inspect
from importlib import import_module
from inspect import cleandoc
from pathlib import Path
from django.apps import apps
from django.contrib import admin
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.admindocs import utils
from django.contrib.admindocs.utils import (
    remove_non_capturing_groups, replace_metacharacters, replace_named_groups,
    replace_unnamed_groups,
)
from django.core.exceptions import ImproperlyConfigured, ViewDoesNotExist
from django.db import models
from django.http import Http404
from django.template.engine import Engine
from django.urls import get_mod_func, get_resolver, get_urlconf
from django.utils._os import safe_join
from django.utils.decorators import method_decorator
from django.utils.functional import cached_property
from django.utils.inspect import (
    func_accepts_kwargs, func_accepts_var_args, get_func_full_args,
    method_has_no_args,
)
from django.utils.translation import gettext as _
from django.views.generic import TemplateView
from .utils import get_view_name

MODEL_METHODS_EXCLUDE = ('_', 'add_', 'delete', 'save', 'set_')

def simplify_regex(pattern):
    r"""
    Clean up urlpattern regexes into something more readable by humans. For
    example, turn "^(?P<sport_slug>\w+)/athletes/(?P<athlete_slug>\w+)/$"
    into "/<sport_slug>/athletes/<athlete_slug>/".
    """
    pattern = remove_non_capturing_groups(pattern)
    pattern = replace_named_groups(pattern)
    pattern = replace_unnamed_groups(pattern)
    pattern = replace_metacharacters(pattern)
    if not pattern.startswith('/'):
        pattern = '/' + pattern
    return pattern
