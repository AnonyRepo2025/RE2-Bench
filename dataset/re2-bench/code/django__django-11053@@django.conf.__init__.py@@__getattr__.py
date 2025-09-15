import importlib
import os
import time
import traceback
import warnings
from pathlib import Path
import django
from django.conf import global_settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.deprecation import RemovedInDjango31Warning
from django.utils.functional import LazyObject, empty

ENVIRONMENT_VARIABLE = "DJANGO_SETTINGS_MODULE"
FILE_CHARSET_DEPRECATED_MSG = (
    'The FILE_CHARSET setting is deprecated. Starting with Django 3.1, all '
    'files read from disk must be UTF-8 encoded.'
)
settings = LazySettings()

class UserSettingsHolder:
    SETTINGS_MODULE = None
    def __init__(self, default_settings):
        """
        Requests for configuration variables not in this class are satisfied
        from the module specified in default_settings (if possible).
        """
        self.__dict__['_deleted'] = set()
        self.default_settings = default_settings
    def __getattr__(self, name):
        if not name.isupper() or name in self._deleted:
            raise AttributeError
        return getattr(self.default_settings, name)
    def __setattr__(self, name, value):
        self._deleted.discard(name)
        if name == 'FILE_CHARSET':
            warnings.warn(FILE_CHARSET_DEPRECATED_MSG, RemovedInDjango31Warning)
        super().__setattr__(name, value)
    def __dir__(self):
        return sorted(
            s for s in [*self.__dict__, *dir(self.default_settings)]
            if s not in self._deleted
        )