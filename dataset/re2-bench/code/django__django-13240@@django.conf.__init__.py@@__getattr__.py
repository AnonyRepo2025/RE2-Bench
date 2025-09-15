import importlib
import os
import time
import traceback
import warnings
from pathlib import Path
import django
from django.conf import global_settings
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.core.validators import URLValidator
from django.utils.deprecation import RemovedInDjango40Warning
from django.utils.functional import LazyObject, empty
from django.urls import get_script_prefix

ENVIRONMENT_VARIABLE = "DJANGO_SETTINGS_MODULE"
PASSWORD_RESET_TIMEOUT_DAYS_DEPRECATED_MSG = (
    'The PASSWORD_RESET_TIMEOUT_DAYS setting is deprecated. Use '
    'PASSWORD_RESET_TIMEOUT instead.'
)
settings = LazySettings()

class LazySettings(LazyObject):
    def __getattr__(self, name):
        """Return the value of a setting and cache it in self.__dict__."""
        if self._wrapped is empty:
            self._setup(name)
        val = getattr(self._wrapped, name)

        # Special case some settings which require further modification.
        # This is done here for performance reasons so the modified value is cached.
        if name in {'MEDIA_URL', 'STATIC_URL'} and val is not None:
            val = self._add_script_prefix(val)
        elif name == 'SECRET_KEY' and not val:
            raise ImproperlyConfigured("The SECRET_KEY setting must not be empty.")

        self.__dict__[name] = val
        return val
    def __setattr__(self, name, value):
        """
        Set the value of setting. Clear all cached values if _wrapped changes
        (@override_settings does this) or clear single values when set.
        """
        if name == '_wrapped':
            self.__dict__.clear()
        else:
            self.__dict__.pop(name, None)
        super().__setattr__(name, value)
    def configure(self, default_settings=global_settings, **options):
        """
        Called to manually configure the settings. The 'default_settings'
        parameter sets where to retrieve any unspecified values from (its
        argument must support attribute access (__getattr__)).
        """
        if self._wrapped is not empty:
            raise RuntimeError('Settings already configured.')
        holder = UserSettingsHolder(default_settings)
        for name, value in options.items():
            if not name.isupper():
                raise TypeError('Setting %r must be uppercase.' % name)
            setattr(holder, name, value)
        self._wrapped = holder
    @property
    def configured(self):
        """Return True if the settings have already been configured."""
        return self._wrapped is not empty