import importlib
import os
import time
import traceback
import warnings
from pathlib import Path
import django
from django.conf import global_settings
from django.core.exceptions import ImproperlyConfigured
from django.utils.deprecation import RemovedInDjango40Warning
from django.utils.functional import LazyObject, empty
from django.urls import get_script_prefix

ENVIRONMENT_VARIABLE = "DJANGO_SETTINGS_MODULE"
PASSWORD_RESET_TIMEOUT_DAYS_DEPRECATED_MSG = (
    'The PASSWORD_RESET_TIMEOUT_DAYS setting is deprecated. Use '
    'PASSWORD_RESET_TIMEOUT instead.'
)
DEFAULT_HASHING_ALGORITHM_DEPRECATED_MSG = (
    'The DEFAULT_HASHING_ALGORITHM transitional setting is deprecated. '
    'Support for it and tokens, cookies, sessions, and signatures that use '
    'SHA-1 hashing algorithm will be removed in Django 4.0.'
)
settings = LazySettings()

class LazySettings(LazyObject):
    @staticmethod
    def _add_script_prefix(value):
        """
        Add SCRIPT_NAME prefix to relative paths.

        Useful when the app is being served at a subpath and manually prefixing
        subpath to STATIC_URL and MEDIA_URL in settings is inconvenient.
        """
        # Don't apply prefix to absolute paths and URLs.
        if value.startswith(('http://', 'https://', '/')):
            return value
        from django.urls import get_script_prefix
        return '%s%s' % (get_script_prefix(), value)