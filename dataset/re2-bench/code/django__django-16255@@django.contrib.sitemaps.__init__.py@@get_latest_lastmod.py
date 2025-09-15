import warnings
from urllib.parse import urlencode
from urllib.request import urlopen
from django.apps import apps as django_apps
from django.conf import settings
from django.core import paginator
from django.core.exceptions import ImproperlyConfigured
from django.urls import NoReverseMatch, reverse
from django.utils import translation
from django.utils.deprecation import RemovedInDjango50Warning

PING_URL = "https://www.google.com/webmasters/tools/ping"

class Sitemap:
    limit = 50000
    protocol = None
    i18n = False
    languages = None
    alternates = False
    x_default = False
    def get_latest_lastmod(self):
        if not hasattr(self, "lastmod"):
            return None
        if callable(self.lastmod):
            try:
                return max([self.lastmod(item) for item in self.items()], default=None)
            except TypeError:
                return None
        else:
            return self.lastmod