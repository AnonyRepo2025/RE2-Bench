import hashlib
import json
import os
import posixpath
import re
import warnings
from urllib.parse import unquote, urldefrag, urlsplit, urlunsplit
from django.conf import settings
from django.contrib.staticfiles.utils import check_settings, matches_patterns
from django.core.cache import (
    InvalidCacheBackendError, cache as default_cache, caches,
)
from django.core.exceptions import ImproperlyConfigured
from django.core.files.base import ContentFile
from django.core.files.storage import FileSystemStorage, get_storage_class
from django.utils.deprecation import RemovedInDjango31Warning
from django.utils.functional import LazyObject

staticfiles_storage = ConfiguredStorage()

class ManifestFilesMixin(HashedFilesMixin):
    manifest_version = '1.0'  # the manifest format standard
    manifest_name = 'staticfiles.json'
    manifest_strict = True
    keep_intermediate_files = False
    def read_manifest(self):
        try:
            with self.open(self.manifest_name) as manifest:
                return manifest.read().decode()
        except FileNotFoundError:
            return None