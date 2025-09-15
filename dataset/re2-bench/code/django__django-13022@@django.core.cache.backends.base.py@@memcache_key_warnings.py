import time
import warnings
from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string

DEFAULT_TIMEOUT = object()
MEMCACHE_MAX_KEY_LENGTH = 250

def memcache_key_warnings(key):
    if len(key) > MEMCACHE_MAX_KEY_LENGTH:
        yield (
            'Cache key will cause errors if used with memcached: %r '
            '(longer than %s)' % (key, MEMCACHE_MAX_KEY_LENGTH)
        )
    for char in key:
        if ord(char) < 33 or ord(char) == 127:
            yield (
                'Cache key contains characters that will cause errors if '
                'used with memcached: %r' % key
            )
            break
