import time
import warnings
from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string
DEFAULT_TIMEOUT = object()
MEMCACHE_MAX_KEY_LENGTH = 250

class BaseCache:
    _missing_key = object()

    def __init__(self, params):
        timeout = params.get('timeout', params.get('TIMEOUT', 300))
        if timeout is not None:
            try:
                timeout = int(timeout)
            except (ValueError, TypeError):
                timeout = 300
        self.default_timeout = timeout
        options = params.get('OPTIONS', {})
        max_entries = params.get('max_entries', options.get('MAX_ENTRIES', 300))
        try:
            self._max_entries = int(max_entries)
        except (ValueError, TypeError):
            self._max_entries = 300
        cull_frequency = params.get('cull_frequency', options.get('CULL_FREQUENCY', 3))
        try:
            self._cull_frequency = int(cull_frequency)
        except (ValueError, TypeError):
            self._cull_frequency = 3
        self.key_prefix = params.get('KEY_PREFIX', '')
        self.version = params.get('VERSION', 1)
        self.key_func = get_key_func(params.get('KEY_FUNCTION'))

    def get_backend_timeout(self, timeout=DEFAULT_TIMEOUT):
        if timeout == DEFAULT_TIMEOUT:
            timeout = self.default_timeout
        elif timeout == 0:
            timeout = -1
        return None if timeout is None else time.time() + timeout

    def make_key(self, key, version=None):
        if version is None:
            version = self.version
        return self.key_func(key, self.key_prefix, version)

    def add(self, key, value, timeout=DEFAULT_TIMEOUT, version=None):
        raise NotImplementedError('subclasses of BaseCache must provide an add() method')

    def get(self, key, default=None, version=None):
        raise NotImplementedError('subclasses of BaseCache must provide a get() method')

    def set(self, key, value, timeout=DEFAULT_TIMEOUT, version=None):
        raise NotImplementedError('subclasses of BaseCache must provide a set() method')

    def touch(self, key, timeout=DEFAULT_TIMEOUT, version=None):
        raise NotImplementedError('subclasses of BaseCache must provide a touch() method')

    def delete(self, key, version=None):
        raise NotImplementedError('subclasses of BaseCache must provide a delete() method')

    def get_many(self, keys, version=None):
        d = {}
        for k in keys:
            val = self.get(k, self._missing_key, version=version)
            if val is not self._missing_key:
                d[k] = val
        return d

    def get_or_set(self, key, default, timeout=DEFAULT_TIMEOUT, version=None):
        val = self.get(key, self._missing_key, version=version)
        if val is self._missing_key:
            if callable(default):
                default = default()
            self.add(key, default, timeout=timeout, version=version)
            return self.get(key, default, version=version)
        return val

    def has_key(self, key, version=None):
        return self.get(key, self._missing_key, version=version) is not self._missing_key

    def incr(self, key, delta=1, version=None):
        value = self.get(key, self._missing_key, version=version)
        if value is self._missing_key:
            raise ValueError("Key '%s' not found" % key)
        new_value = value + delta
        self.set(key, new_value, version=version)
        return new_value

    def decr(self, key, delta=1, version=None):
        return self.incr(key, -delta, version=version)

    def __contains__(self, key):
        return self.has_key(key)

    def set_many(self, data, timeout=DEFAULT_TIMEOUT, version=None):
        for key, value in data.items():
            self.set(key, value, timeout=timeout, version=version)
        return []

    def delete_many(self, keys, version=None):
        for key in keys:
            self.delete(key, version=version)

    def clear(self):
        raise NotImplementedError('subclasses of BaseCache must provide a clear() method')

    def validate_key(self, key):
        for warning in memcache_key_warnings(key):
            warnings.warn(warning, CacheKeyWarning)

    def incr_version(self, key, delta=1, version=None):
        if version is None:
            version = self.version
        value = self.get(key, self._missing_key, version=version)
        if value is self._missing_key:
            raise ValueError("Key '%s' not found" % key)
        self.set(key, value, version=version + delta)
        self.delete(key, version=version)
        return version + delta

    def decr_version(self, key, delta=1, version=None):
        return self.incr_version(key, -delta, version)

    def close(self, **kwargs):
        pass