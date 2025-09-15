import pickle
import time
from collections import OrderedDict
from threading import Lock
from django.core.cache.backends.base import DEFAULT_TIMEOUT, BaseCache

_caches = {}
_expire_info = {}
_locks = {}

class LocMemCache(BaseCache):
    pickle_protocol = pickle.HIGHEST_PROTOCOL
    def add(self, key, value, timeout=DEFAULT_TIMEOUT, version=None):
        key = self.make_key(key, version=version)
        self.validate_key(key)
        pickled = pickle.dumps(value, self.pickle_protocol)
        with self._lock:
            if self._has_expired(key):
                self._set(key, pickled, timeout)
                return True
            return False
    def _set(self, key, value, timeout=DEFAULT_TIMEOUT):
        if len(self._cache) >= self._max_entries:
            self._cull()
        self._cache[key] = value
        self._cache.move_to_end(key, last=False)
        self._expire_info[key] = self.get_backend_timeout(timeout)
    def _has_expired(self, key):
        exp = self._expire_info.get(key, -1)
        return exp is not None and exp <= time.time()