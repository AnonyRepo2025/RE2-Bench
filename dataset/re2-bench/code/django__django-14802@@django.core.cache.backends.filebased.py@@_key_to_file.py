import glob
import hashlib
import os
import pickle
import random
import tempfile
import time
import zlib
from django.core.cache.backends.base import DEFAULT_TIMEOUT, BaseCache
from django.core.files import locks
from django.core.files.move import file_move_safe



class FileBasedCache(BaseCache):
    cache_suffix = '.djcache'
    pickle_protocol = pickle.HIGHEST_PROTOCOL
    def _key_to_file(self, key, version=None):
        """
        Convert a key into a cache file path. Basically this is the
        root cache path joined with the md5sum of the key and a suffix.
        """
        key = self.make_and_validate_key(key, version=version)
        return os.path.join(self._dir, ''.join(
            [hashlib.md5(key.encode()).hexdigest(), self.cache_suffix]))