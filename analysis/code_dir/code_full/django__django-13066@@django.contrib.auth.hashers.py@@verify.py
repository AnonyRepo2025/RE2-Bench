def _load_library(self):
    if self.library is not None:
        if isinstance(self.library, (tuple, list)):
            name, mod_path = self.library
        else:
            mod_path = self.library
        try:
            module = importlib.import_module(mod_path)
        except ImportError as e:
            raise ValueError("Couldn't load %r algorithm library: %s" % (self.__class__.__name__, e))
        return module
    raise ValueError("Hasher %r doesn't specify a library attribute" % self.__class__.__name__)



import base64
import binascii
import functools
import hashlib
import importlib
import warnings
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.core.signals import setting_changed
from django.dispatch import receiver
from django.utils.crypto import constant_time_compare, get_random_string, pbkdf2
from django.utils.module_loading import import_string
from django.utils.translation import gettext_noop as _
UNUSABLE_PASSWORD_PREFIX = '!'
UNUSABLE_PASSWORD_SUFFIX_LENGTH = 40

class Argon2PasswordHasher(BasePasswordHasher):
    algorithm = 'argon2'
    library = 'argon2'
    time_cost = 2
    memory_cost = 102400
    parallelism = 8

    def verify(self, password, encoded):
        argon2 = self._load_library()
        algorithm, rest = encoded.split('$', 1)
        assert algorithm == self.algorithm
        try:
            return argon2.PasswordHasher().verify('$' + rest, password)
        except argon2.exceptions.VerificationError:
            return False