import base64
import binascii
import functools
import hashlib
import importlib
import math
import warnings
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.core.signals import setting_changed
from django.dispatch import receiver
from django.utils.crypto import (
    RANDOM_STRING_CHARS, constant_time_compare, get_random_string, pbkdf2,
)
from django.utils.module_loading import import_string
from django.utils.translation import gettext_noop as _

UNUSABLE_PASSWORD_PREFIX = '!'  # This will never be a valid encoded hash
UNUSABLE_PASSWORD_SUFFIX_LENGTH = 40  # number of random chars to add after UNUSABLE_PASSWORD_PREFIX

class SHA1PasswordHasher(BasePasswordHasher):
    algorithm = "sha1"
    def must_update(self, encoded):
        decoded = self.decode(encoded)
        return must_update_salt(decoded['salt'], self.salt_entropy)