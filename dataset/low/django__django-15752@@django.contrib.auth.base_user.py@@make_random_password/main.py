import unicodedata
import warnings
from django.contrib.auth import password_validation
from django.contrib.auth.hashers import (
    check_password,
    is_password_usable,
    make_password,
)
from django.db import models
from django.utils.crypto import get_random_string, salted_hmac
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.translation import gettext_lazy as _



class BaseUserManager(Manager):
    def make_random_password(
        self,
        length=10,
        allowed_chars="abcdefghjkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789",
    ):
        warnings.warn(
            "BaseUserManager.make_random_password() is deprecated.",
            category=RemovedInDjango51Warning,
            stacklevel=2,
        )
        return get_random_string(length, allowed_chars)