import builtins
import collections.abc
import datetime
import decimal
import enum
import functools
import math
import re
import types
import uuid
from django.conf import SettingsReference
from django.db import models
from django.db.migrations.operations.base import Operation
from django.db.migrations.utils import COMPILED_REGEX_TYPE, RegexObject
from django.utils.functional import LazyObject, Promise
from django.utils.timezone import utc
from django.utils.version import get_docs_version
from django.db.migrations.writer import OperationWriter



class EnumSerializer(BaseSerializer):
    def serialize(self):
        enum_class = self.value.__class__
        module = enum_class.__module__
        return (
            '%s.%s[%r]' % (module, enum_class.__name__, self.value.name),
            {'import %s' % module},
        )