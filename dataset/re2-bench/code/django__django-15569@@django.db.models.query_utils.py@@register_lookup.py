import copy
import functools
import inspect
from collections import namedtuple
from django.core.exceptions import FieldError
from django.db.models.constants import LOOKUP_SEP
from django.utils import tree
from django.db.models.lookups import Lookup
from django.db.models.lookups import Transform

PathInfo = namedtuple(
    "PathInfo",
    "from_opts to_opts target_fields join_field m2m direct filtered_relation",
)

class RegisterLookupMixin:
    @classmethod
    def _clear_cached_lookups(cls):
        for subclass in subclasses(cls):
            subclass.get_lookups.cache_clear()
    @classmethod
    def register_lookup(cls, lookup, lookup_name=None):
        if lookup_name is None:
            lookup_name = lookup.lookup_name
        if "class_lookups" not in cls.__dict__:
            cls.class_lookups = {}
        cls.class_lookups[lookup_name] = lookup
        cls._clear_cached_lookups()
        return lookup