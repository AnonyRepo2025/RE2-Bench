import copy
import functools
import inspect
from collections import namedtuple
from django.core.exceptions import FieldError
from django.db.models.constants import LOOKUP_SEP
from django.utils import tree
from django.db.models.lookups import Lookup
from django.db.models.lookups import Transform

PathInfo = namedtuple('PathInfo', 'from_opts to_opts target_fields join_field m2m direct filtered_relation')

class Q(Node):
    AND = 'AND'
    OR = 'OR'
    default = AND
    conditional = True
    def __init__(self, *args, _connector=None, _negated=False, **kwargs):
        super().__init__(children=[*args, *sorted(kwargs.items())], connector=_connector, negated=_negated)
    def _combine(self, other, conn):
        if not(isinstance(other, Q) or getattr(other, 'conditional', False) is True):
            raise TypeError(other)

        if not self:
            return other.copy() if hasattr(other, 'copy') else copy.copy(other)
        elif isinstance(other, Q) and not other:
            _, args, kwargs = self.deconstruct()
            return type(self)(*args, **kwargs)

        obj = type(self)()
        obj.connector = conn
        obj.add(self, conn)
        obj.add(other, conn)
        return obj
    def deconstruct(self):
        path = '%s.%s' % (self.__class__.__module__, self.__class__.__name__)
        if path.startswith('django.db.models.query_utils'):
            path = path.replace('django.db.models.query_utils', 'django.db.models')
        args = tuple(self.children)
        kwargs = {}
        if self.connector != self.default:
            kwargs['_connector'] = self.connector
        if self.negated:
            kwargs['_negated'] = True
        return path, args, kwargs