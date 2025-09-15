import functools
import inspect
import warnings
from collections import namedtuple
from django.core.exceptions import FieldDoesNotExist, FieldError
from django.db.models.constants import LOOKUP_SEP
from django.utils import tree
from django.utils.deprecation import RemovedInDjango40Warning
from django.db.models.lookups import Lookup
from django.db.models.lookups import Transform

PathInfo = namedtuple('PathInfo', 'from_opts to_opts target_fields join_field m2m direct filtered_relation')

class Q(Node):
    AND = 'AND'
    OR = 'OR'
    default = AND
    conditional = True
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