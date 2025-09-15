import copy
import inspect
import warnings
from functools import partialmethod
from itertools import chain
import django
from django.apps import apps
from django.conf import settings
from django.core import checks
from django.core.exceptions import (
    NON_FIELD_ERRORS, FieldDoesNotExist, FieldError, MultipleObjectsReturned,
    ObjectDoesNotExist, ValidationError,
)
from django.db import (
    DEFAULT_DB_ALIAS, DJANGO_VERSION_PICKLE_KEY, DatabaseError, connection,
    connections, router, transaction,
)
from django.db.models import (
    NOT_PROVIDED, ExpressionWrapper, IntegerField, Max, Value,
)
from django.db.models.constants import LOOKUP_SEP
from django.db.models.constraints import CheckConstraint, UniqueConstraint
from django.db.models.deletion import CASCADE, Collector
from django.db.models.fields.related import (
    ForeignObjectRel, OneToOneField, lazy_related_operation, resolve_relation,
)
from django.db.models.functions import Coalesce
from django.db.models.manager import Manager
from django.db.models.options import Options
from django.db.models.query import F, Q
from django.db.models.signals import (
    class_prepared, post_init, post_save, pre_init, pre_save,
)
from django.db.models.utils import make_model_tuple
from django.utils.encoding import force_str
from django.utils.hashable import make_hashable
from django.utils.text import capfirst, get_text_list
from django.utils.translation import gettext_lazy as _
from django.db import models

DEFERRED = Deferred()
model_unpickle.__safe_for_unpickle__ = True

class Model:
    pk = property(_get_pk_val, _set_pk_val)
    save.alters_data = True
    save_base.alters_data = True
    delete.alters_data = True
    def __getstate__(self):
        """Hook to allow choosing the attributes to pickle."""
        state = self.__dict__.copy()
        state['_state'] = copy.copy(state['_state'])
        state['_state'].fields_cache = state['_state'].fields_cache.copy()
        return state