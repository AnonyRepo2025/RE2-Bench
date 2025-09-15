import copy
import inspect
import warnings
from functools import partialmethod
from itertools import chain
import django
from django.apps import apps
from django.conf import settings
from django.core import checks
from django.core.exceptions import NON_FIELD_ERRORS, FieldDoesNotExist, FieldError, MultipleObjectsReturned, ObjectDoesNotExist, ValidationError
from django.db import DEFAULT_DB_ALIAS, DJANGO_VERSION_PICKLE_KEY, DatabaseError, connection, connections, router, transaction
from django.db.models import NOT_PROVIDED, ExpressionWrapper, IntegerField, Max, Value
from django.db.models.constants import LOOKUP_SEP
from django.db.models.constraints import CheckConstraint, UniqueConstraint
from django.db.models.deletion import CASCADE, Collector
from django.db.models.fields.related import ForeignObjectRel, OneToOneField, lazy_related_operation, resolve_relation
from django.db.models.functions import Coalesce
from django.db.models.manager import Manager
from django.db.models.options import Options
from django.db.models.query import F, Q
from django.db.models.signals import class_prepared, post_init, post_save, pre_init, pre_save
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

    def __init__(self, *args, **kwargs):
        cls = self.__class__
        opts = self._meta
        _setattr = setattr
        _DEFERRED = DEFERRED
        if opts.abstract:
            raise TypeError('Abstract models cannot be instantiated.')
        pre_init.send(sender=cls, args=args, kwargs=kwargs)
        self._state = ModelState()
        if len(args) > len(opts.concrete_fields):
            raise IndexError('Number of args exceeds number of fields')
        if not kwargs:
            fields_iter = iter(opts.concrete_fields)
            for val, field in zip(args, fields_iter):
                if val is _DEFERRED:
                    continue
                _setattr(self, field.attname, val)
        else:
            fields_iter = iter(opts.fields)
            for val, field in zip(args, fields_iter):
                if val is _DEFERRED:
                    continue
                _setattr(self, field.attname, val)
                if kwargs.pop(field.name, NOT_PROVIDED) is not NOT_PROVIDED:
                    raise TypeError(f"{cls.__qualname__}() got both positional and keyword arguments for field '{field.name}'.")
        for field in fields_iter:
            is_related_object = False
            if field.attname not in kwargs and field.column is None:
                continue
            if kwargs:
                if isinstance(field.remote_field, ForeignObjectRel):
                    try:
                        rel_obj = kwargs.pop(field.name)
                        is_related_object = True
                    except KeyError:
                        try:
                            val = kwargs.pop(field.attname)
                        except KeyError:
                            val = field.get_default()
                else:
                    try:
                        val = kwargs.pop(field.attname)
                    except KeyError:
                        val = field.get_default()
            else:
                val = field.get_default()
            if is_related_object:
                if rel_obj is not _DEFERRED:
                    _setattr(self, field.name, rel_obj)
            elif val is not _DEFERRED:
                _setattr(self, field.attname, val)
        if kwargs:
            property_names = opts._property_names
            for prop in tuple(kwargs):
                try:
                    if prop in property_names or opts.get_field(prop):
                        if kwargs[prop] is not _DEFERRED:
                            _setattr(self, prop, kwargs[prop])
                        del kwargs[prop]
                except (AttributeError, FieldDoesNotExist):
                    pass
            for kwarg in kwargs:
                raise TypeError("%s() got an unexpected keyword argument '%s'" % (cls.__name__, kwarg))
        super().__init__()
        post_init.send(sender=cls, instance=self)

    @classmethod
    def from_db(cls, db, field_names, values):
        if len(values) != len(cls._meta.concrete_fields):
            values_iter = iter(values)
            values = [next(values_iter) if f.attname in field_names else DEFERRED for f in cls._meta.concrete_fields]
        new = cls(*values)
        new._state.adding = False
        new._state.db = db
        return new

    def __hash__(self):
        if self.pk is None:
            raise TypeError('Model instances without primary key value are unhashable')
        return hash(self.pk)

    def _get_pk_val(self, meta=None):
        meta = meta or self._meta
        return getattr(self, meta.pk.attname)

    def delete(self, using=None, keep_parents=False):
        if self.pk is None:
            raise ValueError("%s object can't be deleted because its %s attribute is set to None." % (self._meta.object_name, self._meta.pk.attname))
        using = using or router.db_for_write(self.__class__, instance=self)
        collector = Collector(using=using, origin=self)
        collector.collect([self], keep_parents=keep_parents)
        return collector.delete()