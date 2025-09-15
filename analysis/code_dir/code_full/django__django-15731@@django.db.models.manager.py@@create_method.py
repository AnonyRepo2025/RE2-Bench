

import copy
import inspect
from functools import wraps
from importlib import import_module
from django.db import router
from django.db.models.query import QuerySet

class BaseManager:
    creation_counter = 0
    auto_created = False
    use_in_migrations = False

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj._constructor_args = (args, kwargs)
        return obj

    def __init__(self):
        super().__init__()
        self._set_creation_counter()
        self.model = None
        self.name = None
        self._db = None
        self._hints = {}

    def __str__(self):
        return '%s.%s' % (self.model._meta.label, self.name)

    def __class_getitem__(cls, *args, **kwargs):
        return cls

    def deconstruct(self):
        qs_class = self._queryset_class
        if getattr(self, '_built_with_as_manager', False):
            return (True, None, '%s.%s' % (qs_class.__module__, qs_class.__name__), None, None)
        else:
            module_name = self.__module__
            name = self.__class__.__name__
            module = import_module(module_name)
            if not hasattr(module, name):
                raise ValueError("Could not find manager %s in %s.\nPlease note that you need to inherit from managers you dynamically generated with 'from_queryset()'." % (name, module_name))
            return (False, '%s.%s' % (module_name, name), None, self._constructor_args[0], self._constructor_args[1])

    def check(self, **kwargs):
        return []

    @classmethod
    def _get_queryset_methods(cls, queryset_class):

        def create_method(name, method):

            @wraps(method)
            def manager_method(self, *args, **kwargs):
                return getattr(self.get_queryset(), name)(*args, **kwargs)
            return manager_method
        new_methods = {}
        for name, method in inspect.getmembers(queryset_class, predicate=inspect.isfunction):
            if hasattr(cls, name):
                continue
            queryset_only = getattr(method, 'queryset_only', None)
            if queryset_only or (queryset_only is None and name.startswith('_')):
                continue
            new_methods[name] = create_method(name, method)
        return new_methods

    @classmethod
    def from_queryset(cls, queryset_class, class_name=None):
        if class_name is None:
            class_name = '%sFrom%s' % (cls.__name__, queryset_class.__name__)
        return type(class_name, (cls,), {'_queryset_class': queryset_class, **cls._get_queryset_methods(queryset_class)})

    def contribute_to_class(self, cls, name):
        self.name = self.name or name
        self.model = cls
        setattr(cls, name, ManagerDescriptor(self))
        cls._meta.add_manager(self)

    def _set_creation_counter(self):
        self.creation_counter = BaseManager.creation_counter
        BaseManager.creation_counter += 1

    def db_manager(self, using=None, hints=None):
        obj = copy.copy(self)
        obj._db = using or self._db
        obj._hints = hints or self._hints
        return obj

    @property
    def db(self):
        return self._db or router.db_for_read(self.model, **self._hints)

    def get_queryset(self):
        return self._queryset_class(model=self.model, using=self._db, hints=self._hints)

    def all(self):
        return self.get_queryset()

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self._constructor_args == other._constructor_args

    def __hash__(self):
        return id(self)