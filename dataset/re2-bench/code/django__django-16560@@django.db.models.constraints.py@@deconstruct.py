import warnings
from enum import Enum
from types import NoneType
from django.core.exceptions import FieldError, ValidationError
from django.db import connections
from django.db.models.expressions import Exists, ExpressionList, F, OrderBy
from django.db.models.indexes import IndexExpression
from django.db.models.lookups import Exact
from django.db.models.query_utils import Q
from django.db.models.sql.query import Query
from django.db.utils import DEFAULT_DB_ALIAS
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.translation import gettext_lazy as _

__all__ = ["BaseConstraint", "CheckConstraint", "Deferrable", "UniqueConstraint"]

class BaseConstraint:
    default_violation_error_message = _("Constraint “%(name)s” is violated.")
    violation_error_code = None
    violation_error_message = None

    # RemovedInDjango60Warning: When the deprecation ends, replace with:
    # def __init__(
    #     self, *, name, violation_error_code=None, violation_error_message=None
    # ):
    def __init__(
        self, *args, name=None, violation_error_code=None, violation_error_message=None
    ):
        # RemovedInDjango60Warning.
        if name is None and not args:
            raise TypeError(
                f"{self.__class__.__name__}.__init__() missing 1 required keyword-only "
                f"argument: 'name'"
            )
        self.name = name
        if violation_error_code is not None:
            self.violation_error_code = violation_error_code
        if violation_error_message is not None:
            self.violation_error_message = violation_error_message
        else:
            self.violation_error_message = self.default_violation_error_message
        # RemovedInDjango60Warning.
        if args:
            warnings.warn(
                f"Passing positional arguments to {self.__class__.__name__} is "
                f"deprecated.",
                RemovedInDjango60Warning,
                stacklevel=2,
            )
            for arg, attr in zip(args, ["name", "violation_error_message"]):
                if arg:
                    setattr(self, attr, arg)

    @property
    def contains_expressions(self):
        return False

    def constraint_sql(self, model, schema_editor):
        raise NotImplementedError("This method must be implemented by a subclass.")

    def create_sql(self, model, schema_editor):
        raise NotImplementedError("This method must be implemented by a subclass.")

    def remove_sql(self, model, schema_editor):
        raise NotImplementedError("This method must be implemented by a subclass.")

    def validate(self, model, instance, exclude=None, using=DEFAULT_DB_ALIAS):
        raise NotImplementedError("This method must be implemented by a subclass.")

    def get_violation_error_message(self):
        return self.violation_error_message % {"name": self.name}

    def deconstruct(self):
        path = "%s.%s" % (self.__class__.__module__, self.__class__.__name__)
        path = path.replace("django.db.models.constraints", "django.db.models")
        kwargs = {"name": self.name}
        if (
            self.violation_error_message is not None
            and self.violation_error_message != self.default_violation_error_message
        ):
            kwargs["violation_error_message"] = self.violation_error_message
        if self.violation_error_code is not None:
            kwargs["violation_error_code"] = self.violation_error_code
        return (path, (), kwargs)

    def clone(self):
        _, args, kwargs = self.deconstruct()
        return self.__class__(*args, **kwargs)
