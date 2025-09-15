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

class CheckConstraint(BaseConstraint):
    def __eq__(self, other):
        if isinstance(other, CheckConstraint):
            return (
                self.name == other.name
                and self.check == other.check
                and self.violation_error_code == other.violation_error_code
                and self.violation_error_message == other.violation_error_message
            )
        return super().__eq__(other)