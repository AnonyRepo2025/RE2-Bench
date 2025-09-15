import copy
import datetime
import functools
import inspect
from collections import defaultdict
from decimal import Decimal
from uuid import UUID
from django.core.exceptions import EmptyResultSet, FieldError
from django.db import DatabaseError, NotSupportedError, connection
from django.db.models import fields
from django.db.models.constants import LOOKUP_SEP
from django.db.models.query_utils import Q
from django.utils.deconstruct import deconstructible
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable

NoneType = type(None)
_connector_combinations = [
    # Numeric operations - operands of same type.
    {
        connector: [
            (fields.IntegerField, fields.IntegerField, fields.IntegerField),
            (fields.FloatField, fields.FloatField, fields.FloatField),
            (fields.DecimalField, fields.DecimalField, fields.DecimalField),
        ]
        for connector in (
            Combinable.ADD,
            Combinable.SUB,
            Combinable.MUL,
            # Behavior for DIV with integer arguments follows Postgres/SQLite,
            # not MySQL/Oracle.
            Combinable.DIV,
            Combinable.MOD,
            Combinable.POW,
        )
    },
    # Numeric operations - operands of different type.
    {
        connector: [
            (fields.IntegerField, fields.DecimalField, fields.DecimalField),
            (fields.DecimalField, fields.IntegerField, fields.DecimalField),
            (fields.IntegerField, fields.FloatField, fields.FloatField),
            (fields.FloatField, fields.IntegerField, fields.FloatField),
        ]
        for connector in (
            Combinable.ADD,
            Combinable.SUB,
            Combinable.MUL,
            Combinable.DIV,
        )
    },
    # Bitwise operators.
    {
        connector: [
            (fields.IntegerField, fields.IntegerField, fields.IntegerField),
        ]
        for connector in (
            Combinable.BITAND,
            Combinable.BITOR,
            Combinable.BITLEFTSHIFT,
            Combinable.BITRIGHTSHIFT,
            Combinable.BITXOR,
        )
    },
    # Numeric with NULL.
    {
        connector: [
            (field_type, NoneType, field_type),
            (NoneType, field_type, field_type),
        ]
        for connector in (
            Combinable.ADD,
            Combinable.SUB,
            Combinable.MUL,
            Combinable.DIV,
            Combinable.MOD,
            Combinable.POW,
        )
        for field_type in (fields.IntegerField, fields.DecimalField, fields.FloatField)
    },
    # Date/DateTimeField/DurationField/TimeField.
    {
        Combinable.ADD: [
            # Date/DateTimeField.
            (fields.DateField, fields.DurationField, fields.DateTimeField),
            (fields.DateTimeField, fields.DurationField, fields.DateTimeField),
            (fields.DurationField, fields.DateField, fields.DateTimeField),
            (fields.DurationField, fields.DateTimeField, fields.DateTimeField),
            # DurationField.
            (fields.DurationField, fields.DurationField, fields.DurationField),
            # TimeField.
            (fields.TimeField, fields.DurationField, fields.TimeField),
            (fields.DurationField, fields.TimeField, fields.TimeField),
        ],
    },
    {
        Combinable.SUB: [
            # Date/DateTimeField.
            (fields.DateField, fields.DurationField, fields.DateTimeField),
            (fields.DateTimeField, fields.DurationField, fields.DateTimeField),
            (fields.DateField, fields.DateField, fields.DurationField),
            (fields.DateField, fields.DateTimeField, fields.DurationField),
            (fields.DateTimeField, fields.DateField, fields.DurationField),
            (fields.DateTimeField, fields.DateTimeField, fields.DurationField),
            # DurationField.
            (fields.DurationField, fields.DurationField, fields.DurationField),
            # TimeField.
            (fields.TimeField, fields.DurationField, fields.TimeField),
            (fields.TimeField, fields.TimeField, fields.DurationField),
        ],
    },
]
_connector_combinators = defaultdict(list)

class BaseExpression:
    empty_result_set_value = NotImplemented
    is_summary = False
    _output_field_resolved_to_none = False
    filterable = True
    window_compatible = False
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("convert_value", None)
        return state
    def get_source_expressions(self):
        return []
    def set_source_expressions(self, exprs):
        assert not exprs
    def copy(self):
        return copy.copy(self)
    def prefix_references(self, prefix):
        clone = self.copy()
        clone.set_source_expressions(
            [
                F(f"{prefix}{expr.name}")
                if isinstance(expr, F)
                else expr.prefix_references(prefix)
                for expr in self.get_source_expressions()
            ]
        )
        return clone