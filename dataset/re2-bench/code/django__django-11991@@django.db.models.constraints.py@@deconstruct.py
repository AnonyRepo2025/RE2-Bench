from enum import Enum
from django.db.models.query_utils import Q
from django.db.models.sql.query import Query

__all__ = ['CheckConstraint', 'Deferrable', 'UniqueConstraint']

class UniqueConstraint(BaseConstraint):
    def deconstruct(self):
        path, args, kwargs = super().deconstruct()
        kwargs['fields'] = self.fields
        if self.condition:
            kwargs['condition'] = self.condition
        if self.deferrable:
            kwargs['deferrable'] = self.deferrable
        if self.include:
            kwargs['include'] = self.include
        return path, args, kwargs