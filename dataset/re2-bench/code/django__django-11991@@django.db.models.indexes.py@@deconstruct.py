from django.db.backends.utils import names_digest, split_identifier
from django.db.models.query_utils import Q
from django.db.models.sql import Query

__all__ = ['Index']

class Index:
    suffix = 'idx'
    max_name_length = 30
    def deconstruct(self):
        path = '%s.%s' % (self.__class__.__module__, self.__class__.__name__)
        path = path.replace('django.db.models.indexes', 'django.db.models')
        kwargs = {'fields': self.fields, 'name': self.name}
        if self.db_tablespace is not None:
            kwargs['db_tablespace'] = self.db_tablespace
        if self.opclasses:
            kwargs['opclasses'] = self.opclasses
        if self.condition:
            kwargs['condition'] = self.condition
        if self.include:
            kwargs['include'] = self.include
        return (path, (), kwargs)