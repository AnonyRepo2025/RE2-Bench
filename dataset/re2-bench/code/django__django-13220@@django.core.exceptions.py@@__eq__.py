import operator
from django.utils.hashable import make_hashable

NON_FIELD_ERRORS = '__all__'

class ValidationError(Exception):
    def __eq__(self, other):
        if not isinstance(other, ValidationError):
            return NotImplemented
        return hash(self) == hash(other)