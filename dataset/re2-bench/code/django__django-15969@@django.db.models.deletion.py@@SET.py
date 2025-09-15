from collections import Counter, defaultdict
from functools import partial, reduce
from itertools import chain
from operator import attrgetter, or_
from django.db import IntegrityError, connections, models, transaction
from django.db.models import query_utils, signals, sql

SET_NULL.lazy_sub_objs = True
SET_DEFAULT.lazy_sub_objs = True

def SET(value):
    if callable(value):

        def set_on_delete(collector, field, sub_objs, using):
            collector.add_field_update(field, value(), sub_objs)

    else:

        def set_on_delete(collector, field, sub_objs, using):
            collector.add_field_update(field, value, sub_objs)

    set_on_delete.deconstruct = lambda: ("django.db.models.SET", (value,), {})
    set_on_delete.lazy_sub_objs = True
    return set_on_delete
