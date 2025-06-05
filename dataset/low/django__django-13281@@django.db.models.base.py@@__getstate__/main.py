import copy
import inspect
import warnings
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