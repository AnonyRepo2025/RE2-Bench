from collections import namedtuple
from django.db.models.fields.related import RECURSIVE_RELATIONSHIP_CONSTANT

FieldReference = namedtuple('FieldReference', 'to through')

def resolve_relation(model, app_label=None, model_name=None):
    """
    Turn a model class or model reference string and return a model tuple.

    app_label and model_name are used to resolve the scope of recursive and
    unscoped model relationship.
    """
    if isinstance(model, str):
        if model == RECURSIVE_RELATIONSHIP_CONSTANT:
            if app_label is None or model_name is None:
                raise TypeError(
                    'app_label and model_name must be provided to resolve '
                    'recursive relationships.'
                )
            return app_label, model_name
        if '.' in model:
            app_label, model_name = model.split('.', 1)
            return app_label, model_name.lower()
        if app_label is None:
            raise TypeError(
                'app_label must be provided to resolve unscoped model '
                'relationships.'
            )
        return app_label, model.lower()
    return model._meta.app_label, model._meta.model_name
