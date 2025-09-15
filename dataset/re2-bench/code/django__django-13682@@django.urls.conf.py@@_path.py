from functools import partial
from importlib import import_module
from django.core.exceptions import ImproperlyConfigured
from .resolvers import (
    LocalePrefixPattern, RegexPattern, RoutePattern, URLPattern, URLResolver,
)
from django.views import View

path = partial(_path, Pattern=RoutePattern)
re_path = partial(_path, Pattern=RegexPattern)

def _path(route, view, kwargs=None, name=None, Pattern=None):
    from django.views import View

    if isinstance(view, (list, tuple)):
        # For include(...) processing.
        pattern = Pattern(route, is_endpoint=False)
        urlconf_module, app_name, namespace = view
        return URLResolver(
            pattern,
            urlconf_module,
            kwargs,
            app_name=app_name,
            namespace=namespace,
        )
    elif callable(view):
        pattern = Pattern(route, name=name, is_endpoint=True)
        return URLPattern(pattern, view, kwargs, name)
    elif isinstance(view, View):
        view_cls_name = view.__class__.__name__
        raise TypeError(
            f'view must be a callable, pass {view_cls_name}.as_view(), not '
            f'{view_cls_name}().'
        )
    else:
        raise TypeError('view must be a callable or a list/tuple in the case of include().')
