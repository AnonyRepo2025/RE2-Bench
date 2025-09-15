

import functools
from importlib import import_module
from inspect import getfullargspec, unwrap
from django.utils.html import conditional_escape
from django.utils.itercompat import is_iterable
from .base import Node, Template, token_kwargs
from .exceptions import TemplateSyntaxError

class Library:

    def __init__(self):
        self.filters = {}
        self.tags = {}

    def tag(self, name=None, compile_function=None):
        if name is None and compile_function is None:
            return self.tag_function
        elif name is not None and compile_function is None:
            if callable(name):
                return self.tag_function(name)
            else:

                def dec(func):
                    return self.tag(name, func)
                return dec
        elif name is not None and compile_function is not None:
            self.tags[name] = compile_function
            return compile_function
        else:
            raise ValueError('Unsupported arguments to Library.tag: (%r, %r)' % (name, compile_function))

    def tag_function(self, func):
        self.tags[getattr(func, '_decorated_function', func).__name__] = func
        return func

    def filter(self, name=None, filter_func=None, **flags):
        if name is None and filter_func is None:

            def dec(func):
                return self.filter_function(func, **flags)
            return dec
        elif name is not None and filter_func is None:
            if callable(name):
                return self.filter_function(name, **flags)
            else:

                def dec(func):
                    return self.filter(name, func, **flags)
                return dec
        elif name is not None and filter_func is not None:
            self.filters[name] = filter_func
            for attr in ('expects_localtime', 'is_safe', 'needs_autoescape'):
                if attr in flags:
                    value = flags[attr]
                    setattr(filter_func, attr, value)
                    if hasattr(filter_func, '_decorated_function'):
                        setattr(filter_func._decorated_function, attr, value)
            filter_func._filter_name = name
            return filter_func
        else:
            raise ValueError('Unsupported arguments to Library.filter: (%r, %r)' % (name, filter_func))

    def filter_function(self, func, **flags):
        name = getattr(func, '_decorated_function', func).__name__
        return self.filter(name, func, **flags)

    def simple_tag(self, func=None, takes_context=None, name=None):

        def dec(func):
            params, varargs, varkw, defaults, kwonly, kwonly_defaults, _ = getfullargspec(unwrap(func))
            function_name = name or getattr(func, '_decorated_function', func).__name__

            @functools.wraps(func)
            def compile_func(parser, token):
                bits = token.split_contents()[1:]
                target_var = None
                if len(bits) >= 2 and bits[-2] == 'as':
                    target_var = bits[-1]
                    bits = bits[:-2]
                args, kwargs = parse_bits(parser, bits, params, varargs, varkw, defaults, kwonly, kwonly_defaults, takes_context, function_name)
                return SimpleNode(func, takes_context, args, kwargs, target_var)
            self.tag(function_name, compile_func)
            return func
        if func is None:
            return dec
        elif callable(func):
            return dec(func)
        else:
            raise ValueError('Invalid arguments provided to simple_tag')

    def inclusion_tag(self, filename, func=None, takes_context=None, name=None):

        def dec(func):
            params, varargs, varkw, defaults, kwonly, kwonly_defaults, _ = getfullargspec(unwrap(func))
            function_name = name or getattr(func, '_decorated_function', func).__name__

            @functools.wraps(func)
            def compile_func(parser, token):
                bits = token.split_contents()[1:]
                args, kwargs = parse_bits(parser, bits, params, varargs, varkw, defaults, kwonly, kwonly_defaults, takes_context, function_name)
                return InclusionNode(func, takes_context, args, kwargs, filename)
            self.tag(function_name, compile_func)
            return func
        return dec