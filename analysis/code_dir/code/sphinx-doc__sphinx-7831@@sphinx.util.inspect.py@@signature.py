import builtins
import contextlib
import enum
import inspect
import re
import sys
import types
import typing
import warnings
from functools import partial, partialmethod
from inspect import Parameter, isclass, ismethod, ismethoddescriptor, ismodule
from io import StringIO
from typing import Any, Callable, Mapping, List, Optional, Tuple
from typing import cast
from sphinx.deprecation import RemovedInSphinx40Warning, RemovedInSphinx50Warning
from sphinx.pycode.ast import ast
from sphinx.pycode.ast import unparse as ast_unparse
from sphinx.util import logging
from sphinx.util.typing import stringify as stringify_annotation
from types import ClassMethodDescriptorType, MethodDescriptorType, WrapperDescriptorType
from functools import singledispatchmethod
logger = logging.getLogger(__name__)
memory_address_re = re.compile(' at 0x[0-9a-f]{8,16}(?=>)', re.IGNORECASE)

def signature(subject: Callable, bound_method: bool=False, follow_wrapped: bool=False) -> inspect.Signature:
    try:
        try:
            if _should_unwrap(subject):
                signature = inspect.signature(subject)
            else:
                signature = inspect.signature(subject, follow_wrapped=follow_wrapped)
        except ValueError:
            signature = inspect.signature(subject)
        parameters = list(signature.parameters.values())
        return_annotation = signature.return_annotation
    except IndexError:
        if hasattr(subject, '_partialmethod'):
            parameters = []
            return_annotation = Parameter.empty
        else:
            raise
    try:
        annotations = typing.get_type_hints(subject)
        for i, param in enumerate(parameters):
            if isinstance(param.annotation, str) and param.name in annotations:
                parameters[i] = param.replace(annotation=annotations[param.name])
        if 'return' in annotations:
            return_annotation = annotations['return']
    except Exception:
        pass
    if bound_method:
        if inspect.ismethod(subject):
            pass
        elif len(parameters) > 0:
            parameters.pop(0)
    return inspect.Signature(parameters, return_annotation=return_annotation)