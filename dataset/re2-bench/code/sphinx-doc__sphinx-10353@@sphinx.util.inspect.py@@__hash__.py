import builtins
import contextlib
import enum
import inspect
import re
import sys
import types
import typing
from functools import partial, partialmethod
from importlib import import_module
from inspect import Parameter, isclass, ismethod, ismethoddescriptor, ismodule  # NOQA
from io import StringIO
from types import MethodType, ModuleType
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Type, cast
from sphinx.pycode.ast import ast  # for py36-37
from sphinx.pycode.ast import unparse as ast_unparse
from sphinx.util import logging
from sphinx.util.typing import ForwardRef
from sphinx.util.typing import stringify as stringify_annotation
from types import ClassMethodDescriptorType, MethodDescriptorType, WrapperDescriptorType
from typing import Type  # NOQA
from functools import singledispatchmethod  # type: ignore
from functools import cached_property  # cached_property is available since py3.8

logger = logging.getLogger(__name__)
memory_address_re = re.compile(r' at 0x[0-9a-f]{8,16}(?=>)', re.IGNORECASE)

class TypeAliasForwardRef:
    def __hash__(self) -> int:
        return hash(self.name)