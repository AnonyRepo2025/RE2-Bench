import sys
import typing
from struct import Struct
from types import TracebackType
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, TypeVar, Union
from docutils import nodes
from docutils.parsers.rst.states import Inliner
from sphinx.deprecation import RemovedInSphinx60Warning, deprecated_alias
from typing import ForwardRef
from typing import _ForwardRef
from types import Union as types_Union
from typing import Type
from sphinx.util.inspect import safe_getattr
from sphinx.util import inspect
from sphinx.util import inspect
from sphinx.util import inspect
INVALID_BUILTIN_CLASSES = {Struct: 'struct.Struct', TracebackType: 'types.TracebackType'}
TextlikeNode = Union[nodes.Text, nodes.TextElement]
NoneType = type(None)
PathMatcher = Callable[[str], bool]
RoleFunction = Callable[[str, str, str, int, Inliner, Dict[str, Any], List[str]], Tuple[List[nodes.Node], List[nodes.system_message]]]
OptionSpec = Dict[str, Callable[[str], Any]]
TitleGetter = Callable[[nodes.Node], str]
Inventory = Dict[str, Dict[str, Tuple[str, str, str, str]]]

def stringify(annotation: Any) -> str:
    from sphinx.util import inspect
    if isinstance(annotation, str):
        if annotation.startswith("'") and annotation.endswith("'"):
            return annotation[1:-1]
        else:
            return annotation
    elif isinstance(annotation, TypeVar):
        if annotation.__module__ == 'typing':
            return annotation.__name__
        else:
            return '.'.join([annotation.__module__, annotation.__name__])
    elif inspect.isNewType(annotation):
        return annotation.__name__
    elif not annotation:
        return repr(annotation)
    elif annotation is NoneType:
        return 'None'
    elif annotation in INVALID_BUILTIN_CLASSES:
        return INVALID_BUILTIN_CLASSES[annotation]
    elif getattr(annotation, '__module__', None) == 'builtins' and hasattr(annotation, '__qualname__'):
        if hasattr(annotation, '__args__'):
            return repr(annotation)
        else:
            return annotation.__qualname__
    elif annotation is Ellipsis:
        return '...'
    if sys.version_info >= (3, 7):
        return _stringify_py37(annotation)
    else:
        return _stringify_py36(annotation)