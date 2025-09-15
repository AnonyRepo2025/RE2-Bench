import sys
import typing
from struct import Struct
from types import TracebackType
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Type, TypeVar, Union
from docutils import nodes
from docutils.parsers.rst.states import Inliner
from sphinx.deprecation import RemovedInSphinx60Warning, deprecated_alias
from typing import ForwardRef
from typing import _ForwardRef  # type: ignore
from types import UnionType  # type: ignore  # python 3.10 or above
from typing import Type  # NOQA # for python3.5.1
from sphinx.util.inspect import safe_getattr  # lazy loading
from sphinx.util import inspect  # lazy loading
from sphinx.util import inspect  # lazy loading
from sphinx.util import inspect  # lazy loading

INVALID_BUILTIN_CLASSES = {
    Struct: 'struct.Struct',  # Before Python 3.9
    TracebackType: 'types.TracebackType',
}
TextlikeNode = Union[nodes.Text, nodes.TextElement]
NoneType = type(None)
PathMatcher = Callable[[str], bool]
RoleFunction = Callable[[str, str, str, int, Inliner, Dict[str, Any], List[str]],
                        Tuple[List[nodes.Node], List[nodes.system_message]]]
OptionSpec = Dict[str, Callable[[str], Any]]
TitleGetter = Callable[[nodes.Node], str]
Inventory = Dict[str, Dict[str, Tuple[str, str, str, str]]]

def restify(cls: Optional[Type]) -> str:
    """Convert python class to a reST reference."""
    from sphinx.util import inspect  # lazy loading

    try:
        if cls is None or cls is NoneType:
            return ':py:obj:`None`'
        elif cls is Ellipsis:
            return '...'
        elif isinstance(cls, str):
            return cls
        elif cls in INVALID_BUILTIN_CLASSES:
            return ':py:class:`%s`' % INVALID_BUILTIN_CLASSES[cls]
        elif inspect.isNewType(cls):
            if sys.version_info > (3, 10):
                # newtypes have correct module info since Python 3.10+
                print(cls, type(cls), dir(cls))
                return ':py:class:`%s.%s`' % (cls.__module__, cls.__name__)
            else:
                return ':py:class:`%s`' % cls.__name__
        elif UnionType and isinstance(cls, UnionType):
            if len(cls.__args__) > 1 and None in cls.__args__:
                args = ' | '.join(restify(a) for a in cls.__args__ if a)
                return 'Optional[%s]' % args
            else:
                return ' | '.join(restify(a) for a in cls.__args__)
        elif cls.__module__ in ('__builtin__', 'builtins'):
            if hasattr(cls, '__args__'):
                return ':py:class:`%s`\\ [%s]' % (
                    cls.__name__,
                    ', '.join(restify(arg) for arg in cls.__args__),
                )
            else:
                return ':py:class:`%s`' % cls.__name__
        else:
            if sys.version_info >= (3, 7):  # py37+
                return _restify_py37(cls)
            else:
                return _restify_py36(cls)
    except (AttributeError, TypeError):
        return repr(cls)
