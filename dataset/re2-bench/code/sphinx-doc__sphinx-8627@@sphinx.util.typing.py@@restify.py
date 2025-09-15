import sys
import typing
from struct import Struct
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, TypeVar, Union
from docutils import nodes
from docutils.parsers.rst.states import Inliner
from typing import ForwardRef
from typing import _ForwardRef  # type: ignore
from typing import Type  # NOQA # for python3.5.1
from sphinx.util.inspect import safe_getattr  # lazy loading
from sphinx.util import inspect  # lazy loading
from sphinx.util import inspect  # lazy loading
from sphinx.util import inspect  # lazy loading

DirectiveOption = Callable[[str], Any]
TextlikeNode = Union[nodes.Text, nodes.TextElement]
NoneType = type(None)
PathMatcher = Callable[[str], bool]
RoleFunction = Callable[[str, str, str, int, Inliner, Dict[str, Any], List[str]],
                        Tuple[List[nodes.Node], List[nodes.system_message]]]
TitleGetter = Callable[[nodes.Node], str]
Inventory = Dict[str, Dict[str, Tuple[str, str, str, str]]]

def restify(cls: Optional["Type"]) -> str:
    """Convert python class to a reST reference."""
    from sphinx.util import inspect  # lazy loading

    if cls is None or cls is NoneType:
        return ':obj:`None`'
    elif cls is Ellipsis:
        return '...'
    elif cls is Struct:
        # Before Python 3.9, struct.Struct class has incorrect __module__.
        return ':class:`struct.Struct`'
    elif inspect.isNewType(cls):
        return ':class:`%s`' % cls.__name__
    elif cls.__module__ in ('__builtin__', 'builtins'):
        return ':class:`%s`' % cls.__name__
    else:
        if sys.version_info >= (3, 7):  # py37+
            return _restify_py37(cls)
        else:
            return _restify_py36(cls)
