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

def stringify(annotation: Any) -> str:
    """Stringify type annotation object."""
    from sphinx.util import inspect  # lazy loading

    if isinstance(annotation, str):
        if annotation.startswith("'") and annotation.endswith("'"):
            # might be a double Forward-ref'ed type.  Go unquoting.
            return annotation[1:-1]
        else:
            return annotation
    elif isinstance(annotation, TypeVar):
        return annotation.__name__
    elif inspect.isNewType(annotation):
        # Could not get the module where it defiend
        return annotation.__name__
    elif not annotation:
        return repr(annotation)
    elif annotation is NoneType:
        return 'None'
    elif (getattr(annotation, '__module__', None) == 'builtins' and
          hasattr(annotation, '__qualname__')):
        return annotation.__qualname__
    elif annotation is Ellipsis:
        return '...'
    elif annotation is Struct:
        # Before Python 3.9, struct.Struct class has incorrect __module__.
        return 'struct.Struct'

    if sys.version_info >= (3, 7):  # py37+
        return _stringify_py37(annotation)
    else:
        return _stringify_py36(annotation)
