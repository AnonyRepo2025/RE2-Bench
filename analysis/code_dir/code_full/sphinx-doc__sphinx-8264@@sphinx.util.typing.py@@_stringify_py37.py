def is_system_TypeVar(typ: Any) -> bool:
    modname = getattr(typ, '__module__', '')
    return modname == 'typing' and isinstance(typ, TypeVar)

def stringify(annotation: Any) -> str:
    if isinstance(annotation, str):
        if annotation.startswith("'") and annotation.endswith("'"):
            return annotation[1:-2]
        else:
            return annotation
    elif isinstance(annotation, TypeVar):
        return annotation.__name__
    elif not annotation:
        return repr(annotation)
    elif annotation is NoneType:
        return 'None'
    elif getattr(annotation, '__module__', None) == 'builtins' and hasattr(annotation, '__qualname__'):
        return annotation.__qualname__
    elif annotation is Ellipsis:
        return '...'
    if sys.version_info >= (3, 7):
        return _stringify_py37(annotation)
    else:
        return _stringify_py36(annotation)



import sys
import typing
from typing import Any, Callable, Dict, Generator, List, Tuple, TypeVar, Union
from docutils import nodes
from docutils.parsers.rst.states import Inliner
from typing import ForwardRef
from typing import _ForwardRef
DirectiveOption = Callable[[str], Any]
TextlikeNode = Union[nodes.Text, nodes.TextElement]
NoneType = type(None)
PathMatcher = Callable[[str], bool]
RoleFunction = Callable[[str, str, str, int, Inliner, Dict[str, Any], List[str]], Tuple[List[nodes.Node], List[nodes.system_message]]]
TitleGetter = Callable[[nodes.Node], str]
Inventory = Dict[str, Dict[str, Tuple[str, str, str, str]]]

def _stringify_py37(annotation: Any) -> str:
    module = getattr(annotation, '__module__', None)
    if module == 'typing':
        if getattr(annotation, '_name', None):
            qualname = annotation._name
        elif getattr(annotation, '__qualname__', None):
            qualname = annotation.__qualname__
        elif getattr(annotation, '__forward_arg__', None):
            qualname = annotation.__forward_arg__
        else:
            qualname = stringify(annotation.__origin__)
    elif hasattr(annotation, '__qualname__'):
        qualname = '%s.%s' % (module, annotation.__qualname__)
    elif hasattr(annotation, '__origin__'):
        qualname = stringify(annotation.__origin__)
    else:
        return repr(annotation)
    if getattr(annotation, '__args__', None):
        if not isinstance(annotation.__args__, (list, tuple)):
            pass
        elif qualname == 'Union':
            if len(annotation.__args__) > 1 and annotation.__args__[-1] is NoneType:
                if len(annotation.__args__) > 2:
                    args = ', '.join((stringify(a) for a in annotation.__args__[:-1]))
                    return 'Optional[Union[%s]]' % args
                else:
                    return 'Optional[%s]' % stringify(annotation.__args__[0])
            else:
                args = ', '.join((stringify(a) for a in annotation.__args__))
                return 'Union[%s]' % args
        elif qualname == 'Callable':
            args = ', '.join((stringify(a) for a in annotation.__args__[:-1]))
            returns = stringify(annotation.__args__[-1])
            return '%s[[%s], %s]' % (qualname, args, returns)
        elif str(annotation).startswith('typing.Annotated'):
            return stringify(annotation.__args__[0])
        elif all((is_system_TypeVar(a) for a in annotation.__args__)):
            return qualname
        else:
            args = ', '.join((stringify(a) for a in annotation.__args__))
            return '%s[%s]' % (qualname, args)
    return qualname