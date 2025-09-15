from __future__ import annotations
import ast
import inspect
import types
import warnings
from typing import TYPE_CHECKING
import sphinx
from sphinx.deprecation import RemovedInSphinx90Warning
from sphinx.locale import __
from sphinx.pycode.ast import unparse as ast_unparse
from sphinx.util import logging
from typing import Any
from sphinx.application import Sphinx

logger = logging.getLogger(__name__)
_LAMBDA_NAME = (lambda: None).__name__

def _get_arguments(obj: Any, /) -> ast.arguments | None:
    """Parse 'ast.arguments' from an object.

    This tries to parse the original code for an object and returns
    an 'ast.arguments' node.
    """
    try:
        source = inspect.getsource(obj)
        if source.startswith((' ', '\t')):
            # 'obj' is in some indented block.
            module = ast.parse('if True:\n' + source)
            subject = module.body[0].body[0]  # type: ignore[attr-defined]
        else:
            module = ast.parse(source)
            subject = module.body[0]
    except (OSError, TypeError):
        # bail; failed to load source for 'obj'.
        return None
    except SyntaxError:
        if _is_lambda(obj):
            # Most likely a multi-line arising from detecting a lambda, e.g.:
            #
            # class Egg:
            #     x = property(
            #         lambda self: 1, doc="...")
            return None

        # Other syntax errors that are not due to the fact that we are
        # documenting a lambda function are propagated
        # (in particular if a lambda is renamed by the user).
        raise

    return _get_arguments_inner(subject)
