import ast
import enum
import re
import types
from typing import Callable
from typing import Iterator
from typing import Mapping
from typing import Optional
from typing import Sequence
import attr
from _pytest.compat import TYPE_CHECKING
from typing import NoReturn

__all__ = [
    "Expression",
    "ParseError",
]
IDENT_PREFIX = "$"

def not_expr(s: Scanner) -> ast.expr:
    if s.accept(TokenType.NOT):
        return ast.UnaryOp(ast.Not(), not_expr(s))
    if s.accept(TokenType.LPAREN):
        ret = expr(s)
        s.accept(TokenType.RPAREN, reject=True)
        return ret
    ident = s.accept(TokenType.IDENT)
    if ident:
        return ast.Name(IDENT_PREFIX + ident.value, ast.Load())
    s.reject((TokenType.NOT, TokenType.LPAREN, TokenType.IDENT))
