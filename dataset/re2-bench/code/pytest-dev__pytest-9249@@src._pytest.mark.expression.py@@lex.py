import ast
import enum
import re
import types
from typing import Callable
from typing import Iterator
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import TYPE_CHECKING
import attr
from typing import NoReturn

__all__ = [
    "Expression",
    "ParseError",
]
IDENT_PREFIX = "$"

class Scanner:
    __slots__ = ("tokens", "current")
    def lex(self, input: str) -> Iterator[Token]:
        pos = 0
        while pos < len(input):
            if input[pos] in (" ", "\t"):
                pos += 1
            elif input[pos] == "(":
                yield Token(TokenType.LPAREN, "(", pos)
                pos += 1
            elif input[pos] == ")":
                yield Token(TokenType.RPAREN, ")", pos)
                pos += 1
            else:
                match = re.match(r"(:?\w|:|\+|-|\.|\[|\]|\\|/)+", input[pos:])
                if match:
                    value = match.group(0)
                    if value == "or":
                        yield Token(TokenType.OR, value, pos)
                    elif value == "and":
                        yield Token(TokenType.AND, value, pos)
                    elif value == "not":
                        yield Token(TokenType.NOT, value, pos)
                    else:
                        yield Token(TokenType.IDENT, value, pos)
                    pos += len(value)
                else:
                    raise ParseError(
                        pos + 1,
                        f'unexpected character "{input[pos]}"',
                    )
        yield Token(TokenType.EOF, "", pos)