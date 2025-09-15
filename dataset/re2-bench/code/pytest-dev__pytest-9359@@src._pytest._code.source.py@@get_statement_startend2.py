import ast
import inspect
import textwrap
import tokenize
import types
import warnings
from bisect import bisect_right
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Union



def get_statement_startend2(lineno: int, node: ast.AST) -> Tuple[int, Optional[int]]:
    # Flatten all statements and except handlers into one lineno-list.
    # AST's line numbers start indexing at 1.
    values: List[int] = []
    for x in ast.walk(node):
        if isinstance(x, (ast.stmt, ast.ExceptHandler)):
            # Before Python 3.8, the lineno of a decorated class or function pointed at the decorator.
            # Since Python 3.8, the lineno points to the class/def, so need to include the decorators.
            if isinstance(x, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                for d in x.decorator_list:
                    values.append(d.lineno - 1)
            values.append(x.lineno - 1)
            for name in ("finalbody", "orelse"):
                val: Optional[List[ast.stmt]] = getattr(x, name, None)
                if val:
                    # Treat the finally/orelse part as its own statement.
                    values.append(val[0].lineno - 1 - 1)
    values.sort()
    insert_index = bisect_right(values, lineno)
    start = values[insert_index - 1]
    if insert_index >= len(values):
        end = None
    else:
        end = values[insert_index]
    return start, end
