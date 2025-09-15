import sys
from typing import Dict, List, Optional, Type, overload
import ast
from typed_ast import ast3 as ast
import ast  # type: ignore



class _UnparseVisitor(NodeVisitor):
    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        # UnaryOp is one of {UAdd, USub, Invert, Not}, which refer to ``+x``,
        # ``-x``, ``~x``, and ``not x``. Only Not needs a space.
        if isinstance(node.op, ast.Not):
            return "%s %s" % (self.visit(node.op), self.visit(node.operand))
        return "%s%s" % (self.visit(node.op), self.visit(node.operand))