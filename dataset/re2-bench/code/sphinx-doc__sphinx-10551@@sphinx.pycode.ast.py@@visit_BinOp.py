import sys
from typing import Dict, List, Optional, Type, overload
import ast
from typed_ast import ast3 as ast
import ast  # type: ignore



class _UnparseVisitor(NodeVisitor):
    def visit_BinOp(self, node: ast.BinOp) -> str:
        # Special case ``**`` to not have surrounding spaces.
        if isinstance(node.op, ast.Pow):
            return "".join(map(self.visit, (node.left, node.op, node.right)))
        return " ".join(self.visit(e) for e in [node.left, node.op, node.right])