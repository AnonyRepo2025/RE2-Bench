import sys
from typing import Dict, List, Type, Optional
import ast
from typed_ast import ast3 as ast
import ast  # type: ignore

OPERATORS = {
    ast.Add: "+",
    ast.And: "and",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Invert: "~",
    ast.LShift: "<<",
    ast.MatMult: "@",
    ast.Mult: "*",
    ast.Mod: "%",
    ast.Not: "not",
    ast.Pow: "**",
    ast.Or: "or",
    ast.RShift: ">>",
    ast.Sub: "-",
    ast.UAdd: "+",
    ast.USub: "-",
}  # type: Dict[Type[ast.AST], str]

        def visit_Constant(self, node: ast.Constant) -> str:
            if node.value is Ellipsis:
                return "..."
            elif isinstance(node.value, (int, float, complex)):
                if self.code and sys.version_info > (3, 8):
                    return ast.get_source_segment(self.code, node)
                else:
                    return repr(node.value)
            else:
                return repr(node.value)
