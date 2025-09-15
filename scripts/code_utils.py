import ast
from typing import Optional, Union, Iterable
class _DocstringStripper(ast.NodeTransformer):
    """Remove first-statement string literals used as docstrings."""
    def _strip_in(self, node):
        if node.body and isinstance(node.body[0], ast.Expr):
            v = node.body[0].value
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                node.body = node.body[1:]
        return node

    def visit_Module(self, node):
        self.generic_visit(node)
        return self._strip_in(node)

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        return self._strip_in(node)

    def visit_AsyncFunctionDef(self, node):
        self.generic_visit(node)
        return self._strip_in(node)

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        return self._strip_in(node)

def strip_comments_and_docstrings(source: str, keep_shebang: bool = True) -> str:
    """
    Remove all comments and docstrings from Python source code.
    Returns a syntactically valid Python program.

    Notes:
      - Requires Python 3.9+ (uses ast.unparse).
      - Preserves a top-of-file shebang if keep_shebang=True.
      - Will change formatting/quoting/spacing to the AST's normalized form.
      - If your program reads __doc__ at runtime, that value will become None.
    """
    # Keep shebang line (if any)
    shebang = ""
    lines = source.splitlines()
    if keep_shebang and lines and lines[0].startswith("#!"):
        shebang = lines[0] + "\n"
        source_body = "\n".join(lines[1:])
    else:
        source_body = source

    # Parse, strip docstrings, and unparse (comments are not emitted by unparse)
    tree = ast.parse(source_body)
    tree = _DocstringStripper().visit(tree)
    ast.fix_missing_locations(tree)
    cleaned = ast.unparse(tree)

    return shebang + cleaned



def extract_method_code(file_path: str, method_name: str, class_name: Optional[str] = None) -> Optional[str]:
    """
    Return the source code of the first function/method named `method_name`.

    Behavior:
    - If class_name is None:
        1) Prefer a top-level def/async def named `method_name`.
        2) If not found, search recursively for a nested def/async def with that name anywhere in the module
           (including functions inside functions and methods).
    - If class_name is provided:
        1) Find that class.
        2) Prefer a direct method (def/async def) with `method_name` in the class body.
        3) If not found, search recursively within that class' subtree, so methods-with-inner-functions
           (or even nested classes) are handled.

    Returns the exact text slice from the file, or None if not found.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    lines = code.splitlines()

    def segment_for(node: ast.AST) -> Optional[str]:
        # Prefer end_lineno if available (Py3.8+); fall back to ast.get_source_segment.
        if hasattr(node, "lineno") and hasattr(node, "end_lineno") and node.lineno and node.end_lineno:
            start = node.lineno - 1
            end = node.end_lineno
            return "\n".join(lines[start:end])
        try:
            return ast.get_source_segment(code, node)
        except Exception:
            return None

    Fn = (ast.FunctionDef, ast.AsyncFunctionDef)

    # ---- Helpers
    def find_top_level_fn(module: ast.Module, name: str) -> Optional[Union[ast.FunctionDef, ast.AsyncFunctionDef]]:
        for node in module.body:
            if isinstance(node, Fn) and node.name == name:
                return node
        return None

    def find_direct_method(cls: ast.ClassDef, name: str) -> Optional[Union[ast.FunctionDef, ast.AsyncFunctionDef]]:
        for node in cls.body:
            if isinstance(node, Fn) and node.name == name:
                return node
        return None

    def find_class(module: ast.Module, name: str) -> Optional[ast.ClassDef]:
        for node in module.body:
            if isinstance(node, ast.ClassDef) and node.name == name:
                return node
        return None

    def recursive_find(nodes: Iterable[ast.AST], name: str) -> Optional[Union[ast.FunctionDef, ast.AsyncFunctionDef]]:
        """DFS search for a FunctionDef/AsyncFunctionDef named `name` within `nodes` (inclusive)."""
        stack = list(nodes)
        while stack:
            node = stack.pop()
            if isinstance(node, Fn) and node.name == name:
                return node
            # Descend into bodies/children that can contain defs
            for child in ast.iter_child_nodes(node):
                stack.append(child)
        return None

    # ---- Search logic
    if class_name is None:
        # 1) Prefer top-level match
        match = find_top_level_fn(tree, method_name)
        if match:
            return segment_for(match)
        # 2) Otherwise, search anywhere (nested functions, inside class methods, etc.)
        match = recursive_find([tree], method_name)
        return segment_for(match) if match else None

    # class_name provided
    cls = find_class(tree, class_name)
    if not cls:
        return None

    # 1) Prefer a direct method in the class body
    match = find_direct_method(cls, method_name)
    if match:
        return segment_for(match)

    # 2) Otherwise, search recursively in the class subtree (covers functions defined inside methods)
    match = recursive_find([cls], method_name)
    return segment_for(match) if match else None
