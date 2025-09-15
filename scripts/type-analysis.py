import ast
import textwrap
import sys
from typing import Optional
import importlib
import inspect

def extract_class_method_returns(tree):
    class_methods = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    for stmt in ast.walk(item):
                        if isinstance(stmt, ast.Return) and stmt.value:
                            return_type = infer_type_from_node(
                                stmt.value,
                                symbol_table={},
                                class_methods=class_methods,
                                class_inits={class_name}
                            )
                            if return_type:
                                class_methods[f"{class_name}.{item.name}"] = return_type
    return class_methods

def resolve_external_type(obj):
    try:
        sig = inspect.signature(obj)
        return_annotation = sig.return_annotation
        if return_annotation is inspect.Signature.empty:
            return "Any"
        return str(return_annotation).replace("<class '", "").replace("'>", "")
    except Exception:
        return "Any"

def infer_type_from_node(node: ast.AST, symbol_table: dict = None, class_methods: dict = None, class_inits: dict = None) -> Optional[str]:
    if isinstance(node, ast.ListComp) or isinstance(node, ast.List):
        element_types = {infer_type_from_node(elt, symbol_table, class_methods, class_inits) for elt in node.elts}
        element_types.discard(None)
        inner = ", ".join(sorted(element_types)) if element_types else "Any"
        return f"List[{inner}]"
    elif isinstance(node, ast.Tuple):
        element_types = [infer_type_from_node(elt, symbol_table, class_methods, class_inits) or "Any" for elt in node.elts]
        return f"Tuple[{', '.join(element_types)}]"
    elif isinstance(node, ast.Dict):
        key_types = {infer_type_from_node(k, symbol_table, class_methods, class_inits) for k in node.keys if k is not None}
        val_types = {infer_type_from_node(v, symbol_table, class_methods, class_inits) for v in node.values if v is not None}
        key_types.discard(None)
        val_types.discard(None)
        key_str = ", ".join(sorted(key_types)) if key_types else "Any"
        val_str = ", ".join(sorted(val_types)) if val_types else "Any"
        return f"Dict[{key_str}, {val_str}]"
    elif isinstance(node, ast.Set):
        element_types = {infer_type_from_node(elt, symbol_table, class_methods, class_inits) for elt in node.elts}
        element_types.discard(None)
        inner = ", ".join(sorted(element_types)) if element_types else "Any"
        return f"Set[{inner}]"
    elif isinstance(node, ast.Constant):
        return type(node.value).__name__
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and class_methods and node.func.id in class_inits:
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            value = node.func.value
            if isinstance(value, ast.Name):
                instance = value.id
                if symbol_table:
                    class_name = symbol_table.get(instance)
                    if class_name and f"{class_name}.{attr}" in class_methods:
                        return class_methods[f"{class_name}.{attr}"]
            return "Any"
    elif isinstance(node, ast.BinOp):
        return infer_type_from_node(node.left, symbol_table, class_methods, class_inits)
    elif isinstance(node, ast.Compare):
        return "bool"
    elif isinstance(node, ast.Name):
        if symbol_table and node.id in symbol_table:
            return symbol_table[node.id]
        return None
    return None

def process_function(node, source_lines, class_methods, class_inits, target_method):
    if(node.name != target_method):
        return

    symbol_table = {}
    assignments = {}

    # Add arguments to symbol table
    for arg in node.args.args:
        symbol_table[arg.arg] = "Any"
    if node.args.vararg:
        symbol_table[node.args.vararg.arg] = "Tuple[Any, ...]"
    if node.args.kwarg:
        symbol_table[node.args.kwarg.arg] = "Dict[str, Any]"

    for inner in ast.walk(node):
        if isinstance(inner, ast.Assign):
            for target in inner.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = inner.value
                    inferred = infer_type_from_node(inner.value, symbol_table, class_methods, class_inits)
                    if inferred:
                        symbol_table[target.id] = inferred
        elif isinstance(inner, ast.AugAssign):
            if isinstance(inner.target, ast.Name):
                assignments[inner.target.id] = inner.value
                inferred = infer_type_from_node(inner.value, symbol_table, class_methods, class_inits)
                if inferred:
                    symbol_table[inner.target.id] = inferred
        elif isinstance(inner, ast.If):
            test_type = infer_type_from_node(inner.test, symbol_table, class_methods, class_inits)
            for body_node in inner.body + inner.orelse:
                if isinstance(body_node, ast.Assign):
                    for target in body_node.targets:
                        if isinstance(target, ast.Name):
                            inferred = infer_type_from_node(body_node.value, symbol_table, class_methods, class_inits)
                            if inferred:
                                symbol_table[target.id] = inferred

    def resolve_type(name):
        if name in symbol_table:
            return symbol_table[name]
        if name in assignments:
            val = assignments[name]
            inferred = infer_type_from_node(val, symbol_table, class_methods, class_inits)
            if inferred:
                symbol_table[name] = inferred
                return inferred
        return "Any"

    has_return = False
    for stmt in ast.walk(node):
        if isinstance(stmt, ast.Return):
            has_return = True
            lineno = stmt.lineno
            code_line = source_lines[lineno - 1].strip()
            if stmt.value is None:
                inferred_type = "None"
            elif isinstance(stmt.value, ast.Tuple):
                inferred_items = []
                for elt in stmt.value.elts:
                    if isinstance(elt, ast.Name):
                        inferred_items.append(resolve_type(elt.id))
                    else:
                        inferred_items.append(infer_type_from_node(elt, symbol_table, class_methods, class_inits) or "Any")
                inferred_type = f"Tuple[{', '.join(inferred_items)}]"
            elif isinstance(stmt.value, ast.Name):
                inferred_type = resolve_type(stmt.value.id)
            else:
                inferred_type = infer_type_from_node(stmt.value, symbol_table, class_methods, class_inits)
            print(f"  Return at line {lineno}: {code_line}")
            print(f"    Inferred type: {inferred_type}")

    print("  Argument types:")
    for arg in node.args.args:
        print(f"    {arg.arg}: {symbol_table.get(arg.arg, 'Any')}")
    if node.args.vararg:
        print(f"    *{node.args.vararg.arg}: {symbol_table.get(node.args.vararg.arg)}")
    if node.args.kwarg:
        print(f"    **{node.args.kwarg.arg}: {symbol_table.get(node.args.kwarg.arg)}")

    if not has_return:
        print("  No return statement. Inferred return type: None")
    print()

def get_function_types(code: str, source_lines: list, target_method: str) -> None:
    code = textwrap.dedent(code)
    tree = ast.parse(code)
    class_methods = extract_class_method_returns(tree)
    class_inits = {name.split(".")[0] for name in class_methods}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            process_function(node, source_lines, class_methods, class_inits, target_method)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    process_function(item, source_lines, class_methods, class_inits, target_method)


def get_method_name(s: str) -> str:
    parts = s.split("/")
    if len(parts) < 2:
        raise ValueError("The string does not have enough parts separated by '/'")
    second_to_last = parts[-2]
    subparts = second_to_last.split(".")
    return subparts[-1]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_python_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    target_method = get_method_name(file_path)
    print("method name: ",target_method)
    try:
        with open(file_path, "r") as f:
            code = f.read()
            source_lines = code.splitlines()
            get_function_types(code, source_lines, target_method)

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except IndentationError as e:
        print(f"Indentation Error: {e}")
    except SyntaxError as e:
        print(f"Syntax Error: {e}")

