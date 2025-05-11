import ast
from typing import Dict, Any

def generate_ast(code: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(code)
        return ast_to_dict(tree)
    except SyntaxError as e:
        return {"error": str(e)}

def ast_to_dict(node):
    if isinstance(node, ast.AST):
        fields = {}
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                fields[field] = [ast_to_dict(item) for item in value]
            else:
                fields[field] = ast_to_dict(value)
        return {node.__class__.__name__: fields}
    elif isinstance(node, list):
        return [ast_to_dict(item) for item in node]
    return node

def analyze_ast(node):
    if isinstance(node, ast.FunctionDef):
        func_name = node.name
        arg_names = [arg.arg for arg in node.args.args]
        num_args = len(arg_names)

        # Get declared return type (e.g., -> int)
        declared_return_type = None
        if node.returns:
            if isinstance(node.returns, ast.Name):
                declared_return_type = node.returns.id  # Extract type name
            elif isinstance(node.returns, ast.Subscript):  # Handles List[int], Dict[str, int]
                declared_return_type = ast.unparse(node.returns)

        # Extract actual return types from return statements
        return_count = 0
        return_types = set()

        for n in ast.walk(node):
            if isinstance(n, ast.Return) and n.value:
                return_count += 1
                return_types.add(type(n.value).__name__)

        actual_return_type_str = ', '.join(return_types) if return_types else "None"

        return (f"This function '{func_name}' takes {num_args} arguments {arg_names} "
                f"and returns {return_count} values of type(s) {actual_return_type_str}. "
                f"Declared return type is {declared_return_type if declared_return_type else 'None'}.")