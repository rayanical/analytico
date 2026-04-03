"""Sandboxed Python execution helpers for LLM-generated analysis."""

import ast
import contextlib
import io
import traceback
from typing import Any

import numpy as np
import pandas as pd


def secure_exec(code: str, df: pd.DataFrame) -> Any:
    """Execute generated Python code securely using AST whitelisting."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax Error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Security: Imports are not allowed.")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ["open", "eval", "exec", "compile"]:
                raise ValueError(f"Security: Function '{node.func.id}' is banned.")
            if isinstance(node.func, ast.Attribute) and node.func.attr == "__builtins__":
                raise ValueError("Security: Access to __builtins__ is banned.")

    local_scope = {"df": df.copy(), "pd": pd, "np": np, "result": None}

    capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(capture):
            exec(code, {"__builtins__": {}}, local_scope)
    except Exception as e:
        return f"Runtime Error: {e}\n{traceback.format_exc()}"

    output = capture.getvalue().strip()
    if local_scope.get("result") is not None:
        return local_scope["result"]
    return output or "No output."
