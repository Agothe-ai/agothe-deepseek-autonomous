# skill: code_review
# Reviews a Python file for common issues
import ast
from pathlib import Path

file_path = context.get("file_path", "")
lines_out = []

if not file_path:
    result = "[code_review] No file_path in context. Pass context={'file_path': 'yourfile.py'}"
else:
    p = Path(file_path)
    if not p.exists():
        result = f"[code_review] File not found: {file_path}"
    else:
        code = p.read_text(encoding="utf-8", errors="replace")
        file_lines = code.split("\n")
        lines_out.append(f"=== CODE REVIEW: {p.name} ===")
        lines_out.append(f"Lines: {len(file_lines)} | Size: {p.stat().st_size} bytes")

        # Syntax check
        try:
            tree = ast.parse(code)
            lines_out.append("[OK] Syntax: valid")

            # Check for missing module docstring
            if not (tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant)):
                lines_out.append("[WARN] No module docstring")

            # Find functions without docstrings
            no_doc = [n.name for n in ast.walk(tree)
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                      and not (n.body and isinstance(n.body[0], ast.Expr)
                               and isinstance(n.body[0].value, ast.Constant))]
            if no_doc:
                lines_out.append(f"[WARN] Functions without docstrings: {no_doc[:5]}")

            # Find long functions (>50 lines)
            long_fns = [n.name for n in ast.walk(tree)
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and hasattr(n, 'end_lineno')
                        and (n.end_lineno - n.lineno) > 50]
            if long_fns:
                lines_out.append(f"[WARN] Long functions (>50 lines): {long_fns}")

            # Count imports
            imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
            lines_out.append(f"[INFO] Imports: {len(imports)}")

            # Count classes and functions
            classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            if classes:
                lines_out.append(f"[INFO] Classes: {classes}")
            lines_out.append(f"[INFO] Functions: {len(functions)} defined")

        except SyntaxError as e:
            lines_out.append(f"[ERROR] Syntax error at line {e.lineno}: {e.msg}")

        result = "\n".join(lines_out)
