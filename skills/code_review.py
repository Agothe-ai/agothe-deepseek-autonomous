# code_review.py — Jarvis skill: instant code review using DeepSeek R1
# Loaded via: TOOL: load_skill(code_review)
# Usage: set CODE_REVIEW_PATH env var or review clipboard

import subprocess
import os
from pathlib import Path

def review_code(path: str = None):
    if path and Path(path).exists():
        code = Path(path).read_text()
        print(f"Reviewing {path} ({len(code)} chars)...")
    else:
        print("No path specified. Usage: set CODE_REVIEW_PATH=yourfile.py then load skill.")
        return
    
    print(f"Code loaded: {len(code.splitlines())} lines")
    print("Send this to DeepSeek R1 via paul_core for deep review.")
    print("TOOL: run_python(import ast; ast.parse(open('" + (path or '') + "').read()); print('Syntax OK'))")

path = os.environ.get("CODE_REVIEW_PATH")
review_code(path)
