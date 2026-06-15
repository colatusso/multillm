"""Example verifier: runs candidate Python code against a test.

SECURITY WARNING: this executes LLM-generated code on the local machine. Use only
with trusted context. In production, swap it for a container with limits (cpu/mem/net/
time) -- never run it in the main process. The council's verifier is just a
Callable[[str], bool]; this one is a ready-made example for code tasks.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

_CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


def extract_code(text: str) -> str:
    """Extracts the first ```python block; if there is none, returns the raw text."""
    m = _CODE_BLOCK.search(text)
    return (m.group(1) if m else text).strip()


def run_python(candidate_text: str, test_code: str, *, timeout: int = 10) -> bool:
    """True if `extracted_code + test_code` runs without error (exit 0)."""
    program = extract_code(candidate_text) + "\n\n" + test_code
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(program)
            path = f.name
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        if path and os.path.exists(path):
            os.unlink(path)
