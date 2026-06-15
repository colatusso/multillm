"""Loads variables from a .env into the environment (no external dependency).

Precedence: what's already in the environment (export) WINS over the .env (uses
setdefault). Handles `export KEY=val`, quotes and comments. KISS -- no
multiline/interpolation.
"""
from __future__ import annotations

import os
from pathlib import Path

# Default: .env at the project root (next to agents.yaml / pyproject.toml).
_DEFAULT = Path(__file__).resolve().parent.parent / ".env"


def load_env(path: str | os.PathLike | None = None) -> bool:
    """Reads a .env and populates os.environ (without overwriting already-defined vars).

    Returns True if the file existed and was read, False if it doesn't exist.
    """
    p = Path(path) if path else _DEFAULT
    if not p.exists():
        return False
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key:
            os.environ.setdefault(key, val)
    return True
