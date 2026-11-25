"""Config loading: YAML + ${ENV_VAR} expansion + optional .env file."""
from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=VALUE lines; existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _expand(value):
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(path: str | Path) -> dict:
    path = Path(path)
    load_dotenv(path.resolve().parent / ".env")
    load_dotenv(Path.cwd() / ".env")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return _expand(cfg)
