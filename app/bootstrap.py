"""Make `src` importable. Load secrets without touching st.secrets locally."""
from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _toml_paths() -> list[Path]:
    return [
        ROOT / ".streamlit" / "secrets.toml",
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    ]


def _apply_mapping(data: dict) -> None:
    for key, value in data.items():
        if isinstance(value, dict):
            _apply_mapping(value)
            continue
        if isinstance(value, (str, int, float)) and not os.getenv(str(key)):
            os.environ[str(key)] = str(value)


def _likely_streamlit_cloud() -> bool:
    return Path.home() == Path("/home/appuser") or Path("/mount/src").exists()


def setup() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    for path in _toml_paths():
        if path.is_file():
            _apply_mapping(tomllib.loads(path.read_text(encoding="utf-8")))
            return

    # Streamlit Cloud injects secrets into st.secrets (no local toml).
    # Never access st.secrets on a local machine — that prints "No secrets found".
    if not _likely_streamlit_cloud():
        return
    try:
        import streamlit as st

        _apply_mapping(dict(st.secrets))
    except Exception:
        pass
