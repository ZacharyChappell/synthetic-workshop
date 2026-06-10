"""Launcher for the optional Streamlit GUI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def launch_gui(
    *,
    host: str = "127.0.0.1",
    port: int = 8501,
    browser: bool = True,
) -> int:
    """Launch the local Streamlit scene workbench."""

    app_path = Path(__file__).with_name("app.py")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        str(not browser).lower(),
    ]
    return subprocess.call(cmd)
