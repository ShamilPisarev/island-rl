from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.config import Config, load_config  # noqa: E402


@pytest.fixture
def cfg() -> Config:
    """The real default config -- tests run against what ships, not a stub."""
    return load_config()
