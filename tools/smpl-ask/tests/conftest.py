"""Make `smpl_ask` importable from the repo-root venv (the root pytest config does not collect tools/).

    uv run --group dev pytest tools/smpl-ask/tests -q

`smpl_ask` imports numpy / soundfile / scipy lazily where it can and makes no network call at import;
every test here stubs the model, so nothing is billed.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


import pytest


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Belt and braces (AF-06): a test that forgets to inject a transport must fail, not bill."""
    from smpl_ask import backends

    def refuse(*a, **k):
        raise AssertionError("a smpl-ask test tried to reach the network")

    monkeypatch.setattr(backends, "_urllib_transport", refuse)
