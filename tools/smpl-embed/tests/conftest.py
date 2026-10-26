"""Make `smpl_embed` importable from the repo-root venv.

The root pytest config's `testpaths` does NOT collect `tools/`, so this suite is run
explicitly:

    uv run pytest tools/smpl-embed/tests -q

`smpl_embed.backends` imports stdlib only at module top (the heavy encoders are
lazy-imported inside `embed()`), so putting its src dir on `sys.path` is safe in the
light root venv — nothing here pulls torch.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
