"""tests/conftest.py — pytest fixtures + path setup.

Ensures src/ and repo root are importable from any test location.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC       = _REPO_ROOT / "src"

for p in (_SRC, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture(autouse=True)
def _isolate_cb_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Wave 15 P0-5.5: redirect the circuit-breaker trip log to a tmp dir.

    Prevents production ``artefacts/circuit_breaker.log`` from affecting
    unit/integration tests that instantiate CircuitBreaker without an explicit
    ``cb_log_path`` in their config.
    """
    import tradebot.live.circuit_breaker as _cb_module

    monkeypatch.setattr(
        _cb_module, "_CB_LOG_PATH", tmp_path / "circuit_breaker.log"
    )
