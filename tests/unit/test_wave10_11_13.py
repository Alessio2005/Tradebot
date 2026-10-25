"""Structural tests for Wave 12 (CI/CD) and Wave 13 (docs).

De Wave-10/11-tests stonden hier ook: ze toetsten alleen dat de Docker- en
K8s-stubbestanden BESTONDEN, niet dat ze konden draaien. Ze waren groen terwijl elk
image faalde bij `COPY apps/`. De infra wordt nu getoetst in
`tests/unit/test_infra_contract.py` (audit 2026-10-24, E-2).
"""
from __future__ import annotations

import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Wave 12 — CI/CD
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rel_path",
    [
        ".github/workflows/paper-trade-ci.yml",
        ".github/workflows/nightly-regression.yml",
    ],
)
def test_wave12_workflows_exist(rel_path: str):
    assert (_ROOT / rel_path).exists(), f"Missing Wave 12 file: {rel_path}"


def test_wave12_paper_trade_workflow_schedules_weekly():
    text = (_ROOT / ".github/workflows/paper-trade-ci.yml").read_text(encoding="utf-8")
    assert "cron:" in text
    assert "0 2 * * 1" in text   # weekly Monday 02:00 UTC


# ---------------------------------------------------------------------------
# Wave 13 — Documentation
# ---------------------------------------------------------------------------
def test_wave13_changelog_mentions_audit_fixes():
    text = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "AUDIT A-1" in text
    assert "AUDIT D-2" in text
    assert "AUDIT E-4" in text
