# tests/unit/test_factor_alpha.py
"""G4 factor lab guards (Wave 20, step 0.3).

Synthetic known-outcome tests (deterministic, R-5) + an audit-§9
reproduction test that runs only when the cached broad-perp panel exists.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.alpha import G4_FACTORSETS, factor_residual_alpha

ROOT = Path(__file__).resolve().parents[2]
N = 1500
TRUE_ALPHA_DAILY = 0.0008  # ~29%/yr at 365 periods
BETA_MKT, BETA_TSMOM = 0.30, 0.15


def _synthetic(alpha: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=N, freq="D", tz="UTC")
    mkt = pd.Series(rng.normal(0.0005, 0.03, N), idx, name="MKT")
    tsm = pd.Series(rng.normal(0.0002, 0.01, N), idx, name="TSMOM")
    noise = rng.normal(0.0, 0.004, N)
    y = alpha + BETA_MKT * mkt + BETA_TSMOM * tsm + noise
    return pd.Series(y, idx, name="ret"), pd.concat([mkt, tsm], axis=1)


def test_recovers_known_alpha_and_loadings() -> None:
    y, facs = _synthetic(TRUE_ALPHA_DAILY)
    res = factor_residual_alpha(y, facs, unit="synth", market="crypto")
    assert res.alpha_daily == pytest.approx(TRUE_ALPHA_DAILY, abs=3e-4)
    assert res.loadings["MKT"] == pytest.approx(BETA_MKT, abs=0.02)
    assert res.loadings["TSMOM"] == pytest.approx(BETA_TSMOM, abs=0.05)
    assert res.passes
    assert "[PASS]" in res.gate_row()


def test_zero_alpha_does_not_pass() -> None:
    y, facs = _synthetic(0.0)
    res = factor_residual_alpha(y, facs, unit="null", market="crypto")
    assert abs(res.t_alpha) < 2.0
    assert not res.passes
    assert "[FAIL]" in res.gate_row()


def test_determinism_bit_identical() -> None:
    y, facs = _synthetic(TRUE_ALPHA_DAILY)
    a = factor_residual_alpha(y, facs, unit="s", market="crypto")
    b = factor_residual_alpha(y, facs, unit="s", market="crypto")
    assert a == b  # frozen dataclass, exact float equality (R-5)


def test_incomplete_factorset_rejected() -> None:
    y, facs = _synthetic(TRUE_ALPHA_DAILY)
    with pytest.raises(ValueError, match="incomplete"):
        factor_residual_alpha(
            y, facs[["MKT"]], unit="s", market="crypto"
        )


def test_too_few_observations_rejected() -> None:
    y, facs = _synthetic(TRUE_ALPHA_DAILY)
    with pytest.raises(ValueError, match="too few"):
        factor_residual_alpha(y.iloc[:50], facs.iloc[:50], "s", "crypto")


def test_factorsets_match_mandate_g4() -> None:
    assert G4_FACTORSETS["crypto"] == ("MKT", "TSMOM")
    assert set(G4_FACTORSETS["equities"]) == {
        "Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "BAB",
    }
    assert set(G4_FACTORSETS["fx"]) == {"DOLLAR", "CARRY", "TREND"}
    assert set(G4_FACTORSETS["commodities"]) == {"MKT", "CARRY", "MOM"}


@pytest.mark.skipif(
    not (ROOT / "artefacts/broad_perp_daily_close.parquet").exists(),
    reason="cached broad-perp panel not present",
)
def test_audit_section9_reproduction() -> None:
    """Audit §9 book: alpha ≈ +20%/yr, t ≈ 2.65, p ≈ 0.008 vs MKT+TSMOM.

    Loose tolerances: the audit number came from the full sleeve pipeline;
    this guards the regression machinery, not the sleeves.
    """
    import importlib.util
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location(
        "tag", ROOT / "scripts/true_alpha_gates.py"
    )
    tag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tag)

    vb = tag.vb
    c, s, l, _ = vb.sleeves(5.0)
    combo = vb.causal_rp(c, s, l).dropna()
    facs = pd.concat([tag.BTC.rename("MKT"), tag.f_tsmom()], axis=1)
    res = factor_residual_alpha(combo, facs, unit="mn_book", market="crypto")
    assert res.alpha_ann == pytest.approx(0.20, abs=0.08)
    assert res.t_alpha == pytest.approx(2.65, abs=0.6)
    assert res.passes
