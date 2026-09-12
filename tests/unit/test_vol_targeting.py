"""L7 volatility targeting — de formule, en vooral: wat hij weigert te doen.

Phase 4, stap 3. Exit-criterium 4: standalone, en faalt hard bij een
ontbrekende of niet-eindige `sigma_hat`. Nul fallbacks naar constante vol.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from tradebot.risk.contract import BOOK_SCOPE, ConstraintKind
from tradebot.risk.vol_targeting import (
    apply_volatility_target,
    book_sigma_hat,
    require_sigma_hat,
    volatility_scalar,
)
from tradebot.schemas.config import RiskConfig, load_config
from tradebot.utils.failfast import ConfigContractError, DataContractError

CONF = Path(__file__).resolve().parents[2] / "conf"

TARGET = 0.08
MAXLEV = 1.5


class TestTheBindingFormula:
    """`w_t = min(max_leverage, sigma_target / sigma_hat)` — audit sectie 14.1."""

    def test_scales_down_when_vol_exceeds_target(self) -> None:
        w = volatility_scalar(sigma_hat=0.72, sigma_target=0.12, max_leverage=MAXLEV)
        assert w == pytest.approx(0.12 / 0.72)

    def test_max_leverage_caps_the_scalar_in_a_calm_market(self) -> None:
        w = volatility_scalar(sigma_hat=0.01, sigma_target=TARGET, max_leverage=MAXLEV)
        assert w == pytest.approx(MAXLEV)

    def test_scalar_is_exactly_one_when_vol_equals_target(self) -> None:
        assert volatility_scalar(
            sigma_hat=TARGET, sigma_target=TARGET, max_leverage=MAXLEV
        ) == pytest.approx(1.0)

    @pytest.mark.parametrize("bad", [0.0, -0.5])
    def test_non_positive_config_crashes(self, bad: float) -> None:
        with pytest.raises(ConfigContractError):
            volatility_scalar(sigma_hat=0.5, sigma_target=bad, max_leverage=MAXLEV)
        with pytest.raises(ConfigContractError):
            volatility_scalar(sigma_hat=0.5, sigma_target=TARGET, max_leverage=bad)


class TestFailFastOnSigmaHat:
    """Exit-criterium 4. Elk van deze gevallen kreeg vóór Phase 4 een vervangwaarde."""

    @pytest.mark.parametrize(
        "bad", [float("nan"), float("inf"), -float("inf"), None, 0.0, -0.3],
        ids=["nan", "inf", "-inf", "missing", "zero", "negative"],
    )
    def test_invalid_sigma_hat_crashes(self, bad: object) -> None:
        with pytest.raises(DataContractError):
            require_sigma_hat(bad, symbol="ETHUSDT")

    def test_valid_sigma_hat_passes_through(self) -> None:
        assert require_sigma_hat(0.42, symbol="ETHUSDT") == pytest.approx(0.42)

    def test_missing_symbol_in_the_panel_crashes_the_book(self) -> None:
        """Geen stille uitsluiting van één asset: het hele besluit valt om."""
        with pytest.raises(DataContractError):
            book_sigma_hat({"A": 0.5, "B": 0.5}, {"A": 0.4})

    def test_flat_alpha_does_not_excuse_a_missing_sigma_hat(self) -> None:
        """De geldigheid van L7 mag niet afhangen van de output van L4."""
        with pytest.raises(DataContractError):
            book_sigma_hat({"A": 0.0}, {"A": float("nan")})

    def test_there_is_no_constant_vol_fallback_anywhere_in_the_module(self) -> None:
        """Statisch: nul `except`, nul `fillna`, nul `ffill` in deze module."""
        src = (
            Path(__file__).resolve().parents[2]
            / "src" / "tradebot" / "risk" / "vol_targeting.py"
        ).read_text(encoding="utf-8")
        code = "\n".join(
            line for line in src.splitlines()
            if not line.lstrip().startswith("#")
        )
        for forbidden in ("except", "fillna", "ffill", "bfill", "nan_to_num"):
            assert forbidden not in code, f"fallback-idioom {forbidden!r} in vol_targeting.py"


class TestBookVolatilityIsTheComonotoneUpperBound:
    def test_book_sigma_is_the_weighted_sum_of_asset_sigmas(self) -> None:
        assert book_sigma_hat(
            {"A": 0.5, "B": -0.5}, {"A": 0.60, "B": 0.80}
        ) == pytest.approx(0.5 * 0.60 + 0.5 * 0.80)

    def test_shorts_add_risk_rather_than_netting_it_out(self) -> None:
        """rho=1 is conservatief: een short hedget hier niet."""
        long_only = book_sigma_hat({"A": 1.0}, {"A": 0.5, "B": 0.5})
        long_short = book_sigma_hat({"A": 1.0, "B": -1.0}, {"A": 0.5, "B": 0.5})
        assert long_short > long_only

    def test_it_dominates_any_real_correlation_structure(self) -> None:
        """sigma_book(rho=1) >= sqrt(w' Sigma w) voor elke geldige rho."""
        import numpy as np

        w = np.array([0.4, -0.3, 0.6])
        sig = np.array([0.5, 0.9, 0.2])
        bound = book_sigma_hat(
            {"A": w[0], "B": w[1], "C": w[2]},
            {"A": sig[0], "B": sig[1], "C": sig[2]},
        )
        rng = np.random.default_rng(0)
        for _ in range(200):
            a = rng.standard_normal((3, 8))
            corr = np.corrcoef(a)
            cov = np.outer(sig, sig) * corr
            true_vol = math.sqrt(float(w @ cov @ w))
            assert true_vol <= bound + 1e-9


class TestApplyToTheBook:
    SIG = {"A": 0.72, "B": 0.72}

    def test_the_whole_book_is_scaled_by_one_scalar(self) -> None:
        permitted, binding = apply_volatility_target(
            {"A": 1.0, "B": -0.5}, self.SIG,
            sigma_target=TARGET, max_leverage=MAXLEV,
        )
        assert binding is not None
        w = TARGET / (1.5 * 0.72)
        assert permitted["A"] == pytest.approx(w)
        assert permitted["B"] == pytest.approx(-0.5 * w)

    def test_it_only_ever_shrinks(self) -> None:
        """Sectie 5.3: L7 voegt nooit exposure toe, ook niet in een rustige markt."""
        permitted, binding = apply_volatility_target(
            {"A": 0.5}, {"A": 0.001},
            sigma_target=TARGET, max_leverage=MAXLEV,
        )
        assert permitted["A"] == pytest.approx(0.5)
        assert binding is None

    def test_a_binding_target_is_registered_with_its_config_key(self) -> None:
        _, binding = apply_volatility_target(
            {"A": 1.0}, {"A": 0.72}, sigma_target=TARGET, max_leverage=MAXLEV,
        )
        assert binding is not None
        assert binding.kind is ConstraintKind.VOL_TARGET
        assert binding.scope == BOOK_SCOPE
        assert binding.measured == pytest.approx(0.72)
        assert binding.threshold == pytest.approx(TARGET)
        assert binding.config_key == "risk.sigma_target"

    def test_nothing_is_registered_when_the_target_does_not_bind(self) -> None:
        _, binding = apply_volatility_target(
            {"A": 1.0}, {"A": 0.01}, sigma_target=TARGET, max_leverage=MAXLEV,
        )
        assert binding is None

    def test_a_flat_book_stays_flat_without_binding(self) -> None:
        permitted, binding = apply_volatility_target(
            {"A": 0.0, "B": 0.0}, self.SIG,
            sigma_target=TARGET, max_leverage=MAXLEV,
        )
        assert permitted == {"A": 0.0, "B": 0.0}
        assert binding is None

    def test_it_is_pure_and_repeatable(self) -> None:
        """Exit-criterium 2 leunt hierop: identieke input, bit-identieke output."""
        args = ({"A": 1.0, "B": -0.5}, self.SIG)
        kwargs = {"sigma_target": TARGET, "max_leverage": MAXLEV}
        first, _ = apply_volatility_target(*args, **kwargs)
        second, _ = apply_volatility_target(*args, **kwargs)
        assert first == second

    def test_the_source_mapping_is_not_mutated(self) -> None:
        desired = {"A": 1.0}
        apply_volatility_target(
            desired, {"A": 0.72}, sigma_target=TARGET, max_leverage=MAXLEV
        )
        assert desired == {"A": 1.0}


class TestAgainstTheShippedConfig:
    BASELINE_ANNUALISED_VOL = 0.72  # reports/BASELINE_BENCHMARK.md, 1/N ongehefboomd

    def test_the_phase3_baseline_vol_is_de_grossed(self) -> None:
        """72% geannualiseerde vol tegen `sigma_target` uit conf/risk/.

        De drempel is AFGELEID uit de config en niet hardgecodeerd. Zij stond op
        `0.08 / 0.72` met een absolute bovengrens van 0.2; toen de
        mandaatwijziging (`docs/RISK_MANDATE.md`) `sigma_target` op 0.20 zette,
        toetste die literal een mandaat dat niet meer bestond.

        Wat de test WEL vasthoudt: de vol-target moet de baseline nog steeds
        de-grossen. Zou hij dat niet doen, dan is de kalibratie zo ruim dat de
        vol-targeting op dit universum niets meer bindt — precies het defect dat
        RISK_MANDATE §2.3 openzet.
        """
        cfg = load_config(CONF / "risk/default.yaml", RiskConfig)
        w = volatility_scalar(
            sigma_hat=self.BASELINE_ANNUALISED_VOL,
            sigma_target=cfg.sigma_target,
            max_leverage=cfg.max_leverage,
        )
        assert w == pytest.approx(
            cfg.sigma_target / self.BASELINE_ANNUALISED_VOL, rel=1e-9)
        assert w < 1.0, (
            "de baseline-vol wordt niet ge-de-grossed; sigma_target ligt op of "
            f"boven de gemeten baselinevol van {self.BASELINE_ANNUALISED_VOL} "
            "en de vol-targeting bindt dan niet meer")
