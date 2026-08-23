"""DE ONTKOPPELINGSTEST — audit sectie 24, vereist bewijs.

Phase 4, deliverable 8 / stap 9. Exit-criterium 2 heeft twee helften:

**Statisch.** `alpha/` importeert nergens uit `risk/`, en `risk/` kent nul
alpha-parameters.

**Functioneel.** Dit is de scherpste: voer twee VERSCHILLENDE alpha-units met
een identieke `a_t`-reeks door de engine en eis dat de output bit-identiek is.
Is dat niet zo, dan zit er alpha-kennis in de risicolaag - hoe die er ook
binnengekomen is.

WAAROM DEZE SUITE NIET OP DE BESTAANDE HOOK LEUNT
--------------------------------------------------
`alpha/base.py::assert_alpha_isolation` vuurt via `__init_subclass__`, dus
uitsluitend voor modules die een `AlphaUnit` subclassen. Dat zijn er twee.
De statische helft hieronder scant de VOLLEDIGE `alpha/`-boom, inclusief de
legacy-units die de hook nooit bereikt.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.risk import contract as risk_contract
from tradebot.risk.contract import MarketState, RiskState
from tradebot.risk.engine import RiskEngine
from tradebot.schemas.config import RiskConfig, load_config

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "tradebot"
CONF = ROOT / "conf"
TS = pd.Timestamp("2026-01-01T00:00:00Z")

#: Lagen waaruit `alpha/` nooit mag importeren (audit sectie 11.1).
FORBIDDEN_FOR_ALPHA = frozenset({"risk", "portfolio", "execution", "oms"})

#: Woorden die een alpha-grootheid aanduiden. Een risicomodule die er een
#: noemt, kent iets van de strategie die hem aanroept.
ALPHA_VOCABULARY = frozenset(
    {
        "alpha", "expected_alpha", "edge", "mu_hat", "signal", "signals",
        "confidence", "conviction", "meta_label", "metalabel", "score",
        "prediction", "forecast", "sharpe", "hitrate", "win_rate",
        "model_name", "unit_name", "strategy", "strategy_name",
    }
)


def _py_files(package: str) -> list[Path]:
    return sorted(p for p in (SRC / package).rglob("*.py") if "__pycache__" not in p.parts)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            # Relatieve imports: `from ..risk.limits import x` -> module="risk.limits".
            found.add(node.module or "")
    return found


def _head_layer(dotted: str) -> str:
    parts = [p for p in dotted.split(".") if p]
    if not parts:
        return ""
    return parts[1] if parts[0] == "tradebot" and len(parts) > 1 else parts[0]


# --------------------------------------------------------------------------- #
# Statische helft
# --------------------------------------------------------------------------- #
class TestAlphaImportsNothingFromRisk:
    """Scant de VOLLEDIGE alpha-boom, niet alleen de contract-units."""

    def test_the_scan_actually_covers_the_whole_package(self) -> None:
        files = _py_files("alpha")
        assert len(files) >= 25, f"slechts {len(files)} alpha-modules gescand"

    @pytest.mark.parametrize(
        "path", _py_files("alpha"), ids=lambda p: p.name
    )
    def test_no_module_imports_a_forbidden_layer(self, path: Path) -> None:
        offending = sorted(
            m for m in _imported_modules(path)
            if _head_layer(m) in FORBIDDEN_FOR_ALPHA
        )
        assert not offending, f"{path.name} importeert uit {offending}"

    def test_a_conditional_import_would_also_be_caught(self) -> None:
        """De scan is statisch: een import in een functie telt net zo goed mee."""
        source = "def f():\n    from ..risk.limits import apply_gross_cap\n"
        tree = ast.parse(source)
        modules = {
            n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
        }
        assert any(_head_layer(m) in FORBIDDEN_FOR_ALPHA for m in modules)


class TestRiskKnowsNoAlphaParameter:
    """Exit-criterium 2, tweede helft: nul alpha-parameters in het besluitpad."""

    #: De modules die het BESLUIT nemen. `risk/kelly.py`, `risk/var.py` en de
    #: overige analysemodules staan hier bewust niet in: zij zijn gereedschap
    #: dat L8 mag gebruiken, geen onderdeel van de soevereine laag.
    DECISION_PATH = (
        "contract.py", "vol_targeting.py", "limits.py",
        "kill_switches.py", "engine.py",
    )

    @pytest.mark.parametrize("name", DECISION_PATH)
    def test_no_public_signature_names_an_alpha_quantity(self, name: str) -> None:
        tree = ast.parse((SRC / "risk" / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args
            names = {
                a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]
            }
            leaked = names & ALPHA_VOCABULARY
            assert not leaked, f"{name}::{node.name} neemt {sorted(leaked)}"

    @pytest.mark.parametrize("name", DECISION_PATH)
    def test_the_decision_path_imports_no_alpha_bearing_module(self, name: str) -> None:
        forbidden = {"alpha", "kelly", "factor_alpha", "hmm_regime", "train", "tune"}
        for module in _imported_modules(SRC / "risk" / name):
            parts = {p for p in module.split(".") if p}
            assert not (parts & forbidden), f"{name} importeert uit {module}"

    def test_the_market_state_carries_no_alpha_field(self) -> None:
        import dataclasses

        fields = {f.name for f in dataclasses.fields(MarketState)}
        assert not (fields & ALPHA_VOCABULARY), sorted(fields & ALPHA_VOCABULARY)

    def test_the_risk_state_carries_no_alpha_field(self) -> None:
        import dataclasses

        fields = {f.name for f in dataclasses.fields(RiskState)}
        assert not (fields & ALPHA_VOCABULARY), sorted(fields & ALPHA_VOCABULARY)

    def test_the_risk_config_carries_no_alpha_parameter(self) -> None:
        leaked = set(RiskConfig.model_fields) & ALPHA_VOCABULARY
        assert not leaked, sorted(leaked)

    def test_the_engine_signature_takes_only_the_three_contract_arguments(self) -> None:
        params = list(inspect.signature(RiskEngine.decide).parameters)
        assert params == ["self", "desired_exposure", "market_state", "risk_state"]

    def test_factor_alpha_no_longer_lives_in_risk(self) -> None:
        """Bevinding E1 / audit sectie 24: alpha-logica hoort niet in L7."""
        assert not (SRC / "risk" / "factor_alpha.py").exists()
        assert (SRC / "alpha" / "factor_alpha.py").exists()

    def test_the_entangled_portfolio_module_no_longer_lives_in_risk(self) -> None:
        """Exit-criterium 3."""
        assert not (SRC / "risk" / "portfolio.py").exists()
        assert (SRC / "portfolio" / "legacy_sizing.py").exists()
        assert (SRC / "portfolio" / "covariance.py").exists()


# --------------------------------------------------------------------------- #
# Functionele helft — de scherpste
# --------------------------------------------------------------------------- #
class _MomentumLikeUnit:
    """Alpha-unit A: rang-gebaseerd, met een eigen parametrisatie en state."""

    name = "momentum_like"

    def __init__(self, lookback: int) -> None:
        self.lookback = lookback
        self.calls = 0

    def exposures(self, panel: pd.DataFrame) -> dict[str, float]:
        self.calls += 1
        window = panel.iloc[-self.lookback :]
        ranked = window.sum().rank(pct=True)
        centred = 2.0 * (ranked - ranked.mean()) / max(len(ranked) - 1, 1)
        return {str(k): float(np.clip(v, -1.0, 1.0)) for k, v in centred.items()}


class _CarryLikeUnit:
    """Alpha-unit B: een volstrekt ander mechanisme, andere state, andere naam."""

    name = "carry_like"

    def __init__(self, *, scale: float) -> None:
        self.scale = scale
        self.history: list[dict[str, float]] = []

    def exposures(self, panel: pd.DataFrame) -> dict[str, float]:
        out = {
            str(c): float(np.clip(np.tanh(self.scale * panel[c].iloc[-1]), -1.0, 1.0))
            for c in panel.columns
        }
        self.history.append(out)
        return out


class _ReplayUnit:
    """Alpha-unit C: kent geen markt, speelt alleen een vastgelegde reeks af."""

    name = "replay"

    def __init__(self, series: list[dict[str, float]]) -> None:
        self._series = series
        self._i = -1

    def exposures(self, _panel: pd.DataFrame) -> dict[str, float]:
        self._i += 1
        return dict(self._series[self._i])


@pytest.fixture
def engine() -> RiskEngine:
    return RiskEngine(load_config(CONF / "risk/default.yaml", RiskConfig))


@pytest.fixture
def symbols(engine: RiskEngine) -> list[str]:
    return list(engine.config.clusters)


@pytest.fixture
def panel(symbols: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        rng.normal(0.0, 0.05, size=(120, len(symbols))),
        columns=symbols,
        index=pd.date_range("2025-09-01", periods=120, freq="D", tz="UTC"),
    )


def _market(symbols: list[str], sigma: float) -> MarketState:
    return MarketState(
        asof_ts=TS,
        sigma_hat={s: sigma for s in symbols},
        adv_usd={s: 5e8 for s in symbols},
    )


def _state() -> RiskState:
    return RiskState(equity=1.0, high_water_mark=1.0, day_start_equity=1.0)


class TestIdenticalAlphaGivesBitIdenticalRisk:
    """*"Voer twee verschillende alpha-units met een identieke `a_t`-reeks door
    de engine en bewijs dat de output bit-identiek is."*"""

    @pytest.mark.parametrize("sigma", [0.05, 0.30, 0.72, 2.0])
    def test_two_unrelated_units_producing_the_same_a_t_are_indistinguishable(
        self, engine: RiskEngine, symbols: list[str], panel: pd.DataFrame, sigma: float
    ) -> None:
        unit_a = _MomentumLikeUnit(lookback=60)
        a_t = unit_a.exposures(panel)

        # Unit B is een ander mechanisme met andere state, maar levert per
        # constructie exact dezelfde a_t.
        unit_b = _ReplayUnit([dict(a_t)])
        b_t = unit_b.exposures(panel)
        assert a_t == b_t

        market, state = _market(symbols, sigma), _state()
        first = engine.decide(a_t, market, state)
        second = engine.decide(b_t, market, state)

        assert first.as_record() == second.as_record()
        for s in symbols:
            assert first.permitted_exposure[s] == second.permitted_exposure[s]

    def test_a_full_series_replays_bit_identically(
        self, engine: RiskEngine, symbols: list[str], panel: pd.DataFrame
    ) -> None:
        """Niet één bar maar een hele reeks, met wisselende marktstaat."""
        unit_a = _CarryLikeUnit(scale=8.0)
        series = [
            unit_a.exposures(panel.iloc[: i + 30]) for i in range(20)
        ]
        unit_b = _ReplayUnit([dict(x) for x in series])

        state = _state()
        for i, sigma in enumerate(np.linspace(0.05, 1.5, 20)):
            market = _market(symbols, float(sigma))
            from_a = engine.decide(series[i], market, state)
            from_b = engine.decide(unit_b.exposures(panel), market, state)
            assert from_a.as_record() == from_b.as_record()

    def test_the_engine_cannot_see_which_unit_produced_the_book(
        self, engine: RiskEngine, symbols: list[str], panel: pd.DataFrame
    ) -> None:
        """Er is geen kanaal waarlangs identiteit binnen zou kunnen komen.

        `decide()` neemt precies drie argumenten, en geen ervan draagt een naam,
        een model of een parametrisatie. Deze test maakt dat expliciet door twee
        units met verschillende `name`, verschillende state en verschillende
        parameters dezelfde `a_t` te laten leveren.
        """
        a_t = _MomentumLikeUnit(lookback=90).exposures(panel)
        loud = _CarryLikeUnit(scale=1.0)
        loud.name = "a_unit_with_a_very_different_identity"  # type: ignore[misc]
        replay = _ReplayUnit([dict(a_t)])
        replay.name = "another_identity_entirely"  # type: ignore[misc]

        market, state = _market(symbols, 0.72), _state()
        assert (
            engine.decide(a_t, market, state).as_record()
            == engine.decide(replay.exposures(panel), market, state).as_record()
        )

    def test_a_different_a_t_does_change_the_outcome(
        self, engine: RiskEngine, symbols: list[str], panel: pd.DataFrame
    ) -> None:
        """Controlemeting: de test hierboven is niet triviaal groen."""
        a_t = _MomentumLikeUnit(lookback=60).exposures(panel)
        other = {s: -v for s, v in a_t.items()}
        market, state = _market(symbols, 0.72), _state()
        assert (
            engine.decide(a_t, market, state).as_record()
            != engine.decide(other, market, state).as_record()
        )


class TestTheRiskLayerReactsOnlyToMeasuredQuantities:
    def test_the_same_a_t_under_different_vol_gives_different_exposure(
        self, engine: RiskEngine, symbols: list[str]
    ) -> None:
        """De laag reageert wel degelijk - maar op de MARKT, niet op de unit."""
        a_t = {s: 0.8 for s in symbols}
        calm = engine.decide(a_t, _market(symbols, 0.05), _state())
        wild = engine.decide(a_t, _market(symbols, 3.0), _state())
        assert wild.gross() < calm.gross()

    def test_the_contract_module_exposes_no_alpha_entry_point(self) -> None:
        public = {n for n in dir(risk_contract) if not n.startswith("_")}
        assert not (public & ALPHA_VOCABULARY), sorted(public & ALPHA_VOCABULARY)
