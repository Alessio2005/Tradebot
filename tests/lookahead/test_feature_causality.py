"""Feature-causaliteit — burn-in, DI-2 en de multi-granulaire join.

Wat dit bestand NIET meer bevat: de truncatie-invariantie zelf. Die is in Phase
7/8 Stage B-2 verplaatst naar `test_truncation_invariance.py`, waar zij als
benoemde D-1-poort 1 staat, samen met haar negatieve controle. Verplaatst, niet
gekopieerd — twee bestanden die allebei beweren te definiëren wat causaal is,
groeien uit elkaar, en dan bewaakt de strengste niets meer dan de soepelste
toelaat.

Wat hier blijft, meet iets anders dan invariantie onder truncatie:

* **burn-in** — een feature mag zijn opstartfase niet opvullen, en zijn
  gedeclareerde `burn_in_period` moet exact kloppen. Een te lage declaratie
  laat een waarde door die op minder historie steunt dan beloofd.
* **DI-2** — het `fillna(reeks.std())`-idioom, naast zijn causale vervanger, met
  een statische AST-controle op `risk/hmm_regime.py`.
* **de join over granulariteiten** — 8h funding en 1d open interest op een 1d
  bar-raster, backward-only.

De gedeelde machinerie (gecertificeerd frame, snijpunten, `assert_bit_identical`)
staat in `d1_harness.py` en wordt hier geïmporteerd.

Ref: `fase_2_research_falsification.md` exit criteria 1 en 4; audit §17.1.
"""
from __future__ import annotations

import ast
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from tradebot.features.base import BaseFeature, DataRegister, InputSpec
from tradebot.features.positioning import build_certified_micro_frame
from tradebot.features.registry import build_default_registry
from tradebot.features.volatility import causal_expanding_std, log_returns
from tradebot.utils.failfast import CausalityViolationError

from .d1_harness import (
    FEAT_CFG,
    FUNDING_GRANULARITY,
    GRANULARITY,
    ROOT,
    STORE,
    SYMBOLS,
    assert_bit_identical,
    certified_frame,
    cut_points,
    requires_store,
)

pytestmark = pytest.mark.lookahead

REGISTRY = build_default_registry(FEAT_CFG)
FEATURE_IDS = [f.feature_id for f in REGISTRY.features]


@requires_store
class TestBurnInIsNeverFilled:
    """Geen enkele feature vult zijn opstartfase - dat is de kern van DI-2."""

    @pytest.mark.parametrize("symbol", SYMBOLS)
    @pytest.mark.parametrize(
        "feature", REGISTRY.features, ids=[f.feature_id for f in REGISTRY.features]
    )
    def test_burn_in_rows_are_nan(self, feature: BaseFeature, symbol: str) -> None:
        values = feature.transform(certified_frame(symbol)).values
        head = values.iloc[: feature.burn_in_period].to_numpy(dtype="float64")
        assert np.isnan(head).all(), (
            f"{symbol}/{feature.feature_id}: {int(np.isfinite(head).sum())} eindige "
            f"waarde(n) binnen de burn-in van {feature.burn_in_period} bars. Een "
            f"opgevulde burn-in is per definitie niet-causaal."
        )

    @pytest.mark.parametrize("symbol", SYMBOLS)
    @pytest.mark.parametrize(
        "feature", REGISTRY.features, ids=[f.feature_id for f in REGISTRY.features]
    )
    def test_first_value_lands_exactly_on_the_boundary(
        self, feature: BaseFeature, symbol: str
    ) -> None:
        """De burn-in is een BEWERING, geen ruwe schatting.

        Is hij te hoog opgegeven, dan wordt bruikbare informatie weggegooid; is
        hij te laag, dan glipt er een waarde door die op minder historie steunt
        dan gedeclareerd. Features waarvan de INPUT zelf later begint (open
        interest start op deze dataset maanden na OHLCV) zijn uitgezonderd.
        """
        source = certified_frame(symbol)
        input_cols = [
            c for c in type(feature).input_spec.columns if c in source.frame.columns
        ]
        if bool(source.frame[input_cols].isna().to_numpy().any()):
            pytest.skip("input begint later dan het bar-raster; grens niet scherp")
        values = feature.transform(source).values
        first_valid = int(values.notna().to_numpy().argmax(axis=0).min())
        assert first_valid == feature.burn_in_period, (
            f"{symbol}/{feature.feature_id}: eerste waarde op positie "
            f"{first_valid}, gedeclareerde burn_in_period is "
            f"{feature.burn_in_period}."
        )



@requires_store
class TestBurnInGuardCanGoRed:
    """De burn-in-guard in BaseFeature moet een opgevulde opstartfase weigeren."""

    def test_filled_burn_in_raises_causality_violation(self) -> None:
        class _FilledBurnIn(BaseFeature):
            name: ClassVar[str] = "deliberately_filled_burn_in"
            input_spec: ClassVar[InputSpec] = InputSpec(
                datasets=("ohlcv",), columns=("close",)
            )

            def __init__(self) -> None:
                super().__init__(params={})

            @property
            def burn_in_period(self) -> int:
                return int(FEAT_CFG.min_expanding_periods)

            @property
            def output_columns(self) -> tuple[str, ...]:
                return ("filled_vol",)

            def _compute(self, frame: pd.DataFrame) -> pd.DataFrame:
                r = log_returns(frame["close"])
                window = int(FEAT_CFG.min_expanding_periods)
                # Exact het DI-2-idioom.
                vol = r.rolling(window, min_periods=2).std().fillna(r.std())
                return pd.DataFrame(
                    {"filled_vol": vol.to_numpy(dtype="float64")}, index=frame.index
                )

        with pytest.raises(CausalityViolationError, match="burn-in"):
            _FilledBurnIn().transform(certified_frame("BTCUSDT"))



# --------------------------------------------------------------------------- #
# DI-2 in het bijzonder
# --------------------------------------------------------------------------- #
@requires_store
class TestDI2IsClosed:
    """Het concrete DI-2-idioom versus zijn causale vervanger, zij aan zij."""

    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_sample_wide_fillna_is_not_truncation_invariant(self, symbol: str) -> None:
        """`fillna(returns.std())` verandert wel degelijk door toekomstige data."""
        source = certified_frame(symbol)
        window = int(FEAT_CFG.min_expanding_periods)

        def di2(frame: pd.DataFrame) -> pd.Series:
            r = log_returns(frame["close"])
            return r.rolling(window, min_periods=2).std().fillna(r.std())

        full = di2(source.frame)
        cut = cut_points(source)[0]
        truncated = di2(source.truncate(cut).frame)
        expected = full.loc[full.index <= truncated.index[-1]]
        diff = np.abs(
            truncated.to_numpy(dtype="float64") - expected.to_numpy(dtype="float64")
        )
        n_changed = int(np.nansum(diff != 0.0))
        assert n_changed > 0, (
            f"{symbol}: het DI-2-idioom bleek hier toevallig invariant. Dat maakt "
            f"hem niet causaal - het maakt deze toets ongevoelig."
        )

    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_causal_expanding_std_is_truncation_invariant(self, symbol: str) -> None:
        """De vervanger uit `features/volatility.py` is het wel, bit-exact."""
        source = certified_frame(symbol)
        min_periods = int(FEAT_CFG.min_expanding_periods)

        def causal(frame: pd.DataFrame) -> pd.Series:
            return causal_expanding_std(
                log_returns(frame["close"]), min_periods=min_periods
            )

        full = causal(source.frame)
        for cut in cut_points(source):
            truncated = causal(source.truncate(cut).frame)
            assert_bit_identical(
                truncated.to_frame("expanding_std"),
                full.to_frame("expanding_std"),
                what=f"{symbol}/causal_expanding_std",
                cut=cut,
            )

    def test_hmm_regime_no_longer_uses_sample_wide_fill(self) -> None:
        """DI-2 zat in `risk/hmm_regime.py::_features`. Statische AST-controle.

        De toets loopt over de AST en niet over de ruwe tekst: het bestand
        CITEERT het oude idioom in zijn docstring, en dat is documentatie van de
        gesloten bevinding - geen uitvoerbare code.
        """
        path = ROOT / "src" / "tradebot" / "risk" / "hmm_regime.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offenders = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "fillna"
            and any(
                isinstance(a, ast.Call)
                and isinstance(a.func, ast.Attribute)
                and a.func.attr in {"std", "mean", "var", "quantile"}
                for a in node.args
            )
        ]
        assert offenders == [], (
            f"Sample-brede statistiek als fillna-waarde op regel(s) {offenders} "
            f"in risk/hmm_regime.py. Phase 2 sluit DI-2 onvoorwaardelijk."
        )
        assert "causal_expanding_std" in path.read_text(encoding="utf-8"), (
            "risk/hmm_regime.py gebruikt de causale vervanger niet."
        )



# --------------------------------------------------------------------------- #
# Joins over granulariteiten (exit criterium 4)
# --------------------------------------------------------------------------- #
@requires_store
class TestMultiGranularJoinIsBackwardOnly:
    """8h funding en 1d open interest op een 1d bar-raster."""

    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_joined_columns_are_truncation_invariant(self, symbol: str) -> None:
        source = certified_frame(symbol)
        joined_cols = ["funding_rate", "open_interest"]
        full = source.frame[joined_cols]
        for cut in cut_points(source):
            rebuilt = build_certified_micro_frame(
                STORE,
                DataRegister(ROOT / "artefacts" / "governance" / "data_hashes.json"),
                symbol=symbol,
                granularity=GRANULARITY,
                funding_granularity=FUNDING_GRANULARITY,
                open_interest_granularity=GRANULARITY,
                funding_tolerance=pd.Timedelta(hours=FEAT_CFG.funding_tolerance_hours),
                open_interest_tolerance=pd.Timedelta(
                    hours=FEAT_CFG.open_interest_tolerance_hours
                ),
                asset_class="crypto",
            ).truncate(cut)
            assert_bit_identical(
                rebuilt.frame[joined_cols], full, what=f"{symbol}/join", cut=cut
            )

    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_no_funding_value_predates_its_settlement(self, symbol: str) -> None:
        """Elke gekoppelde funding rate was op het beslismoment al gesetteld."""
        source = certified_frame(symbol)
        funding = STORE.load("crypto", "funding", symbol, FUNDING_GRANULARITY)
        first_settlement = int(funding["asof_ts_ns"].min())
        decision = source.frame["asof_ts_ns"].to_numpy(dtype="int64")
        has_rate = source.frame["funding_rate"].notna().to_numpy()
        assert not bool((decision[has_rate] < first_settlement).any()), (
            f"{symbol}: er is een bar met een funding rate VOOR de eerste "
            f"settlement. De join keek vooruit."
        )
