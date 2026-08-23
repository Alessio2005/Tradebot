"""Phase 2, deliverable 6 - TRUNCATIE-INVARIANTIE van elke L3-feature.

Dit is exit criterium 1 van Phase 2.

De toets (stap 5) is onbarmhartig simpel. Bereken de feature-matrix over het
volledige venster `T`. Kap daarna de INPUT af op drie punten `t1, t2, t3`, en
bereken opnieuw. Voor elke `tau <= t_i` moet de waarde BIT-IDENTIEK zijn. Wijkt
er ook maar een bit af, dan gebruikt de feature informatie uit `t+1..T` en is hij
gefalsificeerd - ongeacht hoe plausibel zijn formule oogt.

De tests draaien op de 18 gecertificeerde reeksen uit Phase 1 (6 symbolen x
ohlcv/funding/open_interest). Ontbreekt de PIT-store, dan wordt overgeslagen met
een expliciete reden - nooit stilzwijgend als geslaagd gerapporteerd.

NEGATIEVE CONTROLE
------------------
`TestTheTestCanGoRed` bevat twee bewust lekkende features. Zij MOETEN falen op
precies deze toets. Een invariantietest die nooit rood is geweest, bewijst niets
over de features die hij groen verklaart; hij bewijst alleen dat hij niets meet.
`TestDI2IsClosed` doet hetzelfde voor het concrete DI-2-idioom.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from tradebot.data.pit_store import PitStore
from tradebot.features.base import BaseFeature, CertifiedFrame, DataRegister, InputSpec
from tradebot.features.microstructure import build_certified_micro_frame
from tradebot.features.registry import build_default_registry, current_git_sha
from tradebot.features.volatility import causal_expanding_std, log_returns
from tradebot.schemas.config import DataConfig, FeatureConfig, load_config
from tradebot.utils.failfast import CausalityViolationError

ROOT = Path(__file__).resolve().parents[2]
DATA_CFG = load_config(ROOT / "conf" / "data" / "default.yaml", DataConfig)
FEAT_CFG = load_config(ROOT / "conf" / "features" / "default.yaml", FeatureConfig)
STORE = PitStore(ROOT / DATA_CFG.pit_store_root)

SYMBOLS = list(DATA_CFG.symbols)
GRANULARITY = "1d"
FUNDING_GRANULARITY = "8h"

#: Vaste seed: een gefaalde invariantietest moet exact reproduceerbaar zijn.
TRUNCATION_SEED = 20260823
#: Drie snijpunten, zoals stap 5 voorschrijft.
N_CUTS = 3

pytestmark = pytest.mark.lookahead


def _have(dataset: str, symbol: str, granularity: str) -> bool:
    return bool(STORE.partitions("crypto", dataset, symbol, granularity))


requires_store = pytest.mark.skipif(
    not (
        _have("ohlcv", "BTCUSDT", GRANULARITY)
        and _have("funding", "BTCUSDT", FUNDING_GRANULARITY)
        and _have("open_interest", "BTCUSDT", GRANULARITY)
    ),
    reason=(
        "PIT-store is leeg. Draai eerst de Phase 1-ingestion "
        "(apps/ingest_crypto.py) en scripts/build_data_register.py."
    ),
)

_FRAME_CACHE: dict[str, CertifiedFrame] = {}


def certified_frame(symbol: str) -> CertifiedFrame:
    """Gecertificeerd inputframe (ohlcv + funding + open interest) per symbool."""
    if symbol not in _FRAME_CACHE:
        _FRAME_CACHE[symbol] = build_certified_micro_frame(
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
        )
    return _FRAME_CACHE[symbol]


def cut_points(source: CertifiedFrame) -> list[pd.Timestamp]:
    """Drie snijpunten uit de tweede helft van de reeks, deterministisch geloot.

    De eerste helft wordt gemeden zodat elke feature zijn burn-in ruim voorbij
    is; een snijpunt binnen de burn-in zou twee NaN-blokken vergelijken en de
    toets betekenisloos maken.
    """
    idx = source.frame.index
    rng = np.random.default_rng(TRUNCATION_SEED)
    lo, hi = len(idx) // 2, len(idx) - 1
    positions = sorted(rng.choice(np.arange(lo, hi), size=N_CUTS, replace=False))
    return [idx[int(p)] for p in positions]


def assert_bit_identical(
    truncated: pd.DataFrame, full: pd.DataFrame, *, what: str, cut: pd.Timestamp
) -> None:
    """Elke waarde op of vóór `cut` moet bit-identiek zijn, NaN's inbegrepen."""
    expected = full.loc[full.index <= truncated.index[-1]]
    assert len(truncated) == len(expected), (
        f"{what}: rijaantal wijkt af na truncatie op {cut} "
        f"({len(truncated)} vs {len(expected)})"
    )
    assert truncated.index.equals(expected.index), f"{what}: index wijkt af na {cut}"
    for col in expected.columns:
        a = truncated[col].to_numpy(dtype="float64")
        b = expected[col].to_numpy(dtype="float64")
        same_nan = np.isnan(a) == np.isnan(b)
        assert same_nan.all(), (
            f"{what}/{col}: NaN-patroon verschuift na truncatie op {cut}; "
            f"{int((~same_nan).sum())} rij(en) afwijkend"
        )
        finite = ~np.isnan(a)
        diff = np.abs(a[finite] - b[finite])
        n_bad = int((diff != 0.0).sum())
        assert n_bad == 0, (
            f"{what}/{col}: {n_bad} waarde(n) veranderen door data NA {cut}. "
            f"max |diff| = {float(diff.max()) if diff.size else 0.0:.17g}. "
            f"Dit is een lookahead-lek; de feature is gefalsificeerd."
        )


# --------------------------------------------------------------------------- #
# Exit criterium 1 - elke geregistreerde feature
# --------------------------------------------------------------------------- #
REGISTRY = build_default_registry(FEAT_CFG)
FEATURE_IDS = [f.feature_id for f in REGISTRY.features]


@requires_store
class TestTruncationInvariancePerFeature:
    """Elke feature afzonderlijk, op elk gecertificeerd symbool."""

    @pytest.mark.parametrize("symbol", SYMBOLS)
    @pytest.mark.parametrize(
        "feature", REGISTRY.features, ids=[f.feature_id for f in REGISTRY.features]
    )
    def test_feature_is_truncation_invariant(
        self, feature: BaseFeature, symbol: str
    ) -> None:
        source = certified_frame(symbol)
        full = feature.transform(source).values
        for cut in cut_points(source):
            truncated = feature.transform(source.truncate(cut)).values
            assert_bit_identical(
                truncated, full, what=f"{symbol}/{feature.feature_id}", cut=cut
            )


@requires_store
class TestTruncationInvarianceOfTheMatrix:
    """De volledige matrix in een keer: ook de compositie mag niet lekken."""

    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_matrix_is_truncation_invariant(self, symbol: str) -> None:
        source = certified_frame(symbol)
        pipeline = REGISTRY.pipeline()
        full = pipeline.transform(source).values
        assert len(full) > FEAT_CFG.microstructure.funding_zscore_min_periods
        for cut in cut_points(source):
            truncated = pipeline.transform(source.truncate(cut)).values
            assert_bit_identical(truncated, full, what=f"{symbol}/matrix", cut=cut)

    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_matrix_hash_is_stable_under_truncation(self, symbol: str) -> None:
        """De feature_hash beschrijft de DEFINITIE, niet het gerealiseerde venster.

        Truncatie van de input verandert de matrix-inhoud maar niet de identiteit
        van de features; anders zou elke walk-forward fold een ander artefact
        opleveren en zou de registry onbruikbaar zijn als sleutel.
        """
        source = certified_frame(symbol)
        sha = current_git_sha()
        full_hash = REGISTRY.matrix_hash(data_hashes=source.data_hashes, git_sha=sha)
        for cut in cut_points(source):
            cut_hash = REGISTRY.matrix_hash(
                data_hashes=source.truncate(cut).data_hashes, git_sha=sha
            )
            assert cut_hash == full_hash


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


# --------------------------------------------------------------------------- #
# Negatieve controle - de toets moet aantoonbaar rood kunnen worden
# --------------------------------------------------------------------------- #
class _FutureLeakingFeature(BaseFeature):
    """Bewust lek: gebruikt de close van de VOLGENDE bar (`shift(-1)`)."""

    name: ClassVar[str] = "deliberately_leaking_next_bar"
    input_spec: ClassVar[InputSpec] = InputSpec(datasets=("ohlcv",), columns=("close",))

    def __init__(self) -> None:
        super().__init__(params={})

    @property
    def burn_in_period(self) -> int:
        return 1

    @property
    def output_columns(self) -> tuple[str, ...]:
        return ("leak_next_close_ratio",)

    def _compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        close = frame["close"].astype("float64")
        # `ffill` dekt uitsluitend de LAATSTE rij, die na shift(-1) geen
        # opvolger meer heeft. Zonder die dekking crasht het NaN-contract van
        # BaseFeature al vóór de invariantietoets, en dan bewijst deze
        # negatieve controle niets over de toets zelf.
        out = ((close.shift(-1) / close) - 1.0).ffill()
        out.iloc[0] = np.nan
        return pd.DataFrame(
            {self.output_columns[0]: out.to_numpy(dtype="float64")}, index=frame.index
        )


class _SampleWideScalingFeature(BaseFeature):
    """Bewust lek: normaliseert met het gemiddelde over de VOLLEDIGE sample.

    Dit is de subtiele variant, en in de praktijk de gevaarlijkste: er staat
    nergens een `shift(-1)`, elke rij oogt causaal, en toch verandert elke
    waarde zodra er data achteraan wordt geplakt.
    """

    name: ClassVar[str] = "deliberately_leaking_sample_scaler"
    input_spec: ClassVar[InputSpec] = InputSpec(datasets=("ohlcv",), columns=("close",))

    def __init__(self) -> None:
        super().__init__(params={})

    @property
    def burn_in_period(self) -> int:
        return 1

    @property
    def output_columns(self) -> tuple[str, ...]:
        return ("leak_sample_scaled_return",)

    def _compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        r = log_returns(frame["close"])
        out = r / r.std()
        return pd.DataFrame(
            {self.output_columns[0]: out.to_numpy(dtype="float64")}, index=frame.index
        )


@requires_store
class TestTheTestCanGoRed:
    """Bewijs door falen: een lekkende feature MOET deze toets rood maken."""

    @pytest.mark.parametrize(
        "leaker",
        [_FutureLeakingFeature(), _SampleWideScalingFeature()],
        ids=["shift_minus_one", "sample_wide_scaler"],
    )
    def test_leaking_feature_fails_truncation_invariance(
        self, leaker: BaseFeature
    ) -> None:
        source = certified_frame("BTCUSDT")
        full = leaker.transform(source).values
        failures = 0
        for cut in cut_points(source):
            truncated = leaker.transform(source.truncate(cut)).values
            with pytest.raises(AssertionError):
                assert_bit_identical(
                    truncated, full, what=f"leak/{leaker.name}", cut=cut
                )
            failures += 1
        assert failures == N_CUTS, (
            "De invariantietoets ving het lek niet op elk snijpunt. Een toets "
            "die een bekend lek doorlaat, bewijst niets over de features die hij "
            "groen verklaart."
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
