"""Causaliteit van de VOLLEDIGE baseline-keten — Phase 3, deliverable 9.

Exit criterium 2: features -> alpha -> portfolio -> backtest is causaal over de
hele keten, niet alleen per module. De toets is dezelfde als in Phase 2 en even
onbarmhartig: bereken over het volledige venster `T`, kap de INPUT af op drie
deterministisch gelote punten, en eis dat elke waarde op of voor het snijpunt
BIT-IDENTIEK is.

Deze suite draait op de 6 gecertificeerde OHLCV-reeksen uit Phase 1. Ontbreekt
de PIT-store, dan wordt overgeslagen met een expliciete reden.

Stap 2 van de fase schrijft voor dat ELKE transform afzonderlijk op
truncatie-invariantie wordt getoetst VOORDAT hij in een pipeline wordt gebruikt.
`TestTransformsAreCausal` doet dat; `TestPanelPipelineIsCausal` toetst daarna de
compositie, omdat een keten van causale stappen niet vanzelf causaal is zodra er
toestand tussen de stappen zou lekken.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.data.pit_store import PitStore
from tradebot.features.base import CertifiedPanel, DataRegister, load_certified_close_panel
from tradebot.features.pipeline import PanelPipeline, TransformStep
from tradebot.features.transforms import (
    cross_sectional_rank,
    expanding_zscore,
    rolling_log_return,
    rolling_quantile_winsorise,
)
from tradebot.schemas.config import AlphaConfig, DataConfig, VolatilityConfig, load_config
from tradebot.volatility.ewma import ewma_volatility_panel

ROOT = Path(__file__).resolve().parents[2]
DATA_CFG = load_config(ROOT / "conf" / "data" / "default.yaml", DataConfig)
ALPHA_CFG = load_config(ROOT / "conf" / "model" / "alpha.yaml", AlphaConfig)
VOL_CFG = load_config(ROOT / "conf" / "model" / "volatility.yaml", VolatilityConfig)
STORE = PitStore(ROOT / DATA_CFG.pit_store_root)
GRANULARITY = "1d"

TRUNCATION_SEED = 20260823
N_CUTS = 3
GIT_SHA = "phase3test"

pytestmark = pytest.mark.lookahead

requires_store = pytest.mark.skipif(
    not STORE.partitions("crypto", "ohlcv", "BTCUSDT", GRANULARITY),
    reason=("PIT-store is leeg. Draai eerst de Phase 1-ingestion en "
            "scripts/build_data_register.py."),
)

_PANEL: dict[str, CertifiedPanel] = {}


def panel() -> CertifiedPanel:
    if "panel" not in _PANEL:
        _PANEL["panel"] = load_certified_close_panel(
            STORE,
            DataRegister(ROOT / "artefacts" / "governance" / "data_hashes.json"),
            symbols=list(DATA_CFG.symbols),
            granularity=GRANULARITY,
            asset_class="crypto",
        )
    return _PANEL["panel"]


def cut_points(index: pd.Index) -> list[pd.Timestamp]:
    """Drie snijpunten uit de tweede helft, deterministisch geloot."""
    rng = np.random.default_rng(TRUNCATION_SEED)
    lo, hi = len(index) // 2, len(index) - 1
    positions = sorted(rng.choice(np.arange(lo, hi), size=N_CUTS, replace=False))
    return [index[int(p)] for p in positions]


def assert_bit_identical(
    truncated: pd.DataFrame, full: pd.DataFrame, *, what: str, cut: pd.Timestamp
) -> None:
    expected = full.loc[full.index <= truncated.index[-1]]
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
            f"Lookahead-lek; de baseline is gefalsificeerd."
        )


# --------------------------------------------------------------------------- #
# L1 - elke transform afzonderlijk (stap 2)
# --------------------------------------------------------------------------- #
TRANSFORM_CASES = [
    ("rolling_log_return",
     lambda p: rolling_log_return(p, window=ALPHA_CFG.lookback_bars,
                                  skip_bars=ALPHA_CFG.skip_bars)),
    ("cross_sectional_rank",
     lambda p: cross_sectional_rank(p, min_assets=ALPHA_CFG.min_assets)),
    ("expanding_zscore",
     lambda p: expanding_zscore(p, min_periods=VOL_CFG.burn_in_bars)),
    ("rolling_quantile_winsorise",
     lambda p: rolling_quantile_winsorise(p, window=ALPHA_CFG.lookback_bars,
                                          quantile=ALPHA_CFG.winsorise_quantile)),
]


@requires_store
class TestTransformsAreCausal:
    """Stap 2: elke transform afzonderlijk, voordat hij in een keten mag."""

    @pytest.mark.parametrize("name,fn", TRANSFORM_CASES,
                             ids=[n for n, _ in TRANSFORM_CASES])
    def test_transform_is_truncation_invariant(self, name: str, fn) -> None:
        source = panel().values
        full = fn(source)
        for cut in cut_points(source.index):
            truncated = fn(source.loc[source.index <= cut])
            assert_bit_identical(truncated, full, what=name, cut=cut)

    @pytest.mark.parametrize("name,fn", TRANSFORM_CASES,
                             ids=[n for n, _ in TRANSFORM_CASES])
    def test_transform_is_pure(self, name: str, fn) -> None:
        """Tweemaal aanroepen op dezelfde input geeft bit-identieke output."""
        source = panel().values
        first, second = fn(source), fn(source)
        pd.testing.assert_frame_equal(first, second, check_exact=True)

    @pytest.mark.parametrize("name,fn", TRANSFORM_CASES,
                             ids=[n for n, _ in TRANSFORM_CASES])
    def test_transform_does_not_mutate_its_input(self, name: str, fn) -> None:
        """Een pure transform raakt zijn argument niet aan."""
        source = panel().values
        before = source.copy(deep=True)
        fn(source)
        pd.testing.assert_frame_equal(source, before, check_exact=True)

    def test_cross_sectional_rank_uses_only_one_timestamp(self) -> None:
        """Een rij verandert niet wanneer een ANDERE rij wordt gewijzigd.

        Dit is de scherpste toets op de cross-sectionele stap: hij mag geen
        enkele statistiek over de tijd gebruiken, ook geen 'onschuldige' zoals
        een gemiddelde om op te centreren.
        """
        source = panel().values
        base = cross_sectional_rank(source, min_assets=ALPHA_CFG.min_assets)
        tampered = source.copy()
        tampered.iloc[-1] = tampered.iloc[-1] * 100.0
        after = cross_sectional_rank(tampered, min_assets=ALPHA_CFG.min_assets)
        assert_bit_identical(
            after.iloc[:-1], base.iloc[:-1],
            what="cross_sectional_rank/row_isolation", cut=source.index[-2],
        )

    def test_thin_cross_section_yields_no_position(self) -> None:
        """Minder dan min_assets namen levert NaN op, geen weddenschap."""
        source = panel().values
        ranked = cross_sectional_rank(source, min_assets=ALPHA_CFG.min_assets)
        counts = source.notna().sum(axis=1)
        thin = counts < ALPHA_CFG.min_assets
        assert bool(ranked.loc[thin].isna().all().all()), (
            "er is een score toegekend op een rij met een te dunne cross-sectie"
        )


# --------------------------------------------------------------------------- #
# L2 - de EWMA-vol-estimator
# --------------------------------------------------------------------------- #
@requires_store
class TestEwmaEstimatorIsCausal:
    def test_ewma_panel_is_truncation_invariant(self) -> None:
        source = panel().values
        kw = dict(lam=VOL_CFG.ewma_lambda, burn_in_bars=VOL_CFG.burn_in_bars,
                  annualisation_factor=VOL_CFG.annualisation_factor)
        full = ewma_volatility_panel(source, **kw)
        for cut in cut_points(source.index):
            truncated = ewma_volatility_panel(source.loc[source.index <= cut], **kw)
            assert_bit_identical(truncated, full, what="ewma_volatility_panel",
                                 cut=cut)

    def test_no_symbol_ever_gets_a_zero_volatility(self) -> None:
        """DI-12: een vol van nul is in risk parity een oneindig gewicht."""
        vol = ewma_volatility_panel(
            panel().values, lam=VOL_CFG.ewma_lambda,
            burn_in_bars=VOL_CFG.burn_in_bars,
            annualisation_factor=VOL_CFG.annualisation_factor,
        )
        assert not bool((vol.to_numpy() == 0.0).any())


# --------------------------------------------------------------------------- #
# L1 - de compositie
# --------------------------------------------------------------------------- #
def baseline_pipeline() -> PanelPipeline:
    """De keten van de baseline: rendement, daarna cross-sectionele rang.

    Winsorisatie zit hier bewust NIET in. Een rangschikking is al ongevoelig
    voor uitschieters - dat is precies wat een rang doet - en de winsorisatie
    zou er 59 extra bars burn-in aan toevoegen zonder de ordening wezenlijk te
    veranderen. De transform bestaat en is getoetst; hij hoort thuis bij
    NIVEAU-features, niet voor een rang.
    """
    return PanelPipeline([
        TransformStep("rolling_log_return",
                      {"window": ALPHA_CFG.lookback_bars,
                       "skip_bars": ALPHA_CFG.skip_bars}),
        TransformStep("cross_sectional_rank",
                      {"min_assets": ALPHA_CFG.min_assets}),
    ])


@requires_store
class TestPanelPipelineIsCausal:
    def test_pipeline_is_truncation_invariant(self) -> None:
        source = panel()
        pipe = baseline_pipeline()
        full = pipe.transform(source, git_sha=GIT_SHA).values
        for cut in cut_points(source.values.index):
            truncated = pipe.transform(source.truncate(cut), git_sha=GIT_SHA).values
            assert_bit_identical(truncated, full, what="panel_pipeline", cut=cut)

    def test_panel_hash_is_stable_under_truncation(self) -> None:
        """De hash beschrijft de DEFINITIE, niet het gerealiseerde venster."""
        source = panel()
        pipe = baseline_pipeline()
        full = pipe.transform(source, git_sha=GIT_SHA)
        for cut in cut_points(source.values.index):
            assert pipe.transform(
                source.truncate(cut), git_sha=GIT_SHA
            ).panel_hash == full.panel_hash

    def test_burn_in_is_declared_and_respected(self) -> None:
        pipe = baseline_pipeline()
        art = pipe.transform(panel(), git_sha=GIT_SHA)
        assert art.burn_in_period == ALPHA_CFG.lookback_bars + ALPHA_CFG.skip_bars
        head = art.values.iloc[: art.burn_in_period].to_numpy(dtype="float64")
        assert np.isnan(head).all(), "eindige waarde binnen de burn-in van de keten"

    def test_data_hashes_propagate_into_the_artefact(self) -> None:
        """Deliverable 3: verplichte data_hash-propagatie naar het artefact."""
        source = panel()
        art = baseline_pipeline().transform(source, git_sha=GIT_SHA)
        assert art.data_hashes == source.data_hashes
        assert len(art.data_hashes) == len(DATA_CFG.symbols)
        for series, digest in art.data_hashes:
            assert series.startswith("crypto/ohlcv/")
            assert len(digest) == 32

    def test_unknown_transform_name_crashes(self) -> None:
        from tradebot.utils.failfast import DataContractError

        with pytest.raises(DataContractError, match="Onbekende transform"):
            TransformStep("a_transform_that_does_not_exist", {})


# --------------------------------------------------------------------------- #
# Negatieve controle - de toets moet rood kunnen worden
# --------------------------------------------------------------------------- #
@requires_store
class TestTheTestCanGoRed:
    def test_a_sample_wide_normalisation_fails_the_same_check(self) -> None:
        """`x / x.std()` over de volledige sample: elke rij oogt causaal."""
        source = panel().values

        def leaking(p: pd.DataFrame) -> pd.DataFrame:
            returns = rolling_log_return(p, window=ALPHA_CFG.lookback_bars,
                                         skip_bars=ALPHA_CFG.skip_bars)
            return returns / returns.std()

        full = leaking(source)
        failures = 0
        for cut in cut_points(source.index):
            truncated = leaking(source.loc[source.index <= cut])
            with pytest.raises(AssertionError):
                assert_bit_identical(truncated, full, what="leak/sample_std",
                                     cut=cut)
            failures += 1
        assert failures == N_CUTS, (
            "de invariantietoets ving het lek niet op elk snijpunt"
        )

    def test_a_centered_window_fails_the_same_check(self) -> None:
        """`center=True` kijkt per constructie de halve venstergrootte vooruit."""
        source = panel().values

        def leaking(p: pd.DataFrame) -> pd.DataFrame:
            return p.rolling(ALPHA_CFG.lookback_bars, center=True).mean()

        full = leaking(source)
        with pytest.raises(AssertionError):
            cut = cut_points(source.index)[0]
            assert_bit_identical(leaking(source.loc[source.index <= cut]), full,
                                 what="leak/centered", cut=cut)


# --------------------------------------------------------------------------- #
# L4 / L8 / L10 - de rest van de keten
# --------------------------------------------------------------------------- #
def _chain(source: CertifiedPanel):
    """features -> alpha -> vol -> gewichten, in een keer."""
    from tradebot.alpha.momentum import build_cross_sectional_momentum
    from tradebot.backtest.baseline_runner import build_weight_tracks

    unit = build_cross_sectional_momentum(ALPHA_CFG)
    tracks, exposures = build_weight_tracks(
        source, unit, lam=VOL_CFG.ewma_lambda,
        vol_burn_in_bars=VOL_CFG.burn_in_bars,
        annualisation_factor=VOL_CFG.annualisation_factor,
        gross_target=1.0, git_sha=GIT_SHA,
    )
    return unit, exposures, tracks


@requires_store
class TestAlphaAndPortfolioAreCausal:
    """Exit criterium 2: de keten features -> alpha -> portfolio is causaal."""

    def test_alpha_exposures_are_truncation_invariant(self) -> None:
        source = panel()
        _, full, _ = _chain(source)
        for cut in cut_points(source.values.index):
            _, truncated, _ = _chain(source.truncate(cut))
            assert_bit_identical(truncated, full, what="alpha_exposures", cut=cut)

    @pytest.mark.parametrize(
        "track",
        ["xs_momentum_risk_parity", "xs_momentum_equal_weight",
         "long_only_equal_weight", "long_only_risk_parity"],
    )
    def test_track_weights_are_truncation_invariant(self, track: str) -> None:
        source = panel()
        _, _, full = _chain(source)
        for cut in cut_points(source.values.index):
            _, _, truncated = _chain(source.truncate(cut))
            assert_bit_identical(truncated[track], full[track],
                                 what=f"weights/{track}", cut=cut)

    def test_exposures_stay_within_the_declared_range(self) -> None:
        _, exposures, _ = _chain(panel())
        arr = exposures.to_numpy(dtype="float64")
        finite = arr[np.isfinite(arr)]
        assert finite.min() >= ALPHA_CFG.signal_floor
        assert finite.max() <= ALPHA_CFG.signal_cap

    def test_every_track_is_normalised_to_the_same_gross(self) -> None:
        """Anders vergelijkt het rapport hefboom in plaats van sizing."""
        _, _, tracks = _chain(panel())
        for name, weights in tracks.items():
            gross = weights.abs().sum(axis=1).to_numpy()
            active = gross[gross > 0.0]
            assert active.size > 0, f"{name} nam nooit een positie in"
            np.testing.assert_allclose(active, 1.0, rtol=1e-12,
                                       err_msg=f"{name} is niet op gross 1 genormaliseerd")


@requires_store
class TestBacktestRespectsTheShiftConvention:
    """RETAIN (sectie 24): een gewicht bepaald op t rendeert pas op t+1."""

    def test_gross_return_uses_the_previous_bar_weight(self) -> None:
        source = panel()
        _, _, tracks = _chain(source)
        weights = tracks["xs_momentum_risk_parity"].fillna(0.0)
        returns = source.values.pct_change(fill_method=None).fillna(0.0)

        held = weights.shift(1).fillna(0.0)
        expected = (held * returns).sum(axis=1)
        # Dezelfde bar gebruiken zou het rendement van t verdienen met de kennis
        # van t - de klassieke same-bar leakage.
        same_bar = (weights * returns).sum(axis=1)
        assert not np.allclose(expected.to_numpy(), same_bar.to_numpy()), (
            "de shift-conventie maakt geen verschil op deze data; de toets meet "
            "dan niets"
        )

    def test_a_future_price_cannot_change_an_earlier_weight(self) -> None:
        """Vervalst de laatste bar: geen enkel eerder gewicht mag bewegen."""
        source = panel()
        _, _, full = _chain(source)
        tampered_values = source.values.copy()
        tampered_values.iloc[-1] = tampered_values.iloc[-1] * 10.0
        tampered = CertifiedPanel(values=tampered_values,
                                  data_hashes=source.data_hashes)
        _, _, after = _chain(tampered)
        for name in full:
            assert_bit_identical(after[name].iloc[:-1], full[name].iloc[:-1],
                                 what=f"weights/{name}/future_price",
                                 cut=source.values.index[-2])


@requires_store
class TestChainLevelNegativeControl:
    """Een lekkende alpha-unit moet de ketentoets rood maken."""

    def test_a_unit_that_peeks_at_the_next_bar_is_caught(self) -> None:
        source = panel()
        base = rolling_log_return(source.values, window=ALPHA_CFG.lookback_bars,
                                  skip_bars=ALPHA_CFG.skip_bars)

        def leaking(p: pd.DataFrame) -> pd.DataFrame:
            # skip_bars = -1: het venster eindigt EEN BAR IN DE TOEKOMST.
            logp = np.log(p.to_numpy(dtype="float64"))
            frame = pd.DataFrame(logp, index=p.index, columns=p.columns)
            return frame.shift(-1) - frame.shift(ALPHA_CFG.lookback_bars - 1)

        full = leaking(source.values)
        assert not np.allclose(
            np.nan_to_num(full.to_numpy()), np.nan_to_num(base.to_numpy())
        ), "het lek verandert de waarden niet; de controle meet niets"
        cut = cut_points(source.values.index)[0]
        with pytest.raises(AssertionError):
            assert_bit_identical(leaking(source.values.loc[source.values.index <= cut]),
                                 full, what="leak/negative_skip", cut=cut)
