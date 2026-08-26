"""M0 Causal Vol-Buckets lekt niet — Phase 6, deliverable 14.

De naam van de module bevat het woord "Causal". Dat is een BEWERING, en dit
bestand is de meting ervan.

De toets is dezelfde als in `test_feature_causality.py` en even onbarmhartig:
classificeer het volledige venster, kap de INPUT daarna af op `t`, classificeer
opnieuw, en eis dat de toewijzing op elke bar tot en met `t` bit-identiek is.
Wijkt er iets af, dan gebruikt M0 informatie uit de toekomst -- hoe causaal de
formule er ook uitziet.

Dat is een sterker bewijs dan een code-inspectie, want het vindt ook lekken die
je niet had bedacht. En het is hier extra nodig: M0 is de BASELINE. Lekt de
baseline, dan is elk oordeel over het HMM gemeten tegen een tegenstander die
vals speelt, en dan is `UNPROVEN` net zo waardeloos als `PROMOTED`.

NEGATIEVE CONTROLE
------------------
`TestTheTestCanGoRed` classificeert met een z-score die de volledige sample
gebruikt -- het DI-2-idioom, één regel code en de meest voorkomende vorm van dit
lek. Die moet op deze toets FALEN. Een invariantietest die nooit rood is
geweest, bewijst niets over wat hij groen verklaart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.regime import classify_vol_buckets
from tradebot.schemas.config import M0BucketConfig

SEED = 20260826
LAM = 0.94
BURN_IN = 60


def _ohlc(n: int = 900, *, seed: int = SEED) -> pd.DataFrame:
    """Een reeks met een duidelijke volatiliteitswisseling halverwege.

    De wisseling is er met opzet: op een reeks met constante volatiliteit zou
    M0 vrijwel alles NORMAAL noemen en zou de truncatietest triviaal slagen
    omdat er niets te verschuiven valt.
    """
    rng = np.random.default_rng(seed)
    sigma = np.where(np.arange(n) < n // 2, 0.015, 0.055)
    returns = rng.normal(0.0, sigma)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[100.0], close[:-1]])
    span = np.abs(rng.normal(0.0, sigma)) * close
    return pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) + span,
        "low": np.minimum(open_, close) - span,
        "close": close,
    })


CFG = M0BucketConfig(zscore_min_periods=120)


def _append_crash(frame: pd.DataFrame, *, n_bars: int = 60) -> pd.DataFrame:
    """Plak een echte crash-periode ACHTER de reeks.

    Een RUN van bars en niet één. DEFECT IN MIJN EIGEN TESTWERK, hier
    vastgelegd: de eerste versie plakte één crashbar en verwachtte dat de
    lekkende variant daarop rood zou worden. Dat gebeurde niet — één bar
    verschuift het gemiddelde en de standaardafwijking over ~880 waarden met
    ongeveer 1/880, en dat is te weinig om ook maar één bucket over de
    0,5-drempel te tillen. Het lek zat er wel; mijn probe had de gevoeligheid
    niet.

    Een negatieve controle die te zwak is om het lek te zien dat zij moet
    aantonen, bewijst hetzelfde als een groene test die nooit rood kan worden:
    niets. 60 bars is bovendien realistischer — LUNA was geen enkele bar.
    """
    rng = np.random.default_rng(SEED + 99)
    last = float(frame["close"].iloc[-1])
    returns = rng.normal(-0.05, 0.20, n_bars)
    close = last * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[last], close[:-1]])
    span = np.abs(rng.normal(0.0, 0.20, n_bars)) * close
    crash = pd.DataFrame({
        "open": open_,
        "high": np.maximum(open_, close) + span,
        "low": np.minimum(open_, close) - span,
        "close": close,
    })
    return pd.concat([frame, crash], ignore_index=True)


def _classify(frame: pd.DataFrame) -> pd.Series:
    return classify_vol_buckets(
        frame, CFG, ewma_lambda=LAM, ewma_burn_in_bars=BURN_IN,
        symbol="TEST").buckets


class TestTruncationInvariance:
    @pytest.mark.parametrize("cut", [200, 400, 600, 899])
    def test_classification_up_to_the_cut_is_bit_identical(
        self, cut: int,
    ) -> None:
        full = _classify(_ohlc())
        truncated = _classify(_ohlc().iloc[: cut + 1])
        a = full.iloc[: cut + 1].to_numpy(dtype=np.float64)
        b = truncated.to_numpy(dtype=np.float64)
        # NaN op dezelfde plaatsen, en identieke waarden daarbuiten.
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        np.testing.assert_array_equal(a[~np.isnan(a)], b[~np.isnan(b)])

    @pytest.mark.parametrize("cut", [300, 500, 700])
    def test_the_underlying_signals_are_invariant_too(self, cut: int) -> None:
        """Niet alleen het eindoordeel maar ook de twee assen eronder.

        Een lek dat de z-score verschuift maar de bucket toevallig niet
        verandert, zou anders onopgemerkt blijven tot de drempels wijzigen.
        """
        frame = _ohlc()
        full = classify_vol_buckets(
            frame, CFG, ewma_lambda=LAM, ewma_burn_in_bars=BURN_IN)
        cut_frame = classify_vol_buckets(
            frame.iloc[: cut + 1], CFG, ewma_lambda=LAM,
            ewma_burn_in_bars=BURN_IN)
        for name in ("zscore", "atr_ratio"):
            a = getattr(full, name).iloc[: cut + 1].to_numpy(dtype=np.float64)
            b = getattr(cut_frame, name).to_numpy(dtype=np.float64)
            np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
            np.testing.assert_allclose(
                a[~np.isnan(a)], b[~np.isnan(b)], rtol=0.0, atol=0.0,
                err_msg=f"{name} is niet truncatie-invariant")

    def test_a_future_crash_does_not_change_the_past(self) -> None:
        """De scherpste vorm: plak een crash-periode ACHTER de reeks.

        Bij een full-sample z-score blaast die crash de standaardafwijking op
        en verschuift elke eerdere classificatie. De causale variant mag er
        niets van merken — de bars ervoor zijn niet veranderd, dus hun oordeel
        mag dat ook niet zijn.
        """
        frame = _ohlc()
        extended = _append_crash(frame)

        before = _classify(frame).to_numpy(dtype=np.float64)
        after = _classify(extended).iloc[: len(frame)].to_numpy(dtype=np.float64)
        np.testing.assert_array_equal(np.isnan(before), np.isnan(after))
        np.testing.assert_array_equal(
            before[~np.isnan(before)], after[~np.isnan(after)])


class TestTheTestCanGoRed:
    """De negatieve controle: bewijs dat deze toets een lek DETECTEERT."""

    @staticmethod
    def _leaky_buckets(frame: pd.DataFrame) -> pd.Series:
        """Het DI-2-idioom: standaardiseer op de VOLLEDIGE sample.

        Eén regel, volkomen onschuldig ogend, en hij stopt de volatiliteit van
        het einde van het venster in de classificatie van het begin.
        """
        returns = np.log(frame["close"].astype("float64")).diff()
        vol = returns.rolling(20, min_periods=20).std()
        log_vol = np.log(vol)
        z = (log_vol - log_vol.mean()) / log_vol.std()   # <-- het lek
        out = pd.Series(np.nan, index=frame.index, dtype="float64")
        defined = z.notna()
        out[defined] = 1.0
        out[defined & (z >= 0.5)] = 2.0
        out[defined & (z <= -0.5)] = 0.0
        return out

    def test_the_leaky_variant_fails_truncation_invariance(self) -> None:
        frame = _ohlc()
        cut = 400
        full = self._leaky_buckets(frame).iloc[: cut + 1].to_numpy(
            dtype=np.float64)
        truncated = self._leaky_buckets(frame.iloc[: cut + 1]).to_numpy(
            dtype=np.float64)
        mask = ~np.isnan(full) & ~np.isnan(truncated)
        assert np.any(full[mask] != truncated[mask]), (
            "De lekkende variant kwam ONGESCHONDEN door de truncatietest. Dan "
            "meet de toets niets en bewijst zij ook niets over M0.")

    def test_the_leaky_variant_is_moved_by_a_future_crash(self) -> None:
        """Dezelfde controle op de crash-test hierboven."""
        frame = _ohlc()
        extended = _append_crash(frame)
        before = self._leaky_buckets(frame).to_numpy(dtype=np.float64)
        after = self._leaky_buckets(extended).iloc[: len(frame)].to_numpy(
            dtype=np.float64)
        mask = ~np.isnan(before) & ~np.isnan(after)
        assert np.any(before[mask] != after[mask]), (
            "De lekkende variant bleef ONGEMOEID door een crash van 60 bars. "
            "Dan is deze negatieve controle te ongevoelig om het lek aan te "
            "tonen dat zij moet aantonen.")


class TestNoImplicitNormalDuringBurnIn:
    def test_the_burn_in_is_undefined_and_not_normal(self) -> None:
        """"Geen oordeel" is een uitkomst; "normaal" is een bewering.

        Zou de opstartfase met NORMAAL worden gevuld, dan zou M0 op de eerste
        120 bars een marktbewering doen op grond van niets — en die bars zouden
        meetellen in elke occupancy en elke Sharpe per regime.
        """
        buckets = _classify(_ohlc())
        assert buckets.iloc[:100].isna().all()
        assert buckets.notna().any()

    def test_no_forward_fill_anywhere(self) -> None:
        """Een gat middenin blijft een gat.

        Een `ffill` zou hier de laatst bekende toestand doortrekken, en dat is
        precies de stille degradatie die dit platform verbiedt.
        """
        frame = _ohlc()
        frame.loc[500, ["open", "high", "low", "close"]] = np.nan
        buckets = _classify(frame)
        assert np.isnan(buckets.iloc[500])
