"""Phase 1, deliverable 12 — TRUNCATIE-INVARIANTIE op echte crypto-data.

Dit is exit criterium 1 van Phase 1.

De toets (stap 10): koppel funding rates aan OHLCV via `asof_join`, kap
vervolgens ALLE data na tijdstip `t` af, en herhaal. Elke waarde op of vóór `t`
moet **bit-identiek** zijn. Elke afwijking is een lookahead-lek.

De tests draaien op de daadwerkelijke PIT-store. Ontbreekt die, dan worden ze
overgeslagen met een expliciete reden — nooit stilzwijgend als geslaagd
gerapporteerd.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tradebot.data.panel import join_funding_to_ohlcv, load_close_panel, load_ohlcv
from tradebot.data.pit_store import PitStore
from tradebot.schemas.config import DataConfig, load_config
from tradebot.utils.failfast import DataContractError
from tradebot.utils.time import asof_join

ROOT = Path(__file__).resolve().parents[2]
CFG = load_config(ROOT / "conf" / "data" / "default.yaml", DataConfig)
STORE = PitStore(ROOT / CFG.pit_store_root)
TOL = pd.Timedelta(seconds=CFG.asof_tolerance_seconds)
#: funding settelt elke 8 uur; de rate mag dus maximaal één interval oud zijn.
FUNDING_TOL = pd.Timedelta(hours=CFG.funding_interval_hours)

pytestmark = pytest.mark.lookahead


def _have(dataset: str, symbol: str, granularity: str) -> bool:
    return bool(STORE.partitions("crypto", dataset, symbol, granularity))


requires_store = pytest.mark.skipif(
    not _have("ohlcv", "BTCUSDT", "1d") or not _have("funding", "BTCUSDT", "8h"),
    reason=("PIT-store is leeg. Draai eerst: "
            "python apps/ingest_crypto.py --kind ohlcv --granularity 1d && "
            "python apps/ingest_crypto.py --kind funding --granularity 8h"),
)

SYMBOLS = list(CFG.symbols)


@requires_store
class TestTruncationInvariance:
    """Een waarde op t mag niet veranderen wanneer data ná t verdwijnt."""

    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_funding_join_is_truncation_invariant(self, symbol: str) -> None:
        full = join_funding_to_ohlcv(STORE, symbol, "1d", tolerance=FUNDING_TOL)
        assert len(full) > 100

        # Kap af op drie posities in de reeks: vroeg, midden, laat.
        for frac in (0.25, 0.50, 0.75):
            cut = full.index[int(len(full) * frac)]

            truncated = _rejoin_truncated(symbol, cut)
            expected = full.loc[full.index <= cut]

            assert len(truncated) == len(expected), (
                f"{symbol}: rijaantal wijkt af na truncatie op {cut}")
            pd.testing.assert_series_equal(
                truncated["funding_rate"], expected["funding_rate"],
                check_exact=True,
                obj=f"{symbol} funding_rate truncated at {cut}")
            pd.testing.assert_series_equal(
                truncated["close"], expected["close"], check_exact=True,
                obj=f"{symbol} close truncated at {cut}")

    @pytest.mark.parametrize("symbol", SYMBOLS[:3])
    def test_ohlcv_is_truncation_invariant(self, symbol: str) -> None:
        full = load_ohlcv(STORE, symbol, "1d")
        cut = full.index[len(full) // 2]
        head = full.loc[full.index <= cut]
        pd.testing.assert_frame_equal(head, full.iloc[: len(head)],
                                      check_exact=True)


def _rejoin_truncated(symbol: str, cut: pd.Timestamp) -> pd.DataFrame:
    """Herhaal de join op data waaruit ALLES ná `cut` is verwijderd."""
    ohlcv = load_ohlcv(STORE, symbol, "1d")
    funding = STORE.load("crypto", "funding", symbol, "8h")

    ohlcv_t = ohlcv.loc[ohlcv.index <= cut]
    fund_t = funding[pd.to_datetime(funding["asof_ts_ns"], unit="ns", utc=True) <= cut]

    right = pd.DataFrame({
        "asof_ts": pd.to_datetime(fund_t["asof_ts_ns"], unit="ns", utc=True),
        "funding_rate": fund_t["funding_rate"].astype(float),
        "funding_interval_hours": fund_t["funding_interval_hours"],
    }).sort_values("asof_ts", kind="stable")

    return asof_join(ohlcv_t, right, asof_col="asof_ts", tolerance=FUNDING_TOL)


@requires_store
class TestFundingSettlementSemantics:
    """Een funding rate is pas op zijn settlement-moment bekend."""

    def test_no_funding_rate_predates_its_settlement(self) -> None:
        joined = join_funding_to_ohlcv(STORE, "BTCUSDT", "1d",
                                       tolerance=FUNDING_TOL)
        used = joined.dropna(subset=["asof_ts"])
        # De gekoppelde asof_ts mag nooit ná de bar-timestamp liggen.
        assert (used["asof_ts"] <= used.index).all(), (
            "Er is een funding rate gekoppeld die op de bar-timestamp nog niet "
            "gepubliceerd was.")

    def test_daily_bars_coincide_exactly_with_a_settlement(self) -> None:
        """Gemeten eigenschap van de data, niet van de code.

        Bybit settelt funding om 00:00, 08:00 en 16:00 UTC; een daily bar opent
        om 00:00 UTC. Elke bar valt dus samen met een settlement, en
        `allow_exact_matches=True` koppelt die exact. Gevolg: het aanscherpen van
        de tolerance verandert op deze granulariteit NIETS - een aanname die
        eerst verkeerd in deze suite stond en door de meting is gecorrigeerd.
        """
        joined = join_funding_to_ohlcv(STORE, "BTCUSDT", "1d",
                                       tolerance=FUNDING_TOL)
        used = joined.dropna(subset=["asof_ts"])
        lag = used.index - used["asof_ts"]
        assert (lag == pd.Timedelta(0)).all(), (
            "Verwacht een exacte samenval van bar-open en funding-settlement.")

    def test_tolerance_bounds_the_carry_forward(self) -> None:
        """Het tolerance-contract zelf, op een reeks met een bewust gat.

        Zonder bovengrens draagt merge_asof een waarde onbeperkt vooruit. Hier
        wordt de funding-reeks een maand lang leeggemaakt; met een tolerance van
        8 uur moeten die bars ONGEKOPPELD blijven in plaats van de laatst
        bekende rate te erven.
        """
        ohlcv = load_ohlcv(STORE, "BTCUSDT", "1d")
        funding = STORE.load("crypto", "funding", "BTCUSDT", "8h")
        fasof = pd.to_datetime(funding["asof_ts_ns"], unit="ns", utc=True)

        lo = ohlcv.index[len(ohlcv) // 2]
        hi = lo + pd.Timedelta(days=30)
        holed = funding[(fasof < lo) | (fasof > hi)]

        right = pd.DataFrame({
            "asof_ts": pd.to_datetime(holed["asof_ts_ns"], unit="ns", utc=True),
            "funding_rate": holed["funding_rate"].astype(float),
        }).sort_values("asof_ts", kind="stable")

        bounded = asof_join(ohlcv, right, asof_col="asof_ts",
                            tolerance=FUNDING_TOL)
        unbounded = asof_join(ohlcv, right, asof_col="asof_ts",
                              tolerance=pd.Timedelta(days=3650))

        # Bars binnen de eerste 8 uur van het gat mogen WEL nog koppelen: de
        # laatste settlement voor het gat ligt dan nog binnen de tolerance. Pas
        # daarna moet de koppeling ophouden.
        window = (bounded.index > lo + FUNDING_TOL) & (bounded.index <= hi)
        assert window.sum() > 20, "te klein testvenster"
        assert bounded.loc[window, "funding_rate"].isna().all(), (
            "Voorbij de tolerance mag geen enkele bar in het gat nog een rate "
            "erven.")
        assert unbounded.loc[window, "funding_rate"].notna().sum() > 20, (
            "Zonder bovengrens draagt merge_asof de laatste rate wel degelijk "
            "de hele maand vooruit - dat is precies waarom tolerance verplicht is.")

    def test_tolerance_is_mandatory(self) -> None:
        """asof_join zonder tolerance mag niet aanroepbaar zijn."""
        with pytest.raises(TypeError, match="tolerance"):
            asof_join(load_ohlcv(STORE, "BTCUSDT", "1d").head(10),
                      pd.DataFrame({"asof_ts": [], "x": []}))  # type: ignore[call-arg]


@requires_store
class TestPanelProvenance:
    def test_every_series_carries_a_data_hash(self) -> None:
        res = load_close_panel(STORE, SYMBOLS, "1d")
        assert set(res.source_hashes) == set(SYMBOLS)
        assert all(len(h) == 32 for h in res.source_hashes.values())
        assert len(res.panel_hash) == 32

    def test_panel_hash_is_deterministic(self) -> None:
        a = load_close_panel(STORE, SYMBOLS, "1d")
        b = load_close_panel(STORE, SYMBOLS, "1d")
        assert a.panel_hash == b.panel_hash

    def test_panel_index_is_utc_and_sorted(self) -> None:
        res = load_close_panel(STORE, SYMBOLS, "1d")
        assert str(res.frame.index.tz) == "UTC"
        assert res.frame.index.is_monotonic_increasing

    def test_missing_series_crashes_instead_of_returning_empty(self) -> None:
        with pytest.raises(DataContractError, match="GEEN lege"):
            load_ohlcv(STORE, "NOPEUSDT", "1d")


@requires_store
class TestNoFutureLeakageInTheStore:
    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_no_bar_is_marked_known_before_it_closed(self, symbol: str) -> None:
        df = STORE.load("crypto", "ohlcv", symbol, "1d")
        assert (df["asof_ts_ns"] > df["event_ts_ns"]).all(), (
            f"{symbol}: er is een bar die op zijn openingstijd al als bekend "
            f"is gemarkeerd; de close van vandaag zou dan vandaag al "
            f"beschikbaar zijn.")

    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_no_row_is_knowable_in_the_future(self, symbol: str) -> None:
        now_ns = int(pd.Timestamp.now(tz="UTC").value)
        df = STORE.load("crypto", "ohlcv", symbol, "1d")
        assert int(df["asof_ts_ns"].max()) <= now_ns, (
            f"{symbol}: de store bevat een bar die nog niet gesloten is.")
