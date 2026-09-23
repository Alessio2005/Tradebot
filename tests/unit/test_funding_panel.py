"""Phase 10, stap 1B — het dagelijkse fundingpaneel dat de engine boekt.

`apps/run_phase5_baseline.py` gaf `run_all_layers` tot dusver een fundingpaneel
van louter nullen; `backtest/engine.py` boekt de fundingrate die het krijgt
PRECIES EEN KEER per bar op de notional (`accounting.py::apply_funding`,
`qty * mark_price * rate`). Een dagbar draagt tot drie 8-uurs afrekeningen, dus
het paneel moet de SOM van de afrekeningen binnen die bar dragen, niet de
laatst bekende rate — dat laatste is wat `asof_join(direction="backward")` zou
opleveren (correct voor een feature, verkeerd voor een boeking; zie
`features/positioning.py::build_certified_micro_frame`).

Elke test hieronder bewijst een eigenschap door haar te falsifieren: waar een
implementatie de valkuil (asof/last-known in plaats van som) zou nemen, moet
de test rood worden.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.data.funding_panel import daily_funding_panel
from tradebot.data.pit_store import PartitionRef, PitStore
from tradebot.features.base import DataRegister
from tradebot.utils.failfast import DataContractError

ASSET_CLASS = "crypto"
GRANULARITY = "8h"


def _funding_frame(event_ts_utc: list[pd.Timestamp], rates: list[float],
                   symbol: str) -> pd.DataFrame:
    ev_ns = np.array([int(t.value) for t in event_ts_utc], dtype="int64")
    return pd.DataFrame({
        "event_ts_ns": ev_ns,
        "asof_ts_ns": ev_ns,  # funding is bekend op het moment van settlement
        "funding_rate": np.asarray(rates, dtype="float64"),
        "funding_interval_hours": np.full(len(ev_ns), 8, dtype="int64"),
        "symbol": [symbol] * len(ev_ns),
    })


def _write_certified(tmp_path: Path, frames: dict[str, pd.DataFrame]
                     ) -> tuple[PitStore, DataRegister]:
    """Schrijf elke (symbool -> frame) naar een verse store en certificeer hem."""
    store = PitStore(tmp_path / "pit_store")
    hashes: dict[str, str] = {}
    for symbol, df in frames.items():
        ref = PartitionRef(ASSET_CLASS, "funding", symbol, GRANULARITY, "2024-01-01")
        h = store.write(df, ref, source="test")
        hashes[DataRegister.key(ASSET_CLASS, "funding", symbol, GRANULARITY)] = h
    register_path = tmp_path / "data_hashes.json"
    register_path.write_text(json.dumps(hashes), encoding="utf-8")
    return store, DataRegister(register_path)


def _bar_index(*days: str) -> pd.DatetimeIndex:
    """Dagbars gelabeld op hun SLUITMOMENT (asof_ts), net als het prijspaneel."""
    return pd.DatetimeIndex(
        [pd.Timestamp(d, tz="UTC") for d in days], name="asof_ts")


class TestDailyFundingPanel:
    def test_the_sum_not_the_last(self, tmp_path: Path) -> None:
        """Drie afrekeningen op één dag leveren hun SOM, niet de laatste."""
        symbol = "BTCUSDT"
        bar = _bar_index("2024-01-01", "2024-01-02")  # bar[1] dekt [1e jan, 2e jan)
        events = [pd.Timestamp("2024-01-01T00:00:00Z"),
                  pd.Timestamp("2024-01-01T08:00:00Z"),
                  pd.Timestamp("2024-01-01T16:00:00Z")]
        rates = [1e-4, 2e-4, 3e-4]
        store, register = _write_certified(
            tmp_path, {symbol: _funding_frame(events, rates, symbol)})

        panel = daily_funding_panel(
            store, register, symbols=[symbol], asset_class=ASSET_CLASS,
            funding_granularity=GRANULARITY, bar_index=bar)

        # De asof-valkuil zou hier 3e-4 (de laatst bekende rate) opleveren.
        assert panel.loc[bar[1], symbol] == pytest.approx(6e-4)
        assert panel.loc[bar[1], symbol] != pytest.approx(3e-4)

    def test_the_sign_survives_netting(self, tmp_path: Path) -> None:
        """Een dag met +2e-4 en -1e-4 levert +1e-4: het teken wordt niet weggegooid."""
        symbol = "ETHUSDT"
        bar = _bar_index("2024-01-01", "2024-01-02")
        events = [pd.Timestamp("2024-01-01T00:00:00Z"),
                  pd.Timestamp("2024-01-01T08:00:00Z")]
        rates = [2e-4, -1e-4]
        store, register = _write_certified(
            tmp_path, {symbol: _funding_frame(events, rates, symbol)})

        panel = daily_funding_panel(
            store, register, symbols=[symbol], asset_class=ASSET_CLASS,
            funding_granularity=GRANULARITY, bar_index=bar)

        assert panel.loc[bar[1], symbol] == pytest.approx(1e-4)

    def test_a_day_without_a_settlement_is_zero_not_nan(self, tmp_path: Path) -> None:
        """Een bar zonder afrekening is 0.0. De engine leest `.at[ts, symbol]`
        en een NaN daar zou zich stil door de hele P&L voortplanten."""
        symbol = "SOLUSDT"
        bar = _bar_index("2024-01-01", "2024-01-02", "2024-01-03")
        events = [pd.Timestamp("2024-01-01T00:00:00Z")]
        rates = [1e-4]
        store, register = _write_certified(
            tmp_path, {symbol: _funding_frame(events, rates, symbol)})

        panel = daily_funding_panel(
            store, register, symbols=[symbol], asset_class=ASSET_CLASS,
            funding_granularity=GRANULARITY, bar_index=bar)

        assert not panel.isna().any().any()
        assert panel.loc[bar[2], symbol] == pytest.approx(0.0)

    def test_the_panel_sits_on_exactly_the_price_bar_grid(self, tmp_path: Path) -> None:
        """`build_slices` indexeert noch positioneel noch tolerant: index,
        kolommen en volgorde moeten exact overeenkomen met het prijspaneel."""
        symbols = ["BTCUSDT", "ETHUSDT"]
        bar = _bar_index("2024-01-01", "2024-01-02", "2024-01-03")
        frames = {
            s: _funding_frame(
                [pd.Timestamp("2024-01-01T00:00:00Z")], [1e-4], s)
            for s in symbols
        }
        store, register = _write_certified(tmp_path, frames)

        panel = daily_funding_panel(
            store, register, symbols=symbols, asset_class=ASSET_CLASS,
            funding_granularity=GRANULARITY, bar_index=bar)

        assert panel.index.equals(bar)
        assert list(panel.columns) == symbols

    def test_a_series_that_fails_certification_crashes(self, tmp_path: Path) -> None:
        """Een reeks waarvan de inhoud niet met het register overeenkomt hoort
        te crashen, niet stil door te rekenen."""
        symbol = "AVAXUSDT"
        bar = _bar_index("2024-01-01", "2024-01-02")
        store, register = _write_certified(
            tmp_path, {symbol: _funding_frame(
                [pd.Timestamp("2024-01-01T00:00:00Z")], [1e-4], symbol)})
        # Corrumpeer het register: de gecertificeerde hash klopt niet meer.
        register._hashes[DataRegister.key(
            ASSET_CLASS, "funding", symbol, GRANULARITY)] = "0" * 16

        with pytest.raises(DataContractError):
            daily_funding_panel(
                store, register, symbols=[symbol], asset_class=ASSET_CLASS,
                funding_granularity=GRANULARITY, bar_index=bar)


@pytest.mark.integration
class TestDailyFundingPanelOnTheRealStore:
    """Eenhedencontrole op de gecertificeerde store — belangrijker dan zij oogt.

    Een gesommeerde dagrate op een 8h-perp-funding hoort ordegrootte 1e-4 te
    zijn. 1e-2 betekent een eenhedenfout (bv. percentage in plaats van
    fractie); 1e-8 betekent een lege of verkeerd uitgelijnde reeks.
    """

    def test_the_realised_panel_has_no_nan_and_is_the_right_order_of_magnitude(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[2]
        from tradebot.backtest.baseline_report import load_baseline_configs
        from tradebot.features.base import load_certified_close_panel

        cfg = load_baseline_configs(root)
        store = PitStore(root / cfg["data"].pit_store_root)
        register = DataRegister(root / "artefacts/governance/data_hashes.json")
        symbols = list(cfg["data"].symbols)
        prices = load_certified_close_panel(
            store, register, symbols=symbols, granularity="1d",
            asset_class="crypto").values

        panel = daily_funding_panel(
            store, register, symbols=symbols, asset_class="crypto",
            funding_granularity="8h", bar_index=prices.index)

        assert not panel.isna().any().any()
        for symbol in symbols:
            mean_abs = float(panel[symbol].abs().mean())
            assert 1e-6 < mean_abs < 1e-2, (
                f"{symbol}: gemiddelde |gesommeerde dagfunding| = {mean_abs:.3e} "
                "valt buiten het plausibele bereik voor een 8h-perp gesommeerd "
                "naar dagen -- vermoedelijk een eenhedenfout, geen bevinding."
            )

    def test_solusdt_settled_every_two_hours_in_late_2022_and_nothing_is_doubled(
        self,
    ) -> None:
        """Fase 11 stap 5.3. 41 SOLUSDT-dagen in W_DEV dragen geen drie
        afrekeningen maar 12 (39x), 9 en 6: van 2022-11-10 t/m 2022-12-20
        rekende Bybit SOLUSDT elke TWEE uur af. Nagelopen tegen de Bybit-API:
        483 records, 0 extra, 0 ontbrekend, 0 rateverschillen. Het zijn echte
        betalingen en geen dubbelingen -- deze test zorgt dat niemand ze
        later als uitschieter 'opschoont'.

        Wat NIET klopt, is het veld `funding_interval_hours`: dat is een
        configconstante (8) die de ingestie op elke rij stempelt, geen meting
        (DI-35). De laatste assertie legt dat vast, zodat een reparatie van het
        veld deze test bewust moet bijwerken."""
        root = Path(__file__).resolve().parents[2]
        from tradebot.backtest.baseline_report import load_baseline_configs
        from tradebot.features.base import load_certified_series

        cfg = load_baseline_configs(root)
        df, _ = load_certified_series(
            PitStore(root / cfg["data"].pit_store_root),
            DataRegister(root / "artefacts/governance/data_hashes.json"),
            asset_class="crypto", dataset="funding", symbol="SOLUSDT",
            granularity="8h")
        ts = pd.to_datetime(df["event_ts_ns"], unit="ns", utc=True)
        counts = df.groupby(ts.dt.floor("D")).size()
        w_dev = counts[(counts.index >= "2021-11-15") & (counts.index <= "2025-09-04")]
        odd = w_dev[w_dev != 3]
        assert odd.value_counts().to_dict() == {12: 39, 9: 1, 6: 1}
        assert str(odd.index.min().date()) == "2022-11-10"
        assert str(odd.index.max().date()) == "2022-12-20"
        assert not ts.duplicated().any()
        gaps_h = ts[(ts >= "2022-11-10 08:00") & (ts < "2022-12-21")].diff().dropna()
        assert (gaps_h == pd.Timedelta(hours=2)).mean() > 0.99
        assert set(df["funding_interval_hours"].unique()) == {8}


class TestThereIsOneFundingRoute:
    def test_the_legacy_per_bar_loader_is_gone(self) -> None:
        """Fase 11 stap 5.2, R-3: één implementatie per grootheid. De
        L3-fundingkosten komen uit `daily_funding_panel` op de gecertificeerde
        store. `data/funding.py` was een tweede route naar dezelfde grootheid,
        las een niet-gecertificeerde bron (`macro_crypto_*.parquet`) en gaf
        bij een ontbrekend bestand stil nullen terug. Een tweede implementatie
        is een defect, ook als zij hetzelfde getal zou geven."""
        import importlib.util

        assert importlib.util.find_spec("tradebot.data.funding") is None
