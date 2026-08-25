"""Clusterlabels, concentratie en de haalbaarheid van de 8 %-vol-target.

Fase-opdracht §7.2 en §19. Elk getal in
`reports/phase5_cluster_concentration_audit.md` wordt hier een assertie, zodat
het rapport niet los van de code kan verouderen.

Deze suite draait op de GECERTIFICEERDE store en wordt overgeslagen wanneer die
er niet is — een ontwikkelaar zonder DVC-pull hoort geen rode suite te zien voor
data die hij niet heeft, maar de test mag ook niet stilzwijgend groen zijn.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.risk.contract import MarketState, RiskState
from tradebot.risk.engine import RiskEngine
from tradebot.risk.limits import effective_relative_cap
from tradebot.schemas.config import (
    DataConfig,
    RiskConfig,
    VolatilityConfig,
    load_config,
)

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
CONF = ROOT / "conf"
GOV = ROOT / "artefacts" / "governance"

#: Doelwaarde uit het audit-rapport. Onder de 90 % is de target niet bereikt en
#: moet dat expliciet worden gemeld (§19).
MIN_TARGET_ATTAINMENT = 0.90

EQUITY = 100_000.0


def _certified_available() -> bool:
    data = load_config(CONF / "data/default.yaml", DataConfig)
    return (ROOT / data.pit_store_root).is_dir() and (
        GOV / "data_hashes.json").is_file()


requires_data = pytest.mark.skipif(
    not _certified_available(),
    reason="gecertificeerde PIT-store niet aanwezig (dvc pull)",
)


@pytest.fixture(scope="module")
def cfg() -> RiskConfig:
    return load_config(CONF / "risk/default.yaml", RiskConfig)


@pytest.fixture(scope="module")
def panel():
    from tradebot.data.pit_store import PitStore
    from tradebot.features.base import (
        ASOF_INDEX_NAME,
        DataRegister,
        load_certified_close_panel,
        load_certified_series,
    )
    from tradebot.volatility.ewma import ewma_volatility_panel

    data = load_config(CONF / "data/default.yaml", DataConfig)
    vol_cfg = load_config(CONF / "model/volatility.yaml", VolatilityConfig)
    store = PitStore(ROOT / data.pit_store_root)
    register = DataRegister(GOV / "data_hashes.json")

    prices = load_certified_close_panel(
        store, register, symbols=list(data.symbols), granularity="1d",
        asset_class="crypto",
    ).values
    sigma = ewma_volatility_panel(
        prices, lam=vol_cfg.ewma_lambda, burn_in_bars=vol_cfg.burn_in_bars,
        annualisation_factor=vol_cfg.annualisation_factor,
    )
    adv_cols = {}
    for symbol in data.symbols:
        df, _ = load_certified_series(
            store, register, asset_class="crypto", dataset="ohlcv",
            symbol=symbol, granularity="1d")
        idx = pd.DatetimeIndex(
            pd.to_datetime(df["asof_ts_ns"].to_numpy(), unit="ns", utc=True),
            name=ASOF_INDEX_NAME)
        # Causaal: de turnover van bar t is pas bekend op zijn close.
        adv_cols[symbol] = pd.Series(
            df["turnover"].to_numpy(dtype="float64"), index=idx
        ).rolling(30, min_periods=30).mean().shift(1)
    adv = pd.DataFrame(adv_cols).reindex(prices.index)

    rets = prices.pct_change(fill_method=None)
    usable = sigma.dropna(how="any").index
    usable = usable[usable.isin(rets.dropna(how="any").index)]
    usable = usable[usable.isin(adv.dropna(how="any").index)]
    return {
        "prices": prices, "returns": rets, "sigma": sigma, "adv": adv,
        "usable": usable, "symbols": list(prices.columns),
        "annualisation": vol_cfg.annualisation_factor,
    }


def _run_book(cfg: RiskConfig, panel, labels: dict[str, str]) -> dict:
    """Draai een 1/N-boek door de ECHTE RiskEngine en meet de uitkomst."""
    engine = RiskEngine(cfg)
    syms, usable = panel["symbols"], panel["usable"]
    a_t = {s: 1.0 for s in syms}
    rows, bound = [], {}
    for ts in usable:
        market = MarketState(
            asof_ts=ts,
            sigma_hat={s: float(panel["sigma"].at[ts, s]) for s in syms},
            adv_usd={s: float(panel["adv"].at[ts, s]) for s in syms},
            cluster=labels,
        )
        state = RiskState(equity=EQUITY, high_water_mark=EQUITY,
                          day_start_equity=EQUITY)
        decision = engine.decide(a_t, market, state)
        rows.append([decision.permitted_exposure[s] for s in syms])
        for c in decision.binding_constraints:
            bound[c.kind.value] = bound.get(c.kind.value, 0) + 1

    weights = pd.DataFrame(rows, index=usable, columns=syms)
    book = (weights.shift(1) * panel["returns"].reindex(usable)[syms]).sum(
        axis=1).dropna()
    gross = weights.abs().sum(axis=1)
    return {
        "weights": weights,
        "vol": float(book.std() * np.sqrt(panel["annualisation"])),
        "gross": float(gross.mean()),
        "bound": bound,
        "max_share": float((weights.abs().div(gross, axis=0)).max().max()),
        "n_bars": len(usable),
    }


# =========================================================================== #
# 1. De labels dragen geen informatie — dit is WAAROM ze zijn gecorrigeerd
# =========================================================================== #
@requires_data
class TestTheOldLabellingWasNotSupportedByTheData:
    def test_within_and_between_cluster_correlation_are_indistinguishable(
        self, panel
    ) -> None:
        old = {s: ("crypto_l1" if s != "LINKUSDT" else "crypto_oracle")
               for s in panel["symbols"]}
        corr = panel["returns"].dropna(how="any").corr()
        within, between = [], []
        syms = panel["symbols"]
        for i, a in enumerate(syms):
            for b in syms[i + 1:]:
                (within if old[a] == old[b] else between).append(
                    float(corr.at[a, b]))
        separation = float(np.mean(within) - np.mean(between))
        # Het rapport meet +0,0113; de bootstrap-BI bevat nul.
        assert abs(separation) < 0.05, (
            "de oude labels zouden een substantiele scheiding moeten tonen om "
            "een clusterlimiet te rechtvaardigen"
        )

    def test_the_universe_is_one_highly_correlated_group(self, panel) -> None:
        corr = panel["returns"].dropna(how="any").corr().to_numpy()
        off = corr[~np.eye(len(corr), dtype=bool)]
        assert off.mean() > 0.65, "het universum is minder gecorreleerd dan gemeten"
        assert off.min() > 0.50, "er is een symbool dat wel apart staat"


# =========================================================================== #
# 2. §19 — de feasibility gate
# =========================================================================== #
@requires_data
class TestFeasibilityGate:
    def test_the_target_is_reachable_and_reported(self, cfg, panel) -> None:
        """De 8 %-target IS haalbaar; `RISK_TARGET_INFEASIBLE` geldt niet.

        Dit onderscheid is bindend voor exit-criterium 17: de fase mag niet als
        'infeasible' afsluiten wanneer de oorzaak een corrigeerbaar
        labelingsdefect is.
        """
        out = _run_book(cfg, panel, dict(cfg.clusters))
        attainment = out["vol"] / cfg.sigma_target
        assert attainment >= MIN_TARGET_ATTAINMENT, (
            f"boekvolatiliteit {out['vol']:.4f} is {attainment:.1%} van de "
            f"target {cfg.sigma_target}; onder {MIN_TARGET_ATTAINMENT:.0%} moet "
            f"dit als RISK_TARGET_INFEASIBLE worden gemeld (§19)"
        )
        assert attainment <= 1.0 + 1e-9, (
            "het boek draait BOVEN target; de vol-targeting verkleint niet "
            "alleen maar vergroot"
        )

    def test_the_shortfall_that_remains_is_the_comonotone_bound(
        self, cfg, panel
    ) -> None:
        """Wat er nog onder target zit, is bewuste conservatisme.

        `book_sigma_hat` schat `sum |a_i| * sigma_i` - de comonotone bovengrens.
        Bij een gemiddelde correlatie van 0,735 kost die aanname ~7 %.
        """
        out = _run_book(cfg, panel, dict(cfg.clusters))
        syms, usable = panel["symbols"], panel["usable"]
        cov = panel["returns"].loc[usable].cov().to_numpy() * panel["annualisation"]
        eq = np.ones(len(syms)) / len(syms)
        true_sigma = float(np.sqrt(eq @ cov @ eq))
        comonotone = float(
            sum(eq[i] * panel["sigma"].loc[usable, s].mean()
                for i, s in enumerate(syms)))
        predicted = cfg.sigma_target * true_sigma / comonotone
        assert out["vol"] == pytest.approx(predicted, rel=0.10), (
            "de gerealiseerde volatiliteit is niet verklaard door de "
            "comonotone bovengrens alleen; er bindt nog iets anders"
        )


# =========================================================================== #
# 3. De clusterlimiet is vacuous — en dat is expliciet, niet stilzwijgend
# =========================================================================== #
@requires_data
class TestTheClusterLimitIsVacuousAndSaysSo:
    def test_the_cluster_cap_never_binds_on_this_universe(
        self, cfg, panel
    ) -> None:
        out = _run_book(cfg, panel, dict(cfg.clusters))
        assert out["bound"].get("cluster_cap", 0) == 0, (
            "de clusterlimiet bindt nog steeds; de relabeling is niet effectief"
        )

    def test_only_the_vol_target_binds(self, cfg, panel) -> None:
        out = _run_book(cfg, panel, dict(cfg.clusters))
        assert set(out["bound"]) == {"vol_target"}, (
            f"onverwachte bindende limieten: {sorted(out['bound'])}"
        )

    def test_the_limit_becomes_live_again_with_a_genuine_second_cluster(
        self, cfg, panel
    ) -> None:
        """Vacuous op DIT universum, niet kapot.

        Zonder deze test is 'de limiet doet niets' niet te onderscheiden van
        'de limiet is stuk'.
        """
        syms = panel["symbols"]
        lopsided = {s: ("A" if i < 5 else "B") for i, s in enumerate(syms)}
        out = _run_book(cfg, panel, lopsided)
        assert out["bound"].get("cluster_cap", 0) > 0, (
            "met een 5/1-verdeling hoort de clusterlimiet wel te binden"
        )

    def test_the_effective_cap_of_a_single_cluster_is_one(self, cfg) -> None:
        assert effective_relative_cap(1, cfg.max_cluster_concentration) == 1.0

    def test_the_config_still_lists_the_constraint(self, cfg) -> None:
        """Vacuous betekent niet: verwijderd."""
        assert "cluster_cap" in cfg.constraint_order


# =========================================================================== #
# 4. Het perverse concentratie-effect is weg
# =========================================================================== #
@requires_data
class TestConcentrationIsNoLongerForced:
    def test_no_symbol_is_pushed_onto_the_concentration_limit(
        self, cfg, panel
    ) -> None:
        out = _run_book(cfg, panel, dict(cfg.clusters))
        assert out["max_share"] < cfg.max_concentration - 0.05, (
            f"grootste symboolaandeel {out['max_share']:.3f} ligt tegen "
            f"max_concentration {cfg.max_concentration} aan"
        )

    def test_an_equal_weight_book_stays_equal_weight(self, cfg, panel) -> None:
        """Bij gelijke a_t hoort elk symbool 1/n van de gross te houden."""
        out = _run_book(cfg, panel, dict(cfg.clusters))
        assert out["max_share"] == pytest.approx(
            1.0 / len(panel["symbols"]), abs=0.02)

    def test_the_old_labelling_did_force_it(self, cfg, panel) -> None:
        """Bewijs dat de correctie iets heeft opgelost en niet niets deed."""
        old = {s: ("crypto_l1" if s != "LINKUSDT" else "crypto_oracle")
               for s in panel["symbols"]}
        out = _run_book(cfg, panel, old)
        assert out["max_share"] == pytest.approx(cfg.max_concentration, abs=0.01)
        assert out["bound"].get("cluster_cap", 0) > 0
        assert out["vol"] < cfg.sigma_target * 0.60, (
            "de oude labeling zou het boek ver onder target moeten houden"
        )


# =========================================================================== #
# 5. De configuratie is geregistreerd en verandert M niet
# =========================================================================== #
def test_the_new_config_hash_is_registered(cfg) -> None:
    """Fase-opdracht §6 punt 4 en §26 punt 14."""
    from tradebot.registry.risk_registry import RiskConfigRegistry
    from tradebot.risk.engine import risk_config_hash

    registry = RiskConfigRegistry(GOV / "risk_config_registry.json")
    assert risk_config_hash(cfg) in registry.hashes(), (
        "de gewijzigde risicoconfiguratie staat niet in het register; draai "
        "`python apps/run_stress.py`"
    )


def test_relabelling_did_not_change_the_hypothesis_count(tmp_path: Path) -> None:
    """§18: `M` mag niet bewegen door een risicoconfiguratie.

    De registratie gaat naar een TIJDELIJK register en niet naar
    `artefacts/governance/`. Een eerdere versie van deze test schreef wel naar
    het echte bestand, en zette daar een rij met `git_sha="test"` in - een test
    die een governance-artefact vervuilt, ondermijnt precies de auditbaarheid
    die hij hoort te bewaken.
    """
    from tradebot.registry.hypothesis_ledger import HypothesisLedger
    from tradebot.registry.risk_registry import RiskConfigRegistry
    from tradebot.risk.engine import risk_config_hash

    ledger = HypothesisLedger(GOV / "hypothesis_ledger.json")
    before = ledger.total_n_hypotheses()

    cfg = load_config(CONF / "risk/default.yaml", RiskConfig)
    scratch = RiskConfigRegistry(tmp_path / "risk_config_registry.json")
    assert scratch.register(
        config_hash=risk_config_hash(cfg), git_sha="0000000",
        config=cfg.model_dump(mode="json"), audit_header={},
        notes="hypothesis-ledger isolation test",
    )
    assert ledger.total_n_hypotheses() == before


def test_the_governance_registry_holds_no_test_written_rows() -> None:
    """Een governance-artefact bevat geen sporen van testruns."""
    import json

    doc = json.loads(
        (GOV / "risk_config_registry.json").read_text(encoding="utf-8"))
    offenders = [e for e in doc["entries"]
                 if e["git_sha"] in ("test", "", None)]
    assert not offenders, f"registry bevat test-rijen: {offenders}"
