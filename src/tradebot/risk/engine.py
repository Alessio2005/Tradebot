# src/tradebot/risk/engine.py
"""De soevereine `RiskEngine` — L7, met volledig machineleesbaar auditspoor.

Phase 4, deliverable 4 / stap 6. Componeert vol-targeting, harde limieten en
kill switches tot één laag met de handtekening uit `docs/RISK_CONTRACT.md`:

    decide(desired_exposure, market_state, risk_state) -> RiskDecision

WAT DEZE ENGINE ANDERS DOET DAN `risk/portfolio.py`
---------------------------------------------------
1. **Puur.** Geen `self.equity`, geen `self._returns_history`, geen `update()`
   die stilzwijgend een buffer bijwerkt. Alle toestand komt binnen als
   `risk_state` en gaat eruit als `risk_state_out`. Zonder die eigenschap is
   exit-criterium 2 (bit-identieke output bij identieke `a_t`) onbewijsbaar.
2. **Geen alpha in het besluitpad.** Kelly-sizing op `expected_alpha`, de
   meta-label-confidence en de HMM-regimecap zijn hier NIET geport. De engine
   ziet `a_t`, gemeten marktgrootheden en zijn eigen toestand - verder niets.
3. **Volgorde uit configuratie.** `constraint_order` in `conf/risk/default.yaml`
   bepaalt de volgorde, niet de regelvolgorde in dit bestand. Een limiet die
   daar ontbreekt wordt niet stilzwijgend overgeslagen: dat is een crash.
4. **Elke ingreep geregistreerd.** `SizingDecision.reason` was een f-string
   (`"vol_mult=0.412 corr_mult=0.883"`) waaruit niet af te leiden was WELKE
   drempel bond of waar hij vandaan kwam. Elke ingreep levert hier een
   `BindingConstraint` met soort, scope, gemeten waarde, drempel en de
   `conf/risk/`-sleutel die die drempel zette.

OVER PUURHEID EN DE HALT-STORE
------------------------------
`decide()` is een pure functie van zijn drie argumenten. De optionele
`HaltStore` is een ZIJKANAAL voor duurzaamheid: wanneer een kill switch vuurt,
wordt de halt weggeschreven. Die schrijfactie verandert de teruggegeven
`RiskDecision` niet - twee aanroepen met dezelfde inputs leveren dezelfde
output, met of zonder store. Dat wordt getest en niet aangenomen.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md secties 11.1, 14, 14.1, 19 (L7), 24.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from ..schemas.config import RiskConfig
from ..utils.failfast import ConfigContractError, DataContractError, require
from ..utils.hashing import hash_config
from .contract import (
    BindingConstraint,
    MarketState,
    RiskDecision,
    RiskState,
    validate_desired_exposure,
)
from .kill_switches import (
    HaltStore,
    apply_daily_loss_governor,
    apply_drawdown_breaker,
    apply_halt,
)
from .limits import (
    apply_adv_cap,
    apply_cluster_cap,
    apply_concentration_cap,
    apply_gross_cap,
    apply_net_cap,
    apply_per_asset_cap,
    effective_relative_cap,
)
from .vol_targeting import apply_volatility_target

__all__ = ["KNOWN_CONSTRAINTS", "RiskEngine", "risk_config_hash"]

#: Elke limiet die de engine kan uitvoeren. `constraint_order` in
#: `conf/risk/default.yaml` MOET precies deze verzameling noemen: een naam die
#: hier niet in staat is een typo, en een limiet die niet in de volgorde staat
#: is een limiet die stilzwijgend niet wordt toegepast. Beide crashen.
KNOWN_CONSTRAINTS: frozenset[str] = frozenset(
    {
        "halted",
        "daily_loss_governor",
        "drawdown_breaker",
        "vol_target",
        "per_asset_cap",
        "adv_cap",
        "concentration_cap",
        "cluster_cap",
        "gross_cap",
        "net_cap",
    }
)

_TOL = 1e-12

_Step = Callable[
    [dict[str, float], MarketState, RiskState],
    tuple[dict[str, float], list[BindingConstraint], RiskState],
]


def risk_config_hash(config: RiskConfig) -> str:
    """Deterministische fingerprint van de risicoconfiguratie.

    Een risicoconfiguratie zonder hash is niet auditbaar (stap 11). De hash
    gaat over de VOLLEDIG gevalideerde config, niet over het YAML-bestand: twee
    bestanden met dezelfde inhoud maar andere sleutelvolgorde of commentaar
    horen dezelfde hash te geven, want ze produceren hetzelfde besluit.
    """
    return hash_config(config.model_dump(mode="json"))


class RiskEngine:
    """De laatste instantie vóór portfolio-constructie.

    Parameters
    ----------
    config : het gevalideerde `RiskConfig`. Elke drempel komt hiervandaan; er
        staat geen enkele in deze klasse.
    halt_store : optioneel zijkanaal dat de `HALTED`-toestand duurzaam maakt.
        Zonder store werkt de engine identiek, maar overleeft een halt de
        procesgrens niet - in productie is de store daarom verplicht.
    """

    def __init__(self, config: RiskConfig, *, halt_store: HaltStore | None = None) -> None:
        self._config = config
        self._store = halt_store
        self._config_hash = risk_config_hash(config)
        self._validate_order()

    @property
    def config(self) -> RiskConfig:
        return self._config

    @property
    def config_hash(self) -> str:
        return self._config_hash

    @property
    def halt_store(self) -> HaltStore | None:
        return self._store

    # ------------------------------------------------------------------ #
    # Configuratie
    # ------------------------------------------------------------------ #
    def _validate_order(self) -> None:
        order = tuple(self._config.constraint_order)
        require(
            len(order) > 0,
            "constraint_order is leeg. Een risicolaag zonder toepassingsvolgorde "
            "past geen enkele limiet toe; dat is 'geen limiet' met een "
            "configuratiebestand eromheen.",
            ConfigContractError,
            key="risk.constraint_order",
        )
        unknown = sorted(set(order) - KNOWN_CONSTRAINTS)
        require(
            not unknown,
            "constraint_order noemt een limiet die de engine niet kent.",
            ConfigContractError,
            key="risk.constraint_order",
            unknown=unknown,
            known=sorted(KNOWN_CONSTRAINTS),
        )
        missing = sorted(KNOWN_CONSTRAINTS - set(order))
        require(
            not missing,
            "Een geimplementeerde limiet ontbreekt in constraint_order en zou "
            "dus stilzwijgend niet worden toegepast. Een limiet weglaten is een "
            "expliciete beslissing en hoort niet per omissie te ontstaan.",
            ConfigContractError,
            key="risk.constraint_order",
            missing=missing,
        )
        require(
            len(set(order)) == len(order),
            "constraint_order bevat een dubbele limiet.",
            ConfigContractError,
            key="risk.constraint_order",
            order=list(order),
        )

    # ------------------------------------------------------------------ #
    # Toestand
    # ------------------------------------------------------------------ #
    def hydrate(self, state: RiskState) -> RiskState:
        """Lees de duurzame `HALTED`-toestand in een verse `risk_state`.

        Dit is wat een procesherstart moet aanroepen. Zonder deze stap begint
        een herstart met `halted=False` en handelt hij door een halt heen -
        precies het gat in `live/circuit_breaker.py`.
        """
        if self._store is None:
            return state
        record = self._store.load()
        if record is None:
            return state
        return RiskState(
            equity=state.equity,
            high_water_mark=state.high_water_mark,
            day_start_equity=state.day_start_equity,
            halted=True,
            halt_reason=record.reason,
            halted_at=record.halted_at,
        )

    # ------------------------------------------------------------------ #
    # Het besluit
    # ------------------------------------------------------------------ #
    def decide(
        self,
        desired_exposure: Mapping[str, float],
        market_state: MarketState,
        risk_state: RiskState,
    ) -> RiskDecision:
        """Van gewenste exposure naar toegestane exposure, met auditspoor.

        Puur: dezelfde drie argumenten leveren altijd bit-identiek dezelfde
        `RiskDecision`, ongeacht welke alpha-unit `desired_exposure` heeft
        geproduceerd.
        """
        require(
            isinstance(market_state, MarketState),
            "market_state moet een MarketState zijn.",
            DataContractError,
            got=type(market_state).__name__,
        )
        require(
            isinstance(risk_state, RiskState),
            "risk_state moet een RiskState zijn.",
            DataContractError,
            got=type(risk_state).__name__,
        )

        exposures = validate_desired_exposure(desired_exposure)
        original = dict(exposures)
        state = risk_state
        trail: list[BindingConstraint] = []

        steps = self._steps()
        for name in self._config.constraint_order:
            exposures, bound, state = steps[name](exposures, market_state, state)
            trail.extend(bound)

        permitted = {s: (0.0 if abs(w) <= _TOL else w) for s, w in exposures.items()}
        self._verify(original, permitted, market_state)
        unconstrained = len(trail) == 0
        if unconstrained:
            # Deliverable 4: nooit `a_t` ongewijzigd doorgeven zonder expliciete
            # registratie. `unconstrained=True` IS die registratie, en hij moet
            # kloppen met de uitkomst - anders is een engine die niets doet niet
            # te onderscheiden van een engine die stuk is.
            require(
                all(
                    abs(permitted[s] - original[s]) <= 1e-9 for s in original
                ),
                "Het auditspoor is leeg terwijl de exposure is veranderd. Er is "
                "een ingreep gedaan die zich niet heeft geregistreerd.",
                DataContractError,
                n_symbols=len(original),
            )

        return RiskDecision(
            permitted_exposure=permitted,
            binding_constraints=tuple(trail),
            unconstrained=unconstrained,
            config_hash=self._config_hash,
            risk_state_out=state,
        )

    # ------------------------------------------------------------------ #
    # Postconditie
    # ------------------------------------------------------------------ #
    def _verify(
        self, before: dict[str, float], after: dict[str, float],
        market_state: MarketState,
    ) -> None:
        """Controleer op de UITKOMST dat elke limiet daadwerkelijk houdt.

        Waarom dit nodig is, en niet paranoia. De boekbrede caps (gross, net)
        zijn absoluut en schalen alles proportioneel; die kunnen een eerder
        toegepaste RELATIEVE limiet niet schenden. De relatieve limieten
        onderling wel: de clusterlimiet die één cluster terugschaalt, verhoogt
        het AANDEEL van elk symbool daarbuiten, en dat kan de al toegepaste
        concentratielimiet opnieuw overschrijden.

        Dat is exact de klasse fout die `risk/portfolio.py::size_position`
        maakte door de HMM-cap ná de gross-cap toe te passen. Het verschil is
        dat die stilzwijgend een te grote positie doorliet, terwijl dit crasht:
        een configuratie waarin twee limieten elkaar tegenspreken hoort de
        operator te bereiken, niet de markt.
        """
        cfg = self._config
        gross = sum(abs(w) for w in after.values())

        # Monotoniteit (docs/RISK_CONTRACT.md sectie 5.3) — geldt altijd.
        for symbol, w in after.items():
            require(
                abs(w) <= abs(before[symbol]) + 1e-9,
                "De risicolaag heeft exposure VERGROOT.",
                DataContractError,
                symbol=symbol, before=before[symbol], after=w,
            )
            require(
                w == 0.0 or (w > 0.0) == (before[symbol] > 0.0),
                "De risicolaag heeft een teken omgedraaid.",
                DataContractError,
                symbol=symbol, before=before[symbol], after=w,
            )

        require(
            gross <= cfg.gross_cap + 1e-9,
            "Gross exposure overschrijdt de cap na toepassing van alle limieten.",
            ConfigContractError,
            key="risk.gross_cap", measured=gross, threshold=cfg.gross_cap,
        )
        require(
            abs(sum(after.values())) <= cfg.net_cap + 1e-9,
            "Netto exposure overschrijdt de cap na toepassing van alle limieten.",
            ConfigContractError,
            key="risk.net_cap", measured=abs(sum(after.values())),
            threshold=cfg.net_cap,
        )
        for symbol, w in after.items():
            require(
                abs(w) <= cfg.max_position_pct + 1e-9,
                "Per-asset cap overschreden na toepassing van alle limieten.",
                ConfigContractError,
                key="risk.max_position_pct", symbol=symbol, measured=abs(w),
                threshold=cfg.max_position_pct,
            )
        if gross > _TOL:
            # De relatieve limieten worden geverifieerd tegen hun AFDWINGBARE
            # vorm `max(cap, 1/n_actief)` - dezelfde vloer die de limietlaag
            # zelf toepast. Zie limits.effective_relative_cap.
            n_active = sum(1 for w in after.values() if abs(w) > _TOL)
            conc_cap = effective_relative_cap(max(n_active, 1), cfg.max_concentration)
            worst = max(abs(w) for w in after.values()) / gross
            require(
                worst <= conc_cap + 1e-9,
                "Concentratielimiet overschreden na toepassing van alle "
                "limieten. Twee relatieve limieten spreken elkaar tegen: de "
                "clusterlimiet verhoogt het aandeel van symbolen buiten het "
                "teruggeschaalde cluster. Verruim max_concentration of "
                "verscherp max_cluster_concentration.",
                ConfigContractError,
                key="risk.max_concentration", measured=worst,
                threshold=conc_cap,
            )
            labels = dict(market_state.cluster) or dict(cfg.clusters)
            per_cluster: dict[str, float] = {}
            for symbol, w in after.items():
                label = str(labels.get(symbol, ""))
                if label:
                    per_cluster[label] = per_cluster.get(label, 0.0) + abs(w)
            n_clusters = sum(1 for v in per_cluster.values() if v > _TOL)
            cluster_cap = effective_relative_cap(
                max(n_clusters, 1), cfg.max_cluster_concentration
            )
            for label, exposure in per_cluster.items():
                require(
                    exposure / gross <= cluster_cap + 1e-9,
                    "Clusterlimiet overschreden na toepassing van alle limieten.",
                    ConfigContractError,
                    key="risk.max_cluster_concentration", cluster=label,
                    measured=exposure / gross, threshold=cluster_cap,
                )

    # ------------------------------------------------------------------ #
    # De limieten, gebonden aan de configuratie
    # ------------------------------------------------------------------ #
    def _steps(self) -> dict[str, _Step]:
        cfg = self._config

        def _halted(
            e: dict[str, float], _m: MarketState, s: RiskState
        ) -> tuple[dict[str, float], list[BindingConstraint], RiskState]:
            out, bound = apply_halt(e, s)
            return out, bound, s

        def _daily_loss(
            e: dict[str, float], m: MarketState, s: RiskState
        ) -> tuple[dict[str, float], list[BindingConstraint], RiskState]:
            return apply_daily_loss_governor(
                e, s, daily_loss_limit=cfg.daily_loss_limit,
                store=self._store, asof_ts=m.asof_ts,
            )

        def _drawdown(
            e: dict[str, float], m: MarketState, s: RiskState
        ) -> tuple[dict[str, float], list[BindingConstraint], RiskState]:
            return apply_drawdown_breaker(
                e, s, levels=cfg.drawdown_breaker_levels,
                max_drawdown_pct=cfg.max_drawdown_pct,
                store=self._store, asof_ts=m.asof_ts,
            )

        def _vol_target(
            e: dict[str, float], m: MarketState, s: RiskState
        ) -> tuple[dict[str, float], list[BindingConstraint], RiskState]:
            out, binding = apply_volatility_target(
                e, m.sigma_hat,
                sigma_target=cfg.sigma_target, max_leverage=cfg.max_leverage,
            )
            return out, ([binding] if binding is not None else []), s

        def _per_asset(
            e: dict[str, float], _m: MarketState, s: RiskState
        ) -> tuple[dict[str, float], list[BindingConstraint], RiskState]:
            out, bound = apply_per_asset_cap(e, max_position_pct=cfg.max_position_pct)
            return out, bound, s

        def _adv(
            e: dict[str, float], m: MarketState, s: RiskState
        ) -> tuple[dict[str, float], list[BindingConstraint], RiskState]:
            out, bound = apply_adv_cap(
                e, m.adv_usd,
                participation_cap=cfg.adv_participation_cap, equity=s.equity,
            )
            return out, bound, s

        def _concentration(
            e: dict[str, float], _m: MarketState, s: RiskState
        ) -> tuple[dict[str, float], list[BindingConstraint], RiskState]:
            out, bound = apply_concentration_cap(
                e, max_concentration=cfg.max_concentration
            )
            return out, bound, s

        def _cluster(
            e: dict[str, float], m: MarketState, s: RiskState
        ) -> tuple[dict[str, float], list[BindingConstraint], RiskState]:
            # De GEMETEN clusterstructuur wint van de geconfigureerde. Normaal
            # levert de caller er geen en valt de engine terug op `conf/risk/`,
            # maar de structuur is een marktfeit: wanneer alle correlaties naar
            # 1 gaan, IS het universum een cluster (scenario S2). Een engine die
            # dan blijft rekenen met de labels uit de config, meet een
            # diversificatie die er niet meer is.
            out, bound = apply_cluster_cap(
                e, dict(m.cluster) or cfg.clusters,
                max_cluster_concentration=cfg.max_cluster_concentration,
            )
            return out, bound, s

        def _gross(
            e: dict[str, float], _m: MarketState, s: RiskState
        ) -> tuple[dict[str, float], list[BindingConstraint], RiskState]:
            out, bound = apply_gross_cap(e, cap=cfg.gross_cap)
            return out, bound, s

        def _net(
            e: dict[str, float], _m: MarketState, s: RiskState
        ) -> tuple[dict[str, float], list[BindingConstraint], RiskState]:
            out, bound = apply_net_cap(e, cap=cfg.net_cap)
            return out, bound, s

        return {
            "halted": _halted,
            "daily_loss_governor": _daily_loss,
            "drawdown_breaker": _drawdown,
            "vol_target": _vol_target,
            "per_asset_cap": _per_asset,
            "adv_cap": _adv,
            "concentration_cap": _concentration,
            "cluster_cap": _cluster,
            "gross_cap": _gross,
            "net_cap": _net,
        }

    # ------------------------------------------------------------------ #
    # Rapportage
    # ------------------------------------------------------------------ #
    def audit_header(self, asof_ts: pd.Timestamp | None = None) -> dict[str, Any]:
        """De regel die elk rapport en elke ledger-entry over deze engine draagt."""
        return {
            "config_hash": self._config_hash,
            "constraint_order": list(self._config.constraint_order),
            "sigma_target": float(self._config.sigma_target),
            "max_leverage": float(self._config.max_leverage),
            "gross_cap": float(self._config.gross_cap),
            "net_cap": float(self._config.net_cap),
            "max_drawdown_pct": float(self._config.max_drawdown_pct),
            "daily_loss_limit": float(self._config.daily_loss_limit),
            "asof_ts": None if asof_ts is None else asof_ts.isoformat(),
        }
