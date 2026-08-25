"""De power-analyse die elke Phase 6-pre-registratie meedraagt in zijn hash.

WAAROM DIT IN DE PRE-REGISTRATIE HOORT EN NIET IN HET RAPPORT
==============================================================
Exit-criterium 10 eist een power-analyse per pre-registratie, met het minimaal
detecteerbare effect naast het verwachte effect. Een power-analyse die pas in
het eindrapport verschijnt, kan achteraf zijn bijgesteld op de uitkomst — en dat
is precies dezelfde manoeuvre als het verlagen van een drempel.

Daarom komt zij hier terecht in `PreRegistration.parameters`, waarover de
`preregistration_id` wordt gehasht. Verandert er één aanname, dan verandert de
ID en is de registratie aantoonbaar een andere.

DE DRIE GETALLEN DIE HIER GEMETEN ZIJN EN NIET AANGENOMEN
==========================================================
* ``mean_pairwise_correlation`` — de gemiddelde paarsgewijze correlatie van de
  zes log-returnreeksen op het bruikbare venster. Bepaalt hoeveel ONAFHANKELIJKE
  reeksen zes perpetuals waard zijn.
* ``oos_bars`` — het aantal out-of-sample bars dat de walk-forward-geometrie
  feitelijk oplevert, niet het geconfigureerde minimum.
* ``uniqueness_ratio`` — de fractie waarmee overlappende triple-barrier labels
  het nominale aantal events reduceren.

Alle drie komen uit `artefacts/governance/phase6_data_adequacy.json`, dat door
`apps/run_data_adequacy.py` is geproduceerd voordat er ook maar iets is gefit.

DE AANNAMES DIE WEL AANNAMES ZIJN
==================================
De verwachte effectgroottes en de correlatie tussen twee strategieën zijn
literatuurwaarden en geen metingen. Zij staan hier expliciet als aanname, met
bron, zodat na de run de gemeten waarde ernaast kan worden gelegd.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..schemas.config import PowerConfig
from ..validation.data_adequacy import (
    auc_minimum_detectable,
    effective_independent_series,
    mean_difference_power,
    sharpe_difference_power,
)

__all__ = ["H1_ASSUMPTIONS", "H2_ASSUMPTIONS", "H3_ASSUMPTIONS", "power_block"]

# DEFECT IN MIJN EIGEN WERK, gevonden bij het bevriezen. De scenariolijsten
# hieronder stonden eerst als TUPLES. `hash_config` serialiseert een tuple als
# een JSON-array, dus de `preregistration_id` was correct en stabiel — maar
# `freeze_preregistration` vergelijkt bij een tweede aanroep het IN-MEMORY
# document met het teruggelezen JSON, en daar is de tuple een list geworden.
# `(0.0, 0.2, 0.4) != [0.0, 0.2, 0.4]`, dus een idempotente her-freeze crashte
# met de melding "AFWIJKENDE inhoud" terwijl de inhoud identiek was.
#
# De les is algemeen: alles wat in `PreRegistration.parameters` terechtkomt,
# moet een JSON-ronde overleven zonder van type te veranderen. Lists, geen
# tuples; floats, geen numpy-scalars.

#: H1. Hansen & Lunde (2005) vergeleken 330 volatiliteitsmodellen en vonden dat
#: vrijwel niets een GARCH(1,1) verslaat; de winst van GARCH boven
#: RiskMetrics-EWMA is consistent klein en vaak niet significant. 0,10 SD is een
#: genereuze bovengrens op het verwachte effect.
H1_ASSUMPTIONS: Mapping[str, Any] = {
    "expected_effect_sd_units": 0.10,
    "expected_effect_source": "Hansen & Lunde (2005), bovengrens",
    "loss_differential_ar1_scenarios": [0.0, 0.2, 0.4],
    "proxy": "daily range-estimator (geen intraday RV beschikbaar)",
}

#: H2. `artefacts/baseline/phase5_revaluation.json`, track
#: `xs_momentum_risk_parity`, laag `L3_execution`. Het aantal jaren volgt uit
#: het bruikbare venster 2021-11-15 t/m 2026-08-23.
H2_ASSUMPTIONS: Mapping[str, Any] = {
    "baseline_sharpe": -0.33136,
    "baseline_source": "phase5_revaluation.json :: xs_momentum_risk_parity / L3_execution",
    "n_years": 4.769,
    "expected_sharpe_gain": 0.08,
    "expected_effect_source": "orde van grootte uit de regime-literatuur; ook "
                              "het voorbeeld in de fase-opdracht §0.3",
    "strategy_correlation_scenarios": [0.80, 0.90, 0.95, 0.99],
    "primary_track": "xs_momentum_risk_parity",
    "primary_track_rationale": (
        "long_only_equal_weight halteert definitief op 2022-05-10 (LUNA) en "
        "handelt ~130 van 1.743 bars; elk regime-experiment daarop meet de "
        "eerste zes maanden en daarna niets"
    ),
}

#: H3. De drempel 0,58 komt uit §24 van de architectuuraudit.
H3_ASSUMPTIONS: Mapping[str, Any] = {
    "threshold_auc": 0.58,
    "threshold_source": "ARCHITECTUUR_AUDIT_2026-08-22.md §24",
    "null_auc": 0.50,
}


def power_block(
    hypothesis: str,
    *,
    cfg: PowerConfig,
    mean_pairwise_correlation: float,
    oos_bars_per_symbol: int,
    n_symbols: int,
    uniqueness_ratio: float,
) -> dict[str, Any]:
    """Bouw het power-blok voor één hypothese.

    Het blok bevat ELK scenario, niet alleen het gunstigste. Een power-analyse
    die één getal noemt, verbergt de aanname waaronder dat getal geldt.
    """
    k_eff = effective_independent_series(n_symbols, mean_pairwise_correlation)
    common = {
        "alpha": cfg.alpha,
        "target_power": cfg.target_power,
        "two_sided": cfg.two_sided,
        "measured_mean_pairwise_correlation": mean_pairwise_correlation,
        "effective_independent_series": k_eff,
        "oos_bars_per_symbol": int(oos_bars_per_symbol),
        "n_symbols": int(n_symbols),
    }

    if hypothesis == "H1":
        scenarios = {}
        for ar1 in H1_ASSUMPTIONS["loss_differential_ar1_scenarios"]:
            design_effect = (1.0 + ar1) / (1.0 - ar1)
            n_eff = oos_bars_per_symbol * k_eff / design_effect
            analysis = mean_difference_power(
                n_eff, float(H1_ASSUMPTIONS["expected_effect_sd_units"]), cfg,
                assumptions={"loss_differential_ar1": ar1,
                             "serial_design_effect": design_effect},
            )
            scenarios[f"ar1_{ar1:.1f}"] = analysis.as_record()
        return {**common, **dict(H1_ASSUMPTIONS), "scenarios": scenarios}

    if hypothesis == "H2":
        scenarios = {}
        for corr in H2_ASSUMPTIONS["strategy_correlation_scenarios"]:
            analysis = sharpe_difference_power(
                float(H2_ASSUMPTIONS["n_years"]),
                float(H2_ASSUMPTIONS["baseline_sharpe"]),
                corr,
                float(H2_ASSUMPTIONS["expected_sharpe_gain"]),
                cfg,
            )
            scenarios[f"rho_{corr:.2f}"] = analysis.as_record()
        return {**common, **dict(H2_ASSUMPTIONS), "scenarios": scenarios}

    if hypothesis == "H3":
        nominal = oos_bars_per_symbol * n_symbols
        variants = {
            "nominal": float(nominal),
            "uniqueness_corrected": nominal * uniqueness_ratio,
            "uniqueness_and_cross_section": (
                nominal * uniqueness_ratio * k_eff / n_symbols),
        }
        scenarios = {}
        for name, n in variants.items():
            half = max(int(n / 2), 2)
            analysis = auc_minimum_detectable(
                half, half, cfg,
                null_auc=float(H3_ASSUMPTIONS["null_auc"]),
                expected_auc=float(H3_ASSUMPTIONS["threshold_auc"]),
                assumptions={"effective_n": n, "variant": name},
            )
            scenarios[name] = analysis.as_record()
        return {
            **common, **dict(H3_ASSUMPTIONS),
            "measured_uniqueness_ratio": uniqueness_ratio,
            "scenarios": scenarios,
        }

    raise ValueError(f"Onbekende hypothese {hypothesis!r}; verwacht H1, H2 of H3.")
