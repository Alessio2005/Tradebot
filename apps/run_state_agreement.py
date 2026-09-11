# apps/run_state_agreement.py
"""Stap 8 -- EWMA(0,94) tegen walk-forward GARCH(1,1)-t op de TOESTANDSAS.

De wetenschap staat in `regime/state_agreement.py`; deze app doet uitsluitend
bedrading. Zij bouwt twee sigma-dak-panelen op de ONTWIKKELSAMPLE, wijst beide
toe met dezelfde bevroren kwantielregel, en telt.

Nul trials: er wordt niets uit geselecteerd. De uitkomst wordt in
`reports/phase10_state_agreement.md` tegen de VOORAF vastgelegde beslisregel
van substap 8.5 gelegd -- die regel is niet van deze run afhankelijk en wordt
er ook niet op bijgesteld.

Geen CLI-vlaggen en geen JSON-artefact (YAGNI): het rapport is de uitkomst, en
deze stdout is wat het citeert.

R-6 stelt een grens van 80 regels aan een app; dit bestand telt er 204:
30 docstring, 25 import, 35 commentaar, 21 leeg en 93 code. Dat wordt niet
weggeschreven als "past wel" -- het staat als meting in DI-4, precies zoals
`apps/run_vol_competition.py` zijn 134 noteert. De grootste post is `_print`,
en die is er omdat het rapport uitsluitend deze stdout citeert: er is bewust
geen JSON-artefact om uit te renderen.

    python apps/run_state_agreement.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from tradebot.alpha.momentum import build_cross_sectional_momentum
from tradebot.backtest.baseline_report import load_baseline_configs
from tradebot.cv.walk_forward import WalkForwardCV
from tradebot.data.phase6_universe import load_phase6_universe
from tradebot.features.registry import current_git_sha
from tradebot.regime.state import VolState, assign_by_variance
from tradebot.regime.state_agreement import AgreementReport, agreement
from tradebot.schemas.config import (
    ValidationConfig,
    adequacy_config,
    load_config,
    regime_config,
)
from tradebot.validation.holdout import development_slice
from tradebot.volatility.garch import (
    GARCH_FAMILY,
    ConvergenceSummary,
    summarise_convergence,
    walk_forward_variance_forecasts,
)

LOCK = ROOT / "artefacts/governance/holdout_lock.json"

#: De uitdager van H1, op de horizon waarop de toestand leeft: de toestand van
#: bar `t` leest sigma-dak op `t-1`, dus de eenstapsforecast. h = 1 is een van
#: de twee gepre-registreerde H1-horizonnen; er wordt hier niet over horizonnen
#: gezocht, want dan zou deze stap selecteren en niet tellen.
SPEC, HORIZON = GARCH_FAMILY["garch"], 1

#: De VOORAF vastgelegde beslisregel van substap 8.5. Zij staat hier als
#: constante zodat de run haar niet kan herformuleren op het getal dat eruit
#: komt; het oordeel zelf hoort in het rapport en niet in deze stdout thuis.
MIN_IDENTICAL, MAX_DECISION_CHANGE = 0.95, 0.05


def _garch_sigma(
    returns: pd.DataFrame, cv: WalkForwardCV
) -> tuple[pd.DataFrame, dict[str, ConvergenceSummary]]:
    """Het walk-forward GARCH-paneel, per symbool en zonder enige vulling.

    De wortel is geen schaalafstemming maar leesbaarheid: `assign_by_variance`
    is invariant onder elke monotone herschaling (ruling P35), dus variantie of
    volatiliteit levert exact dezelfde toestanden. Wat NaN is blijft NaN -- een
    ontbrekende fold is dekking die er niet is, en het rapport telt hem.
    """
    panel, convergence = {}, {}
    for symbol in map(str, returns.columns):
        variance, fits = walk_forward_variance_forecasts(
            returns[symbol], SPEC, cv, adequacy_config(),
            symbol=symbol, horizon=HORIZON)
        panel[symbol] = np.sqrt(variance)
        convergence[symbol] = summarise_convergence(fits)
    return pd.DataFrame(panel, index=returns.index), convergence


def _print(
    report: AgreementReport,
    convergence: dict[str, ConvergenceSummary],
    garch: pd.DataFrame,
    n_bars: int,
) -> None:
    print(f"\nDEKKING  (ontwikkelsample: {n_bars} bars x {len(garch.columns)} namen)")
    # `comparable` is de poort van H1 ZELF (`ConvergenceSummary.comparable`),
    # hier gelezen en niet nagebouwd (R-3): boven
    # `max_boundary_solution_ratio` bestaat de onvoorwaardelijke variantie
    # niet en is de forecast een random walk in variantie. Deze stap
    # selecteert daar NIETS op weg -- een naam uitsluiten omdat zijn fits
    # lelijk zijn, zou van de telling op de rest een keuze maken -- maar een
    # overeenstemmingsgetal op zulke forecasts hoort niet zonder dit label
    # te worden gelezen.
    adequacy = adequacy_config()
    for symbol, summary in convergence.items():
        print(f"  {symbol:<9} garch-bars {int(garch[symbol].notna().sum()):>5}"
              f"  folds {summary.n_fits:>2}  geconvergeerd "
              f"{summary.convergence_ratio:>6.1%}  rand {summary.boundary_ratio:>6.1%}"
              f"  H1-vergelijkbaar "
              f"{'ja ' if summary.comparable(adequacy) else 'NEE'}")
    print(f"  cellen vergeleken {report.n_cells_compared} van "
          f"{report.n_cells_compared + report.n_cells_dropped} "
          f"({report.n_cells_dropped} weggevallen)")
    print(f"  bars   vergeleken {report.n_bars_compared} van "
          f"{report.n_bars_compared + report.n_bars_dropped} "
          f"({report.n_bars_dropped} weggevallen; elke naam moet bekend zijn)")
    # WELKE bars, en niet alleen hoeveel. Een walk-forward-paneel is aan BEIDE
    # uiteinden leeg -- voor de eerste trainperiode en na de laatste volle
    # testperiode -- dus het vergelijkingsvenster ligt MIDDEN in de
    # ontwikkelsample en niet aan het eind, waar een lezer het zou zoeken.
    print(f"  venster   {report.bars_compared[0].date()} .. "
          f"{report.bars_compared[-1].date()}  "
          f"(waarover elke barbreuk hieronder loopt)")

    pooled = sum(report.confusion.values())
    print("\nVERWARRING gepoold (rijen EWMA, kolommen GARCH, VolState-volgorde)")
    print("           " + "".join(f"{s.name:>9}" for s in VolState) + "   totaal")
    for state in VolState:
        row = pooled[int(state)]
        print(f"  {state.name:<9}" + "".join(f"{int(v):>9}" for v in row)
              + f"{int(row.sum()):>9}")
    print("  totaal   " + "".join(f"{int(v):>9}" for v in pooled.sum(axis=0)))

    # Dezelfde matrices per symbool -- het VELD, uitgeschreven. De gepoolde
    # tabel middelt over zes namen die het onderling oneens zijn, en in stap 6
    # was juist dat de bevinding. De laatste kolom is de spoor/totaal-lezing
    # van dezelfde matrix, niet een tweede meting.
    print("\nVERWARRING per symbool (EWMA-rij -> GARCH-kolom, L/N/H)")
    print("  symbool  " + "".join(f"{x.name[0] + y.name[0]:>6}"
                                  for x in VolState for y in VolState) + "     ruw")
    for symbol, matrix in report.confusion.items():
        total = max(int(matrix.sum()), 1)
        print(f"  {symbol:<9}" + "".join(f"{int(v):>6}" for v in matrix.reshape(-1))
              + f"{int(np.trace(matrix)) / total:>8.3f}")

    print(f"\nOVEREENSTEMMING  ruw {report.fraction_identical:.4f} "
          f"({report.n_identical}/{report.n_cells_compared})"
          f"   Cohens kappa {report.cohen_kappa:.4f}")
    print(f"BESLUITWIJZIGING poort {sorted(s.name for s in report.flat_states)}"
          f"  {report.n_decision_changes}/{report.n_bars_compared} = "
          f"{report.fraction_bars_with_decision_change:.4f}")
    print(f"BOVENGRENS       elke toestandswissel  "
          f"{report.n_any_state_changes}/{report.n_bars_compared} = "
          f"{report.fraction_bars_with_any_state_change:.4f}")
    print(f"\nBESLISREGEL 8.5  ruw >= {MIN_IDENTICAL:.0%}: "
          f"{report.fraction_identical >= MIN_IDENTICAL}"
          f"   EN besluitwijziging < {MAX_DECISION_CHANGE:.0%}: "
          f"{report.fraction_bars_with_decision_change < MAX_DECISION_CHANGE}")


def main() -> int:
    cfg, state_cfg = load_baseline_configs(ROOT), regime_config().state
    validation = load_config(ROOT / "conf/validation/default.yaml", ValidationConfig)
    universe = load_phase6_universe(ROOT, cfg,
        build_cross_sectional_momentum(cfg["alpha"]), git_sha=current_git_sha())

    returns = development_slice(universe.log_returns, lock_path=LOCK)
    # HETZELFDE vergelijkingsvenster. `sigma_annual` draagt een bar meer dan de
    # log-returns (de eerste, die geen return heeft) en het GARCH-paneel leeft
    # op de returnindex; die ene bar stilzwijgend laten staan zou de twee
    # toewijzingen op verschillende vensters zetten.
    sigma_ewma = development_slice(
        universe.sigma_annual, lock_path=LOCK).loc[returns.index]
    # De foldstructuur wordt NIET opnieuw bedacht (R-3): dit is de constructie
    # van `apps/run_vol_competition.py`, met dezelfde `conf/validation/`-waarden
    # en dezelfde purge/embargo als de H1-campagne.
    sigma_garch, convergence = _garch_sigma(returns, WalkForwardCV(
        train_size=validation.train_bars, test_size=validation.test_bars,
        step=validation.test_bars, mode="rolling",
        min_train=validation.train_bars, embargo_bars=validation.embargo_bars))

    # Dezelfde bevroren kwantielREGEL op beide panelen (ruling P35): identieke
    # `low_q`, `high_q`, `min_periods` en `lag`, elk model tegen zijn eigen
    # verdeling. Een niveaudrempel van het ene model op het andere leggen zou
    # een schaalverschil als toestandsverschil rapporteren.
    rule = {"low_q": float(state_cfg.low_q), "high_q": float(state_cfg.high_q),
            "min_periods": int(state_cfg.min_periods), "lag": int(state_cfg.lag)}
    report = agreement(
        assign_by_variance(
            sigma_ewma, source=f"ewma_lambda_{cfg['vol'].ewma_lambda}", **rule),
        assign_by_variance(
            sigma_garch, source=f"{SPEC.label}_wf_h{HORIZON}", **rule))

    print(f"git_sha {current_git_sha()}  trials 0  selecteert niets")
    print(f"venster {returns.index[0].date()} .. {returns.index[-1].date()}  "
          f"regel {rule}")
    _print(report, convergence, sigma_garch, len(returns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
