"""H1 — de QLIKE-competitie draaien. Phase 6 stap 7, deliverable 13.

De wetenschap staat in `validation/vol_campaign.py`, de opmaak in
`reporting/phase6_vol_competition.py`; deze app doet uitsluitend argumenten en
artefacten -- er staat geen enkele meting in.

R-6 stelt een grens van 80 regels en dit bestand telt er 129: negen docstring,
achtentwintig imports en twaalf regels toelichting, met ~70 regels bedrading.
Dat wordt hier niet weggeschreven als "past wel": het staat als meting in DI-4,
dat de regel repo-breed volgt.

    python apps/run_vol_competition.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.alpha.momentum import build_cross_sectional_momentum
from tradebot.backtest.baseline_report import load_baseline_configs
from tradebot.cv.walk_forward import WalkForwardCV
from tradebot.data.phase6_universe import load_phase6_universe
from tradebot.features.registry import current_git_sha
from tradebot.reporting.phase6_vol_competition import (
    build_h1_payload,
    render_competition_report,
)
from tradebot.schemas.config import (
    AdequacyConfig,
    ValidationConfig,
    econometrics_config,
    load_config,
)
from tradebot.validation.adequacy_report import measure_fold_geometry
from tradebot.validation.vol_campaign import build_proxy_panel, run_campaign
from tradebot.validation.vol_competition import EXPECTED_EFFECT_SD

#: Vier PER-BAR proxies, de primaire vooraan. `yang_zhang` doet niet mee: die is
#: een venster-estimator en zou informatie uit naburige bars in de realisatie
#: van deze bar brengen. Waarom Rogers-Satchell primair is, staat in §2 van het
#: rapport; de keuze ligt hier vast vóór de run en niet erna.
PROXIES = ("rogers_satchell", "squared_return", "parkinson", "garman_klass")
GOV = "artefacts/governance"
PREREG = f"{GOV}/preregistration_cef1a3b9a6811d7bde1afc92a2a9503f.json"

#: De ledger-entry waarin de 48 trials bij het bevriezen zijn geboekt. Deze run
#: voert ze uit; hij boekt ze niet opnieuw.
LEDGER_UNIT = "phase6_h1_garch_vs_ewma"


def _load(path: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=f"{GOV}/phase6_h1_competition.json")
    ap.add_argument("--report-out", default="reports/GARCH_VS_EWMA_COMPETITION.md")
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--control-replicates", type=int, default=200)
    args = ap.parse_args(argv)

    cfg = load_baseline_configs(ROOT)
    adequacy = load_config(ROOT / "conf/model/adequacy.yaml", AdequacyConfig)
    validation = load_config(ROOT / "conf/validation/default.yaml", ValidationConfig)
    git_sha = current_git_sha()
    universe = load_phase6_universe(
        ROOT, cfg, build_cross_sectional_momentum(cfg["alpha"]), git_sha=git_sha)

    returns = universe.log_returns
    proxies, proxy_records = build_proxy_panel(
        ohlc=universe.ohlc, index=returns.index, names=PROXIES)
    econometrics = _load(f"{GOV}/phase6_econometrics.json")
    result = run_campaign(
        returns=returns, proxies=proxies,
        arch_p_values={
            symbol: record["log_return"]["tests"]["engle_arch"]["p_value"]
            for symbol, record in econometrics["symbols"].items()},
        cv=WalkForwardCV(
            train_size=validation.train_bars, test_size=validation.test_bars,
            step=validation.test_bars, mode="rolling",
            min_train=validation.train_bars,
            embargo_bars=validation.embargo_bars),
        primary_proxy=PROXIES[0], horizons=(1, 5), seed=args.seed,
        n_control_replicates=args.control_replicates, adequacy=adequacy)

    # M IS AL GEBOEKT. De 48 trials van H1 staan sinds het bevriezen van de
    # pre-registratie in de ledger (wave 30, `result: interim`) -- terecht: wie
    # een parameterruimte vastlegt, heeft die kansen genomen. Deze run voert ze
    # uit en boekt ze dus NIET opnieuw; het oordeel gaat als amendement terug.
    ledger = _load(f"{GOV}/hypothesis_ledger.json")
    total = ledger["seed_total"] + sum(e["n_trials"] for e in ledger["entries"])
    booked = next(e for e in ledger["entries"] if e["unit"] == LEDGER_UNIT)
    if result.n_trials > booked["n_trials"]:
        raise SystemExit(
            f"De campagne fitte {result.n_trials} combinaties terwijl er "
            f"{booked['n_trials']} zijn gepre-registreerd. Meer zoeken dan "
            f"vooraf geboekt maakt elke DSR erna te gunstig; dat vraagt een "
            f"nieuwe pre-registratie en niet het oprekken van deze.")
    payload = build_h1_payload(
        git_sha=git_sha, universe=universe.as_record(),
        fold_geometry=measure_fold_geometry(len(returns), validation).as_record(),
        embargo_bars=validation.embargo_bars, alpha=econometrics_config().alpha,
        planned_trials=result.n_planned_trials,
        expected_effect_sd=EXPECTED_EFFECT_SD, preregistration=_load(PREREG),
        adequacy_artefact=_load(f"{GOV}/phase6_data_adequacy.json"),
        econometrics_artefact=econometrics, proxy_records=proxy_records,
        campaign=result.as_record(), ledger_before=total, ledger_after=total,
        trials_booked_at_freeze=int(booked["n_trials"]),
        ledger_unit=LEDGER_UNIT, ledger_config_hash=booked["config_hash"],
        artefact_path=args.out)

    for path, text in (
        (ROOT / args.out, json.dumps(payload, indent=2, default=str)),
        (ROOT / args.report_out, render_competition_report(payload)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    print(f"trials: {result.n_trials} gefit van {booked['n_trials']} bij de "
          f"pre-registratie geboekt; M blijft {total}")
    for status, count in sorted(result.by_status().items()):
        print(f"  {status:<10} {count}")
    print(f"artefact: {args.out}\nrapport:  {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
