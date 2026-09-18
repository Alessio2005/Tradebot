"""Bevries de monitoringdrempels vóór de 60-daagse klok. Stage D-3, D-5.

No-go 10 van de fase: *"Een monitoringdrempel is ná de start van de 60-daagse
klok vastgesteld."* Een drempel die tijdens de rit meebeweegt met de uitkomst is
geen drempel maar een verklaring achteraf.

Dit script schrijft de hash van `conf/monitoring/default.yaml` naar
`artefacts/governance/monitoring_config_hash.json`, samen met de git-sha en het
tijdstip. `tests/unit/test_monitoring_thresholds_are_frozen.py` weigert daarna
groen te worden zodra de gemeten hash afwijkt.

Een drempel wijzigen mag; hem wijzigen zonder opnieuw te bevriezen niet — en
opnieuw bevriezen tijdens een lopende klok herstart de klok (D-6).

    python apps/freeze_monitoring.py
    python apps/freeze_monitoring.py --check     # alleen verifieren, niets schrijven
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.features.registry import current_git_sha
from tradebot.schemas.config import monitoring_config
from tradebot.utils.hashing import hash_config

ARTEFACT = ROOT / "artefacts" / "governance" / "monitoring_config_hash.json"


def measured_hash() -> tuple[str, dict[str, float | int]]:
    """De hash van de GELADEN configuratie, niet van de bestandstekst.

    Bewust op het gevalideerde object: commentaar of witruimte in de YAML
    veranderen dan de hash niet, een drempel wel. Zo hoeft het bevriezen niet
    opnieuw bij elke redactionele wijziging, en kan het niet worden overgeslagen
    bij een inhoudelijke.
    """
    values = dict(monitoring_config().model_dump())
    return hash_config(values), values


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verifieer tegen het bevroren artefact zonder te schrijven")
    args = ap.parse_args(argv)

    digest, values = measured_hash()

    if args.check:
        if not ARTEFACT.is_file():
            print(f"Niet bevroren: {ARTEFACT} bestaat niet.")
            return 1
        frozen = json.loads(ARTEFACT.read_text(encoding="utf-8"))
        if frozen["config_hash"] != digest:
            print(f"AFWIJKING: bevroren {frozen['config_hash']}, "
                  f"gemeten {digest}.")
            return 1
        print(f"ongewijzigd sinds {frozen['frozen_utc']} ({digest})")
        return 0

    from datetime import datetime, timezone

    ARTEFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTEFACT.write_text(json.dumps({
        "config_hash": digest,
        "git_sha": current_git_sha(),
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "source": "conf/monitoring/default.yaml",
        "values": values,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"bevroren: {digest}\nartefact: {ARTEFACT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
