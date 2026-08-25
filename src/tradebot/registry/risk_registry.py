# src/tradebot/registry/risk_registry.py
"""Append-only register van risicoconfiguraties, met `config_hash`.

Phase 4, stap 11: *"registreer de risicoconfiguratie in de ledger met
`config_hash`. Een risicoconfiguratie zonder hash is niet auditbaar."*

WAAROM NIET IN DE HYPOTHESE-LEDGER
----------------------------------
`registry/hypothesis_ledger.py` telt via `n_trials` het cumulatieve aantal
beproefde hypothesen, en dat getal is de `M` waarmee de Deflated Sharpe Ratio
deflateert (`backtest/metrics.py`). Elke rij die daar bij komt, verhoogt `M`.

Een risicoconfiguratie is GEEN hypothese. Zij voorspelt niets, wordt niet
getoetst tegen een nulhypothese en verdient geen deflatie. Haar daar toch
neerzetten zou de Phase 3-statistiek stilzwijgend verslechteren: de DSR van de
baseline zou dalen omdat iemand een limiet heeft opgeschreven. Dat is precies
het soort onopgemerkte koppeling dat deze fase opruimt.

Vandaar een eigen register met dezelfde eigenschappen die de ledger auditbaar
maken - append-only, atomair geschreven, gehasht - en zonder de eigenschap die
hier schadelijk is.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md secties 14, 20, 26.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ..utils.failfast import DataContractError, require
from ..utils.time import now_utc

__all__ = ["DEFAULT_RISK_REGISTRY_PATH", "RiskConfigRegistry"]

DEFAULT_RISK_REGISTRY_PATH = Path("artefacts/governance/risk_config_registry.json")


class RiskConfigRegistry:
    """Append-only JSON-register: welke risicoconfiguratie gold wanneer.

    Een hash die al geregistreerd staat, wordt NIET opnieuw toegevoegd en niet
    overschreven. Dat maakt het register idempotent: `run_stress.py` twee keer
    draaien levert geen tweede rij op, en een herschreven historie is per
    constructie zichtbaar als een hash-conflict.
    """

    def __init__(self, path: Path | str = DEFAULT_RISK_REGISTRY_PATH) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"schema": "risk_config_registry.v1", "entries": []}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        require(
            isinstance(raw, dict) and isinstance(raw.get("entries"), list),
            "Corrupt risicoconfiguratie-register.",
            DataContractError,
            path=str(self.path),
        )
        return raw

    def hashes(self) -> set[str]:
        return {str(e["config_hash"]) for e in self.load()["entries"]}

    def register(
        self,
        *,
        config_hash: str,
        git_sha: str,
        config: dict[str, Any],
        audit_header: dict[str, Any],
        notes: str = "",
    ) -> bool:
        """Leg deze configuratie vast. Geeft `False` als de hash er al stond.

        De VOLLEDIGE gevalideerde configuratie gaat mee, niet alleen de hash.
        Een hash zonder de inhoud erachter bewijst dat er iets is veranderd,
        maar niet wat - en dat is precies de vraag bij een post-mortem.
        """
        require(
            bool(config_hash) and bool(git_sha),
            "Een registratie zonder config_hash of git_sha is niet auditbaar.",
            DataContractError,
            config_hash=config_hash,
            git_sha=git_sha,
        )
        doc = self.load()
        existing = {str(e["config_hash"]) for e in doc["entries"]}
        if config_hash in existing:
            return False
        doc["entries"].append(
            {
                "config_hash": config_hash,
                "git_sha": git_sha,
                "registered_at": now_utc().isoformat(),
                "config": config,
                "audit_header": audit_header,
                "notes": notes,
            }
        )
        self._write_atomic(doc)
        return True

    def _write_atomic(self, doc: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")
        os.replace(tmp, self.path)
