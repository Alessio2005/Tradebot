# tests/unit/test_monitoring_thresholds_are_frozen.py
"""Monitoringdrempels staan vast vóór de klok. Stage D-3 / D-5, no-go 10.

*"Een drempel die achteraf wordt vastgesteld, is geen drempel."* De fase
verbiedt het bijstellen van een monitoringdrempel nadat de 60-daagse klok is
gestart — dat is de meest directe manier om een operationele periode groen te
laten eindigen zonder dat er iets is verbeterd.

De sluiting bestaat uit drie delen, en alleen samen werken ze:

1. de drempels staan in `conf/monitoring/default.yaml` en niet in code
   (`monitoring/drift.py` leest ze daar);
2. `apps/freeze_monitoring.py` legt hun hash vast in
   `artefacts/governance/monitoring_config_hash.json`;
3. deze test weigert groen te worden zodra de gemeten hash afwijkt.

Zonder (3) is (2) een bestand dat niemand leest.

Ref: fase-opdracht Stage D-3; exit-criterium D5; no-go 10.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.registry.hypothesis_ledger import hash_config
from tradebot.schemas.config import MonitoringConfig, monitoring_config

ARTEFACT = ROOT / "artefacts" / "governance" / "monitoring_config_hash.json"


class TestTheThresholdsAreFrozen:
    def test_the_freeze_artefact_exists(self) -> None:
        assert ARTEFACT.is_file(), (
            "De monitoringdrempels zijn niet bevroren. Draai "
            "`python apps/freeze_monitoring.py` VOOR de 60-daagse klok start.")

    def test_the_live_config_matches_the_frozen_hash(self) -> None:
        frozen = json.loads(ARTEFACT.read_text(encoding="utf-8"))
        measured = hash_config(dict(monitoring_config().model_dump()))
        assert measured == frozen["config_hash"], (
            f"De monitoringdrempels zijn gewijzigd sinds het bevriezen op "
            f"{frozen['frozen_utc']}: bevroren {frozen['config_hash']}, gemeten "
            f"{measured}. Dat mag — maar niet stilzwijgend. Bevries opnieuw en "
            f"herstart de 60-daagse klok (no-go 10 en 11).")

    def test_the_frozen_values_are_the_live_values(self) -> None:
        """Niet alleen de hash maar ook de getallen zelf, zodat een lezer van
        het artefact ziet waarop de klok rust."""
        frozen = json.loads(ARTEFACT.read_text(encoding="utf-8"))
        assert frozen["values"] == monitoring_config().model_dump()


class TestTheThresholdsAreNotInCodeAnymore:
    def test_drift_reads_its_thresholds_from_the_config(self) -> None:
        from tradebot.monitoring import drift

        cfg = monitoring_config()
        assert drift.PSI_MODERATE == cfg.psi_moderate
        assert drift.PSI_CRITICAL == cfg.psi_critical
        assert drift.REFERENCE_INTRA_PSI_LIMIT == cfg.reference_intra_psi_limit
        assert drift._MIN_SAMPLES == cfg.min_samples


class TestTheDetectorCanGoRed:
    """Zonder deze twee bewaakt de test hierboven niets."""

    def test_a_changed_threshold_changes_the_hash(self) -> None:
        base = monitoring_config()
        tweaked = base.model_copy(update={"psi_critical": 0.25})
        assert hash_config(dict(tweaked.model_dump())) != hash_config(
            dict(base.model_dump()))

    def test_the_schema_refuses_an_incoherent_pair(self) -> None:
        """psi_critical onder psi_moderate laat het alarm van stabiel naar
        kritiek springen zonder tussenzone."""
        with pytest.raises(ValueError, match="psi_critical"):
            MonitoringConfig(psi_moderate=0.30, psi_critical=0.20)
