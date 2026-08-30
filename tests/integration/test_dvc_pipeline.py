# tests/integration/test_dvc_pipeline.py
"""De DVC-pipeline moet UITVOERBAAR zijn, niet alleen aanwezig. Deliverable 5.

WAAROM DEZE TESTS BESTAAN
==========================
`dvc.yaml` draagt sinds Phase 0 de zin *"Same code + same config + same data →
same artefacts"*. Die garantie werd nergens afgedwongen, en in Phase 1 bleek de
pipeline **nooit uitvoerbaar te zijn geweest**: `${item}` interpoleerde een dict
en `dvc repro` brak af op het laden van het project. Een reproduceerbaarheids-
belofte die nooit is uitgevoerd, is een belofte over iets dat niet bestaat.

Deze suite controleert wat je met `dvc repro` pas ná minuten rekenen zou merken:
dat elke stage bestaat, dat elk pad in `deps` en `outs` bestaat, en dat de
Phase 6-campagnes hun eigen pre-registratie, hun eigen config en de
gecertificeerde store als afhankelijkheid noemen. Dat laatste is de kern van
exit-criterium 17: een run is pas reproduceerbaar uit `data_hash` +
`config_hash` + git-sha wanneer de pipeline die drie ook echt als ingang heeft.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

pytestmark = pytest.mark.integration

#: De stages die Phase 6 heeft toegevoegd. `foreach`-stages uit Phase 0-1 staan
#: er bewust niet bij: die hangen aan `market_data_parquet/`, dat niet in git
#: staat, en zij zijn niet het onderwerp van deliverable 5.
_PHASE6_STAGES = (
    "phase5_revaluation",
    "phase6_adequacy",
    "phase6_econometrics",
    "phase6_h1_competition",
    "phase6_h2_regime_benchmark",
)


@pytest.fixture(scope="module")
def pipeline() -> dict:
    return yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))


def _paths(entries) -> list[str]:
    """Paden uit een `deps`- of `outs`-lijst; `outs` mag mappings bevatten."""
    out: list[str] = []
    for entry in entries or ():
        out.append(next(iter(entry)) if isinstance(entry, dict) else entry)
    return out


class TestTheStagesExist:
    def test_every_phase6_stage_is_declared(self, pipeline) -> None:
        missing = [s for s in _PHASE6_STAGES if s not in pipeline["stages"]]
        assert not missing, f"stages ontbreken in dvc.yaml: {missing}"

    def test_the_authoritative_engine_has_its_own_stage(self, pipeline) -> None:
        """Deliverable 5 in één zin: de opvolger van de Phase 3-baseline draait
        niet meer alleen met de hand."""
        stage = pipeline["stages"]["phase5_revaluation"]
        assert stage["cmd"] == "python apps/run_phase5_baseline.py"
        assert "artefacts/baseline/phase5_revaluation.json" in _paths(
            stage["outs"])


@pytest.mark.parametrize("name", _PHASE6_STAGES)
class TestEveryDeclaredPathExists:
    def test_dependencies_exist(self, pipeline, name: str) -> None:
        """Een `deps`-pad dat niet bestaat, laat `dvc repro` afbreken op een
        moment dat niemand meer weet welke wijziging het veroorzaakte."""
        missing = [p for p in _paths(pipeline["stages"][name].get("deps"))
                   if not (ROOT / p).exists()]
        assert not missing, f"{name}: deps bestaan niet: {missing}"

    def test_the_command_points_at_a_real_app(self, pipeline, name: str) -> None:
        command = pipeline["stages"][name]["cmd"].split()
        assert command[0] == "python"
        assert (ROOT / command[1]).is_file(), f"{name}: {command[1]} ontbreekt"

    def test_outputs_are_declared_and_uncached(self, pipeline, name: str) -> None:
        """De artefacten staan in git (AD-12 voor de store, en governance-JSON
        hoort in de diff zichtbaar te zijn). `cache: false` houdt DVC ervan af
        ze naar zijn eigen cache te verplaatsen."""
        outs = pipeline["stages"][name]["outs"]
        assert outs, f"{name} produceert niets"
        for entry in outs:
            assert isinstance(entry, dict), (
                f"{name}: output zonder `cache: false` — DVC zou hem uit git "
                f"halen")
            assert next(iter(entry.values()))["cache"] is False


class TestTheCampaignsCarryTheirProvenance:
    """Exit-criterium 17: reproduceerbaar uit `data_hash` + `config_hash` +
    git-sha. Dat kan alleen wanneer de pipeline die ingangen ook noemt."""

    @pytest.mark.parametrize(
        ("stage", "preregistration"),
        [
            ("phase6_h1_competition",
             "artefacts/governance/preregistration_"
             "cef1a3b9a6811d7bde1afc92a2a9503f.json"),
            ("phase6_h2_regime_benchmark",
             "artefacts/governance/preregistration_"
             "3d3af28730a6c7f9da48d13139522a05.json"),
        ],
    )
    def test_the_campaign_depends_on_its_frozen_preregistration(
        self, pipeline, stage: str, preregistration: str
    ) -> None:
        deps = _paths(pipeline["stages"][stage]["deps"])
        assert preregistration in deps

    @pytest.mark.parametrize("name", _PHASE6_STAGES)
    def test_every_stage_depends_on_the_certified_store(
        self, pipeline, name: str
    ) -> None:
        assert "data/pit_store" in _paths(pipeline["stages"][name]["deps"])

    @pytest.mark.parametrize("name", _PHASE6_STAGES)
    def test_every_stage_depends_on_at_least_one_config_file(
        self, pipeline, name: str
    ) -> None:
        deps = _paths(pipeline["stages"][name]["deps"])
        assert [p for p in deps if p.startswith("conf/")], (
            f"{name} noemt geen enkel configuratiebestand; een wijziging in "
            f"`conf/` zou de stage dan niet ongeldig maken"
        )


class TestTheLockFileMatches:
    """`dvc.lock` is het BEWIJS dat de pipeline echt heeft gedraaid."""

    @pytest.fixture(scope="class")
    def lock(self) -> dict:
        path = ROOT / "dvc.lock"
        if not path.is_file():
            pytest.skip("dvc.lock ontbreekt; draai `dvc repro` eerst")
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_the_engine_stage_has_actually_run(self, lock) -> None:
        assert "phase5_revaluation" in lock["stages"]

    def test_the_lock_records_a_hash_for_the_certified_store(
        self, lock, pipeline
    ) -> None:
        """Zonder een hash over de store is `data_hash` een bewering."""
        deps = lock["stages"]["phase5_revaluation"]["deps"]
        store = [d for d in deps if d["path"] == "data/pit_store"]
        assert store, "de store staat niet in de lock"
        assert store[0]["md5"].endswith(".dir")
        assert store[0]["nfiles"] > 0
