# tests/unit/test_measurement_carries_policy_hash.py
"""Elke meting draagt de policy-hash van de configuratie die haar produceerde.

Fase 11, stap 2. AD-27, R-13, exit-criterium 2.

WAT HIER WORDT AFGEDWONGEN
==========================
Op 2026-09-12 verving `b18aeda` het propfirm-risicobudget door een
eigen-kapitaalmandaat: `sigma_target` 0,08 -> 0,20, `max_leverage` 1,5 -> 4,0,
`max_position_age_h` 48 -> 720. De `config_hash` sprong van `1b60cb664fbf9a2a`
naar `9961e1613bc907a5`. `b18aeda` is GEEN voorouder van `5dad1ab` (fase 10's
laatste meting); de twee sporen liepen parallel en zijn stilzwijgend
samengevoegd.

Precies EEN ding in deze repository merkte dat op:
`tests/regression/test_dust_breaks_relative_limits.py`, en alleen omdat iemand
daar een hash had vastgepind. Dat is toeval van dekking, geen mechanisme. Zes
artefacten met een risicobesluit meten vandaag nog onder het vervallen beleid,
en een daarvan (`phase10_h10_1.json`) noemt helemaal geen hash.

Deze test maakt er een poort van, met drie eisen:

1. een artefact dat een risicobesluit bevat en GEEN `config_hash` draagt, is een
   `DataContractError`;
2. een hash die niet in `risk_config_registry.json` staat, is een
   `DataContractError` -- ook als hij uit een configbron komt die git niet ziet
   (`conf/env/`, DI-32);
3. een artefact onder een VERVALLEN beleid moet dat expliciet zeggen
   (`superseded_by_policy`), anders is het een `DataContractError`.

WAAROM DE MARKERING DE OPVOLGER MOET NOEMEN
===========================================
`superseded_by_policy.superseded_by` moet gelijk zijn aan de HUIDIGE hash. Dat
maakt de markering opzettelijk bederfelijk: zodra er een vierde beleid wordt
geregistreerd, wordt elke markering rood en moet iemand per artefact opnieuw
beslissen of hij herkomst blijft of opnieuw wordt afgeleid. Een markering die
een volgende mandaatwijziging overleeft, is precies het stempel dat deze fase
heeft veroorzaakt.

DE NEGATIEVE CONTROLES
======================
`TestTheNegativeControlOnTheRealLadder` voert het ECHTE ladderartefact met een
verkeerde hash op en eist dat de poort rood wordt. Zonder die controle bewijst
een groene suite alleen dat de poort niets doet.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.registry.policy_carry import (
    SUPERSEDED_KEY,
    carries_risk_decision,
    current_policy_hash,
    declared_policy_hashes,
    latest_registered_policy,
    registered_policy_hashes,
    require_policy_carry,
    require_registered_policy,
)
from tradebot.risk.engine import risk_config_hash
from tradebot.schemas.config import RiskConfig, load_config
from tradebot.utils.failfast import DataContractError

CONF = ROOT / "conf"
LADDER = ROOT / "artefacts" / "baseline" / "phase5_revaluation.json"

#: Het vervallen beleid waaronder fase 5, 6, 9 en 10 hebben gemeten.
LAPSED = "1b60cb664fbf9a2a"
#: Een hash die nooit is geregistreerd. 16 hex, zelfde vorm, andere inhoud.
NEVER_REGISTERED = "0123456789abcdef"


@pytest.fixture(scope="module")
def registered() -> frozenset[str]:
    return registered_policy_hashes()


@pytest.fixture(scope="module")
def current() -> str:
    return current_policy_hash()


@pytest.fixture(scope="module")
def ladder() -> dict[str, Any]:
    return json.loads(LADDER.read_text(encoding="utf-8"))


def _decision(**extra: Any) -> dict[str, Any]:
    """Een minimaal document dat een risicobesluit BEVAT.

    `n_sovereign_halted` is de haltteller van `backtest/engine.py`: hij bestaat
    alleen als de soevereine laag daadwerkelijk heeft gedraaid.
    """
    return {"rows": [{"n_sovereign_halted": 1566, "n_sovereign_clipped": 1742}],
            **extra}


def _marking(measured_under: list[str], superseded_by: str,
             disposition: str = "PROVENANCE_ONLY",
             **extra: Any) -> dict[str, Any]:
    return {SUPERSEDED_KEY: {
        "measured_under": measured_under,
        "superseded_by": superseded_by,
        "disposition": disposition,
        "hash_source": "declared",
        "decided_in": "fase 11, stap 2.3",
        "reason": "Herkomst; niet opnieuw afgeleid.",
        **extra,
    }}


class TestTheGateRefusesAMeasurementWithoutItsPolicy:
    def test_a_decision_without_any_hash_is_a_contract_error(
        self, registered: frozenset[str], current: str
    ) -> None:
        """Dit is `phase10_h10_1.json` zoals fase 10 hem achterliet."""
        with pytest.raises(DataContractError, match="config_hash"):
            require_policy_carry(_decision(), source="synthetisch",
                                 registered=registered, current=current)

    def test_a_hash_that_was_never_registered_is_a_contract_error(
        self, registered: frozenset[str], current: str
    ) -> None:
        doc = _decision(risk_policy_hash=NEVER_REGISTERED)
        with pytest.raises(DataContractError, match="geregistreerd"):
            require_policy_carry(doc, source="synthetisch",
                                 registered=registered, current=current)

    def test_the_current_policy_passes(
        self, registered: frozenset[str], current: str
    ) -> None:
        doc = _decision(risk_policy_hash=current)
        assert require_policy_carry(doc, source="synthetisch",
                                    registered=registered,
                                    current=current) == frozenset({current})

    def test_a_lapsed_policy_without_a_marking_is_a_contract_error(
        self, registered: frozenset[str], current: str
    ) -> None:
        doc = _decision(risk_policy_hash=LAPSED)
        with pytest.raises(DataContractError, match=r"(?i)vervallen"):
            require_policy_carry(doc, source="synthetisch",
                                 registered=registered, current=current)

    def test_a_lapsed_policy_with_a_marking_passes(
        self, registered: frozenset[str], current: str
    ) -> None:
        doc = _decision(risk_policy_hash=LAPSED, **_marking([LAPSED], current))
        assert require_policy_carry(doc, source="synthetisch",
                                    registered=registered, current=current)

    def test_a_marking_that_names_the_wrong_successor_is_a_contract_error(
        self, registered: frozenset[str], current: str
    ) -> None:
        """Een markering die een VOLGENDE mandaatwijziging overleeft, is een
        stempel. Zij moet de nu geldende opvolger noemen."""
        doc = _decision(risk_policy_hash=LAPSED, **_marking([LAPSED], LAPSED))
        with pytest.raises(DataContractError, match="opvolger"):
            require_policy_carry(doc, source="synthetisch",
                                 registered=registered, current=current)

    def test_a_marking_that_covers_the_wrong_hash_is_a_contract_error(
        self, registered: frozenset[str], current: str
    ) -> None:
        doc = _decision(risk_policy_hash=LAPSED,
                        **_marking(["47821e47fe2cec30"], current))
        with pytest.raises(DataContractError, match="markering"):
            require_policy_carry(doc, source="synthetisch",
                                 registered=registered, current=current)

    def test_an_unknown_disposition_is_a_contract_error(
        self, registered: frozenset[str], current: str
    ) -> None:
        doc = _decision(risk_policy_hash=LAPSED,
                        **_marking([LAPSED], current, disposition="LATER"))
        with pytest.raises(DataContractError, match="disposition"):
            require_policy_carry(doc, source="synthetisch",
                                 registered=registered, current=current)

    def test_a_marking_may_supply_an_attribution_the_run_never_wrote(
        self, registered: frozenset[str], current: str
    ) -> None:
        """`phase10_h10_1.json` telt 1.566 halts en noemt geen enkel beleid.
        De hash is daar achteraf uit het eigen `git_sha` afgeleid; dat mag,
        mits de afleiding erbij staat en als AFGELEID is gemarkeerd."""
        doc = _decision(**_marking(
            [LAPSED], current, hash_source="derived_from_git_sha",
            derivation="git show <git_sha>:conf/risk/default.yaml -> hash"))
        assert require_policy_carry(doc, source="synthetisch",
                                    registered=registered,
                                    current=current) == frozenset({LAPSED})

    def test_a_derived_attribution_without_its_derivation_is_a_contract_error(
        self, registered: frozenset[str], current: str
    ) -> None:
        """Een reconstructie die niet na te rekenen is, is een bewering."""
        doc = _decision(**_marking([LAPSED], current,
                                   hash_source="derived_from_git_sha"))
        with pytest.raises(DataContractError, match="AFGELEIDE"):
            require_policy_carry(doc, source="synthetisch",
                                 registered=registered, current=current)

    def test_a_marking_without_a_hash_source_is_a_contract_error(
        self, registered: frozenset[str], current: str
    ) -> None:
        mark = _marking([LAPSED], current)
        del mark[SUPERSEDED_KEY]["hash_source"]
        with pytest.raises(DataContractError, match="hash_source"):
            require_policy_carry(_decision(risk_policy_hash=LAPSED, **mark),
                                 source="synthetisch",
                                 registered=registered, current=current)

    def test_a_sixteen_character_label_is_not_a_hash(self) -> None:
        """`hypothesis_ledger.json` draagt `config_hash: "h10_3_lapsed_no_"` --
        zestien tekens, geen hex. Een lengtetoets alleen telt dat als hash, en
        dan meldt de poort "nooit geregistreerd" waar "geen hash" de waarheid
        is. De foutmelding moet het juiste defect noemen."""
        assert declared_policy_hashes(
            {"config_hash": "h10_3_lapsed_no_"}) == frozenset()
        assert declared_policy_hashes(
            {"config_hash": LAPSED}) == frozenset({LAPSED})

    def test_a_document_that_decides_nothing_needs_no_hash(
        self, registered: frozenset[str], current: str
    ) -> None:
        """De poort eist geen hash van een document dat niets beslist. Zij
        eist hem van een document dat een risicobesluit DRAAGT -- anders wordt
        elke preregistratie en elke datacatalogus onterecht rood."""
        assert require_policy_carry({"universe": ["BTCUSDT"], "n_bars": 1743},
                                    source="synthetisch",
                                    registered=registered,
                                    current=current) == frozenset()


class TestTheNegativeControlOnTheRealLadder:
    """De poort moet aantoonbaar rood kunnen worden op een ECHT artefact."""

    def test_the_ladder_carries_a_risk_decision(self, ladder: dict) -> None:
        assert carries_risk_decision(ladder)
        assert declared_policy_hashes(ladder)

    def test_the_ladder_as_it_stands_passes_the_gate(
        self, ladder: dict, registered: frozenset[str], current: str
    ) -> None:
        require_policy_carry(ladder, source=str(LADDER),
                             registered=registered, current=current)

    def test_the_same_ladder_with_a_wrong_hash_is_rejected(
        self, ladder: dict, registered: frozenset[str], current: str
    ) -> None:
        """DE NEGATIEVE CONTROLE. Eén hash vervangen door een die nooit is
        geregistreerd, en de poort hoort rood te worden. Slaagt deze test
        niet, dan bewijst de groene sweep hieronder niets."""
        broken = copy.deepcopy(ladder)
        for row in broken["rows"]:
            if "risk_policy_hash" in row:
                row["risk_policy_hash"] = NEVER_REGISTERED
        with pytest.raises(DataContractError, match=NEVER_REGISTERED):
            require_policy_carry(broken, source="ladder-met-verkeerde-hash",
                                 registered=registered, current=current)

    def test_the_same_ladder_with_its_hash_removed_is_rejected(
        self, ladder: dict, registered: frozenset[str], current: str
    ) -> None:
        stripped = copy.deepcopy(ladder)
        stripped.pop(SUPERSEDED_KEY, None)
        for row in stripped["rows"]:
            row.pop("risk_policy_hash", None)
            row.pop("policy_hash", None)
        with pytest.raises(DataContractError, match="config_hash"):
            require_policy_carry(stripped, source="ladder-zonder-hash",
                                 registered=registered, current=current)


class TestTheGateSeesAPolicyThatGitDoesNotTrack:
    """DI-32. `conf/env/` is ongetrackt en door `.gitignore` afgeschermd; een
    poort die alleen naar `artefacts/` kijkt, mist dat kanaal volledig."""

    def test_the_shipped_policy_is_registered(
        self, registered: frozenset[str], current: str
    ) -> None:
        assert current in registered

    def test_a_threshold_that_nobody_registered_is_rejected(
        self, registered: frozenset[str]
    ) -> None:
        """De negatieve controle op het configkanaal: één drempel terugzetten
        op de propfirm-waarde levert een hash die nergens staat."""
        shipped = load_config(CONF / "risk/default.yaml", RiskConfig)
        overlay = dict(shipped.model_dump(mode="json"))
        overlay["sigma_target"] = 0.08
        rogue = risk_config_hash(RiskConfig(**overlay))
        assert rogue not in registered
        with pytest.raises(DataContractError, match=rogue):
            require_registered_policy(rogue, source="synthetische overlay",
                                      registered=registered)

    def test_the_defaults_order_keeps_env_profiles_from_setting_risk(self) -> None:
        """De enige reden dat `conf/env/prod.yaml`'s `risk:`-blok vandaag niet
        bindt, is de volgorde van `defaults:` in `conf/config.yaml` -- en die
        volgorde bewaakte niets. Nu wel."""
        defaults = yaml.safe_load(
            (CONF / "config.yaml").read_text(encoding="utf-8"))["defaults"]
        keys = [next(iter(d)) if isinstance(d, dict) else d for d in defaults]
        assert "risk" in keys and "env" in keys
        assert keys.index("risk") > keys.index("env"), (
            "`- risk: default` staat niet meer NA `- env:`. Een `risk:`-blok "
            "in een env-profiel overschrijft daarmee de enige bron van "
            "waarheid voor elke risicodrempel (RISK_CONTRACT §8, DI-32).")

    def test_a_local_env_profile_cannot_smuggle_in_an_unregistered_policy(
        self, registered: frozenset[str]
    ) -> None:
        """`conf/env/*.yaml` staat niet in git en ontbreekt op CI. Waar hij
        WEL staat, moet elk `risk:`-blok ofwel een geregistreerd beleid
        samenstellen, ofwel door de `defaults:`-volgorde onbindend zijn."""
        env_dir = CONF / "env"
        if not env_dir.is_dir():
            pytest.skip("conf/env/ bestaat hier niet (ongetrackt, DI-32)")
        shipped = load_config(CONF / "risk/default.yaml", RiskConfig)
        defaults = yaml.safe_load(
            (CONF / "config.yaml").read_text(encoding="utf-8"))["defaults"]
        keys = [next(iter(d)) if isinstance(d, dict) else d for d in defaults]
        env_binds = keys.index("env") > keys.index("risk")
        for profile in sorted(env_dir.glob("*.yaml")):
            block = (yaml.safe_load(profile.read_text(encoding="utf-8"))
                     or {}).get("risk")
            if not block:
                continue
            merged = dict(shipped.model_dump(mode="json")) | dict(block)
            composed = risk_config_hash(RiskConfig(**merged))
            assert composed in registered or not env_binds, (
                f"{profile.name} stelt beleid {composed} samen, dat nergens is "
                f"geregistreerd, en de `defaults:`-volgorde laat het binden.")


class TestTheRegisterIsAnIndependentSource:
    """Stap 2.4. `test_dust_breaks_relative_limits` pinde een hash-literal;
    die is vervangen door `latest_registered_policy()`. Dat is alleen een
    reparatie als de twee kanten van die assertie ECHT uit elkaar kunnen
    lopen -- anders is het `x == x` met extra stappen."""

    def test_the_register_and_the_shipped_config_agree_today(
        self, current: str
    ) -> None:
        """Loopt dit uiteen, dan draait er een risicobeleid dat niemand heeft
        vastgelegd. Dat is de onaangekondigde wijziging, en dan hoort de
        dust-regressietest rood te zijn."""
        assert latest_registered_policy() == current

    def test_a_fourth_policy_in_the_register_moves_only_one_side(
        self, tmp_path: Path, current: str
    ) -> None:
        """DE NEGATIEVE CONTROLE OP 2.4. Eén rij aan het register toevoegen
        verandert `latest_registered_policy()` en laat `conf/risk/default.yaml`
        ongemoeid. De twee kanten van de assertie zijn dus onafhankelijk."""
        doc = json.loads(
            (ROOT / "artefacts/governance/risk_config_registry.json")
            .read_text(encoding="utf-8"))
        doc["entries"].append({
            "config_hash": NEVER_REGISTERED, "git_sha": "deadbee",
            "registered_at": "2030-01-01T00:00:00+00:00",
            "config": {}, "audit_header": {}, "notes": "synthetisch",
        })
        forged = tmp_path / "risk_config_registry.json"
        forged.write_text(json.dumps(doc), encoding="utf-8")
        assert latest_registered_policy(forged) == NEVER_REGISTERED
        assert latest_registered_policy(forged) != current

    def test_a_register_whose_last_row_is_not_the_newest_is_rejected(
        self, tmp_path: Path
    ) -> None:
        """Append-only is de eigenschap waarop 'de laatste rij' rust."""
        forged = tmp_path / "risk_config_registry.json"
        forged.write_text(json.dumps({"entries": [
            {"config_hash": "a" * 16, "registered_at": "2030-01-01T00:00:00+00:00"},
            {"config_hash": "b" * 16, "registered_at": "2020-01-01T00:00:00+00:00"},
        ]}), encoding="utf-8")
        with pytest.raises(DataContractError, match="append-only"):
            latest_registered_policy(forged)


class TestEveryMeasurementInTheRepositoryCarriesItsPolicy:
    """De sweep. Elke JSON onder `artefacts/` en `reports/` die een
    risicobesluit draagt, gaat door dezelfde poort."""

    @staticmethod
    def _candidates() -> list[Path]:
        out: list[Path] = []
        for base in ("artefacts", "reports"):
            for path in sorted((ROOT / base).rglob("*.json")):
                try:
                    doc = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if carries_risk_decision(doc):
                    out.append(path)
        return out

    def test_the_sweep_actually_finds_something(self) -> None:
        """Een sweep over nul bestanden is groen en bewijst niets."""
        assert len(self._candidates()) >= 7

    def test_every_measurement_passes_the_gate(
        self, registered: frozenset[str], current: str
    ) -> None:
        offenders: list[str] = []
        for path in self._candidates():
            doc = json.loads(path.read_text(encoding="utf-8"))
            try:
                require_policy_carry(doc, source=str(path),
                                     registered=registered, current=current)
            except DataContractError as exc:
                offenders.append(f"{path.relative_to(ROOT).as_posix()}: {exc}")
        assert not offenders, "\n".join(offenders)
