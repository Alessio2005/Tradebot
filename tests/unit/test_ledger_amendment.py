"""Een oordeel is geen nieuwe zoektocht — de amendement-entry.

HET PROBLEEM, GEMETEN OP 2026-08-29
====================================
De 48 trials van H1 zijn bij het BEVRIEZEN van de pre-registratie al in `M`
geboekt (`wave 30, unit phase6_h1_garch_vs_ewma, n_trials=48, result=interim`).
Dat is precies goed: wie een parameterruimte vastlegt, heeft die kansen al
genomen, en `M` mag niet pas groeien als de uitkomst bevalt.

Maar het maakt het BOEKEN VAN DE UITKOMST tot een probleem. De ledger is
append-only, `total_n_hypotheses` telt `seed_total + sum(n_trials)`, en
`n_trials >= 1` is afgedwongen. Een tweede entry met dezelfde 48 trials zou
`M` van 2.776 naar 2.824 brengen voor onderzoek dat één keer is gedaan.

Beide fouten zijn even erg en gaan de andere kant op:

    onderteld  -> elke DSR erna is te GUNSTIG
    dubbelgeteld -> elke DSR erna is te STRENG, en het getal is even onwaar

Vandaar dit contract: een entry die een eerdere entry AMENDEERT draagt
`amends` en `n_trials = 0`. Hij verandert het oordeel en niet de telling.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradebot.registry.hypothesis_ledger import HypothesisLedger, LedgerEntry


def _ledger(tmp_path: Path, entries: list[dict] | None = None) -> HypothesisLedger:
    path = tmp_path / "ledger.json"
    path.write_text(
        json.dumps({"seed_total": 100, "entries": entries or []}),
        encoding="utf-8")
    return HypothesisLedger(path)


def _entry(**overrides: object) -> LedgerEntry:
    base = dict(
        wave=30, unit="phase6_h1_garch_vs_ewma", market="crypto",
        config_hash="cef1a3b9a6811d7b", n_trials=48, result="interim",
        git_sha="abc1234", data_hash="deadbeef",
        preregistration_id="cef1a3b9a6811d7bde1afc92a2a9503f",
    )
    base.update(overrides)
    return LedgerEntry(**base)  # type: ignore[arg-type]


class TestAnAmendmentCarriesNoTrials:
    def test_zero_trials_is_allowed_when_it_amends(self) -> None:
        entry = _entry(n_trials=0, result="archived", amends="cef1a3b9a6811d7b")
        assert entry.n_trials == 0
        assert entry.amends == "cef1a3b9a6811d7b"

    def test_zero_trials_without_an_amendment_is_still_refused(self) -> None:
        """De oude regel blijft staan waar hij hoort.

        Zonder `amends` is `n_trials = 0` een entry die beweert onderzoek te
        hebben gedaan zonder een kans te hebben genomen — dat is precies de
        boekhouding die `M` onbruikbaar maakt.
        """
        with pytest.raises(ValueError, match="n_trials"):
            _entry(n_trials=0)

    def test_an_amendment_may_not_smuggle_in_new_trials(self) -> None:
        """Een amendement dat wél trials meebrengt, is een nieuwe zoektocht met
        een etiket dat het tegenovergestelde zegt."""
        with pytest.raises(ValueError, match="amend"):
            _entry(n_trials=48, amends="cef1a3b9a6811d7b")


class TestTheLedgerCountsAnAmendmentAsZero:
    def test_the_total_does_not_move(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        before = ledger.append(_entry())
        after = ledger.append(
            _entry(n_trials=0, result="archived", amends="cef1a3b9a6811d7b"))
        assert before == 148
        assert after == 148

    def test_the_verdict_is_readable_from_the_ledger(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        ledger.append(_entry())
        ledger.append(_entry(
            n_trials=0, result="archived", amends="cef1a3b9a6811d7b",
            notes="UNPROVEN — de proxy-premisse is geschonden"))
        entries = ledger.entries()
        assert entries[-1]["result"] == "archived"
        assert entries[-1]["amends"] == entries[0]["config_hash"]
        assert "UNPROVEN" in entries[-1]["notes"]

    def test_an_amendment_of_an_unknown_entry_is_refused(
        self, tmp_path: Path,
    ) -> None:
        """Een amendement wijst naar iets dat bestaat, anders hangt het oordeel
        aan niets en is niet na te gaan wat er is geamendeerd."""
        ledger = _ledger(tmp_path)
        with pytest.raises(ValueError, match="Amendement"):
            ledger.append(_entry(
                n_trials=0, result="archived", amends="0000000000000000"))
