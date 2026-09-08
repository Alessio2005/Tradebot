"""De reset verlaagt een drempel. Deze tests bewijzen dat hij dat precies een
keer doet, dat hij de oude telling niet wist, en dat de verlaagde drempel nog
steeds weigert wat hij hoort te weigeren.

Ruling P17 parametriseert de negatieve controle over BEIDE samples die in deze
fase voorkomen, in plaats van de brief's enkele `n_obs=1615` te vervangen:
`docs/MEASUREMENT_CONTRACT.md` §2.5 scoopt N=1615 uitdrukkelijk tot het lezen
van legacy-artefacten ("geen drempel die een nieuwe meting mag kiezen"), en
ruling T1-B legt het venster van dit programma vast op W_FULL = 1743 bars. De
primaire case is daarom W_FULL; de legacy purged-WF OOS-mask (1615) blijft
uitsluitend als continuiteitscontrole staan. `sr_variance` volgt `n_obs` in
elk geval (`1.0 / n_obs`) -- `deflated_sharpe` raist anders op een variantie
die niet bij het `"normal"`-label hoort.

FIXRONDE 1 (ruling T3-B t/m T3-H) -- wat hier is toegevoegd en waarom
======================================================================
De oorspronkelijke `test_the_reset_does_not_make_the_gate_permissive`
hardcodede `n_trials=25` en opende het bevroren artefact nooit. Een reviewer
die `25` handmatig naar `2` veranderde, kreeg nog steeds 6 passed: de
assertie `dsr < 0.95` houdt stand voor ELKE geldige M, dus de test bewees
niets over de reset zelf (ruling T3-C). Hij is hieronder herbouwd met drie
onderdelen: (1) `m_new` komt uit het bevroren artefact, niet uit een letterlijke
25 in de test; (2) een M-gevoeligheidstoets die onafhankelijk (via bisectie,
geen tweede kopie van de DSR-formule) de drempel-Sharpe bij die `m_new`
herleidt en die tegen de gemeten waarde pint, zodat een verlaagde `m_new` de
test rood maakt; (3) de bestaande 0,156-smoketest blijft, maar nadrukkelijk
als smoketest op annualisatie en `deflated_sharpe`, niet als de reset-poort.

Daarnaast pint een nieuwe test het gecommitte artefact zelf (ruling T3-B): dat
het bestaat, `m_new == 25` draagt, en dat zijn eigen hash gelijk is aan de
`data_hash` op het AD-24-amendement in de ledger -- zonder die controle is het
bevriezen van het bestand een notitie die niemand leest, en kan iemand
`m_new` in het gecommitte bestand veranderen zonder dat de suite het merkt.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from scipy.optimize import brentq

from tradebot.backtest.metrics import deflated_sharpe
from tradebot.registry.ledger_reset import (
    ResetAlreadyExists,
    active_trial_count,
    freeze_reset,
)
from tradebot.utils.failfast import DataContractError
from tradebot.utils.hashing import DATA_HASH_LENGTH, hash_file

LEDGER = Path("artefacts/governance/hypothesis_ledger.json")
RESET = Path("artefacts/governance/ledger_reset.json")
BARS_PER_YEAR = 365.0


def _hurdle_sharpe(*, n_obs: int, n_trials: int) -> float:
    """De geannualiseerde Sharpe waarbij `deflated_sharpe` DSR=0,95 haalt.

    Onafhankelijk gevonden met bisectie (ruling T3-C): dit is GEEN tweede
    kopie van de DSR-formule. Een verlaagde `n_trials` verlaagt deze drempel,
    wat precies is wat een test op de reset moet kunnen opmerken.
    """
    def _dsr_minus_target(ann_sr: float) -> float:
        per_bar = ann_sr / (BARS_PER_YEAR ** 0.5)
        result = deflated_sharpe(
            per_bar, n_obs=n_obs, n_trials=n_trials,
            sr_variance=1.0 / n_obs, skew=0.0, kurtosis=3.0,
            bars_per_year=BARS_PER_YEAR,
        )
        return result.dsr - 0.95

    return brentq(_dsr_minus_target, 0.0, 10.0, xtol=1e-10)


def test_reset_is_frozen_and_carries_its_grounds(tmp_path: Path) -> None:
    out = tmp_path / "ledger_reset.json"
    record = freeze_reset(
        m_new=25,
        rationale="AD-24; nieuw programma op dagbars, kandidaatverzameling "
                  "disjunct van F1-F20",
        git_sha="deadbee",
        out=out,
    )
    assert record.m_new == 25
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["m_new"] == 25
    assert payload["git_sha"] == "deadbee"
    assert payload["rationale"]
    assert payload["frozen_utc"]


def test_a_second_reset_is_refused(tmp_path: Path) -> None:
    """R5. Twee resets maken M een parameter in plaats van een meting."""
    out = tmp_path / "ledger_reset.json"
    freeze_reset(m_new=25, rationale="eerste", git_sha="deadbee", out=out)
    with pytest.raises(ResetAlreadyExists):
        freeze_reset(m_new=4, rationale="tweede", git_sha="deadbee", out=out)


def test_the_old_ledger_stays_readable(tmp_path: Path) -> None:
    """R3. De reset archiveert; hij wist niet."""
    out = tmp_path / "ledger_reset.json"
    freeze_reset(m_new=25, rationale="r", git_sha="deadbee", out=out)
    count = active_trial_count(reset_path=out, ledger_path=LEDGER)
    assert count.total == 25
    assert count.archived_total == 2776


def test_active_trial_count_refuses_a_mismatched_archived_total(
    tmp_path: Path,
) -> None:
    """Ruling T3-D. `archived_total` in het artefact is een claim, geen bron.

    Wijkt de claim af van wat de ledger zelf optelt, dan is het artefact of de
    ledger veranderd sinds het bevriezen, en moet de meting weigeren in plaats
    van de claim te geloven.
    """
    out = tmp_path / "ledger_reset.json"
    freeze_reset(m_new=25, rationale="r", git_sha="deadbee", out=out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    payload["archived_total"] = payload["archived_total"] + 1
    out.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DataContractError):
        active_trial_count(reset_path=out, ledger_path=LEDGER)


def test_the_frozen_reset_artefact_is_pinned_and_linked_to_the_ledger() -> None:
    """Ruling T3-B. `git grep ledger_reset.json` gaf vóór deze test precies één
    hit: de schrijver zelf. Niets las het terug, dus R5's "een reset" werd
    door niets afgedwongen -- een operator kon `m_new` in het gecommitte
    bestand veranderen, of het verwijderen en opnieuw bevriezen, en de suite
    bleef groen. Deze test pint het bestand zelf, EN de cryptografische
    koppeling naar het AD-24-amendement (wave 31, `ad24_ledger_reset`) in de
    ledger, zodat die koppeling een hek is en geen decoratie.
    """
    assert RESET.is_file(), (
        f"{RESET} ontbreekt. R6: M_new moet bevroren zijn vóór de eerste fit."
    )
    payload = json.loads(RESET.read_text(encoding="utf-8"))
    assert payload["m_new"] == 25

    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    amendment = next(
        entry for entry in ledger["entries"]
        if entry.get("wave") == 31 and entry.get("unit") == "ad24_ledger_reset"
    )
    assert amendment["data_hash"] == hash_file(RESET, DATA_HASH_LENGTH), (
        "De data_hash op het AD-24-amendement wijkt af van de huidige hash "
        "van ledger_reset.json -- het bevroren bestand is gewijzigd sinds "
        "het amendement is geboekt."
    )


@pytest.mark.parametrize(
    ("n_obs", "expected_hurdle", "label"),
    [
        (1743, 1.6682757106971973, "W_FULL per ruling T1-B"),
        (1615, 1.7332625031264708, "legacy purged-WF OOS mask, continuiteitscontrole"),
    ],
)
def test_the_reset_does_not_make_the_gate_permissive(
    n_obs: int, expected_hurdle: float, label: str
) -> None:
    """De negatieve controle op de reset zelf (ruling T3-C), in drie delen.

    (1) `m_new` komt uit het BEVROREN artefact, niet uit een letterlijke 25 --
    anders test de test zichzelf en niet de reset.
    (2) M-gevoeligheidstoets: de geannualiseerde Sharpe waarbij
    `deflated_sharpe` DSR=0,95 haalt bij die bevroren `m_new`, onafhankelijk
    gevonden met bisectie, moet gelijk zijn aan de gemeten drempel. Verlaag
    `m_new` en dit wordt rood -- dat is het hele punt.
    (3) De bestaande smoketest op de beste ooit gemeten track (0,156): een
    echte sanity-check op annualisatie en `deflated_sharpe`, maar NIET de
    reset-poort zelf (dat is deel 2).
    """
    m_new = int(json.loads(RESET.read_text(encoding="utf-8"))["m_new"])
    assert m_new == 25

    hurdle = _hurdle_sharpe(n_obs=n_obs, n_trials=m_new)
    assert hurdle == pytest.approx(expected_hurdle, abs=5e-4), label

    per_bar_best_measured = 0.156 / (365 ** 0.5)
    result = deflated_sharpe(
        per_bar_best_measured,
        n_obs=n_obs,
        n_trials=m_new,
        sr_variance=1.0 / n_obs,
        skew=0.0,
        kurtosis=3.0,
        bars_per_year=365,
    )
    assert result.dsr < 0.95, label


@pytest.mark.parametrize("m_new", [0, 1])
def test_m_new_must_be_positive(m_new: int, tmp_path: Path) -> None:
    """R1 + ruling T3-E. M_new < 2 heeft geen expected maximum om tegen te
    corrigeren (Bailey-Lopez de Prado), en R5 maakt een bevroren fout
    onherstelbaar zonder een tweede reset -- dus de grens ligt bij 2, niet bij
    0. De brief's oorspronkelijke `m_new=0`-geval blijft staan; `m_new=1` is
    nieuw (ruling T3-E)."""
    with pytest.raises(DataContractError):
        freeze_reset(
            m_new=m_new, rationale="r", git_sha="deadbee",
            out=tmp_path / "ledger_reset.json",
        )
