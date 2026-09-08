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
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradebot.backtest.metrics import deflated_sharpe
from tradebot.registry.ledger_reset import (
    ResetAlreadyExists,
    active_trial_count,
    freeze_reset,
)
from tradebot.utils.failfast import DataContractError

LEDGER = Path("artefacts/governance/hypothesis_ledger.json")


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


@pytest.mark.parametrize(
    ("n_obs", "label"),
    [
        (1743, "W_FULL per ruling T1-B"),
        (1615, "legacy purged-WF OOS mask, continuiteitscontrole"),
    ],
)
def test_the_reset_does_not_make_the_gate_permissive(n_obs: int, label: str) -> None:
    """De negatieve controle op de reset zelf, over beide samples (ruling P17).

    Bij M_new = 25 is de DSR-eis 1,67 (n_obs=1743) resp. 1,73 (n_obs=1615)
    geannualiseerd. Het beste dat deze repository ooit heeft gemeten is 0,156.
    Een reset die 0,156 zou doorlaten, zou geen reset zijn maar een
    uitschakeling -- op geen van beide samples.
    """
    per_bar_best_measured = 0.156 / (365 ** 0.5)
    result = deflated_sharpe(
        per_bar_best_measured,
        n_obs=n_obs,
        n_trials=25,
        sr_variance=1.0 / n_obs,
        skew=0.0,
        kurtosis=3.0,
        bars_per_year=365,
    )
    assert result.dsr < 0.95, label


def test_m_new_must_be_positive(tmp_path: Path) -> None:
    """R1. Een programma zonder trials heeft geen kandidaten."""
    with pytest.raises(DataContractError):
        freeze_reset(
            m_new=0, rationale="r", git_sha="deadbee",
            out=tmp_path / "ledger_reset.json",
        )
