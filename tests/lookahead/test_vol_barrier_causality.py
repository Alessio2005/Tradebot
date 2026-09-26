"""De triple-barrier labels van Phase 6 gebruiken geen informatie van vóór t+2.

Deliverable 19 eist dat de labeling de `shift(2)`-conventie respecteert. Dat is
geen eigenschap die je kunt aflezen uit de code — een off-by-one in de
monitoringlus ziet er identiek uit en wordt pas maanden later zichtbaar als een
AUC die te mooi is. Deze suite MEET hem in plaats van hem te lezen.

DE METHODE
----------
Een label is causaal wanneer het niet verandert als je de bars WIJZIGT die het
niet had mogen zien. De tests hieronder muteren precies één bar tegelijk:

    bar t      mag het label beïnvloeden (sigma, side, en de close voor niets
               anders dan het event zelf)
    bar t+1    mag de ENTRYPRIJS beïnvloeden en verder niets
    bar t+2..  mogen de uitkomst beïnvloeden

De scherpe grens is bar `t+1`: als het muteren van zijn HIGH of LOW het label
verandert, monitort de lus de entry-bar mee en is de conventie feitelijk
`shift(1)`.
"""
from __future__ import annotations

import numpy as np
import pytest

from tradebot.labeling.vol_barriers import label_triple_barrier
from tradebot.schemas.config import LabelingConfig
from tradebot.utils.failfast import CausalityViolationError

pytestmark = pytest.mark.lookahead

CFG = LabelingConfig(profit_target_sigma=2.0, stop_loss_sigma=2.0,
                     horizon_bars=10, entry_lag_bars=1, min_sigma_obs=60)


def _series(n: int = 200, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, n)))
    wiggle = np.abs(rng.normal(0.0, 0.01, n))
    return {
        "close": close,
        "high": close * (1.0 + wiggle),
        "low": close * (1.0 - wiggle),
        "sigma": np.full(n, 0.02),
        "side": np.ones(n),
    }


def _labels(data: dict[str, np.ndarray]):
    return label_triple_barrier(data["high"], data["low"], data["close"],
                                data["sigma"], data["side"], CFG)


def test_mutating_the_entry_bar_high_low_does_not_change_any_label() -> None:
    """De scherpe grens. Bar t+1 is de FILL-bar en wordt niet gemonitord.

    Zou de lus op `entry_idx` beginnen in plaats van op `entry_idx + 1`, dan
    verandert een extreme high of low op die bar de uitkomst van het event dat
    er direct aan voorafgaat — en dat is de `shift(1)`-aanname.
    """
    base = _series()
    labels_before = _labels(base)

    for event_t in (20, 50, 120):
        mutated = {k: v.copy() for k, v in base.items()}
        entry = event_t + CFG.entry_lag_bars
        # Een high die de profit target ruim overschrijdt en een low die de
        # stop loss ruim onderschrijdt, tegelijk, op precies de fill-bar.
        mutated["high"][entry] = base["close"][entry] * 5.0
        mutated["low"][entry] = base["close"][entry] * 0.2
        labels_after = _labels(mutated)

        row = np.searchsorted(labels_before.event_idx, event_t)
        assert labels_before.event_idx[row] == event_t
        assert labels_after.barrier_outcome[row] == labels_before.barrier_outcome[row], (
            f"het label van event {event_t} verandert door de high/low van zijn "
            f"eigen fill-bar {entry}; de monitoringlus begint een bar te vroeg "
            "en de conventie is feitelijk shift(1)"
        )


def test_mutating_the_first_monitored_bar_DOES_change_the_label() -> None:
    """Negatieve controle (§0.10): de test kan rood worden.

    Zonder deze helft bewijst de vorige test niets — een labelfunctie die naar
    GEEN ENKELE bar kijkt, zou hem ook doorstaan. Hier wordt aangetoond dat bar
    `t+2` de uitkomst wél bepaalt, zodat de grens scherp is en niet leeg.
    """
    base = _series()
    labels_before = _labels(base)

    changed = 0
    for event_t in (20, 50, 120):
        mutated = {k: v.copy() for k, v in base.items()}
        first_monitored = event_t + CFG.entry_lag_bars + 1
        mutated["high"][first_monitored] = base["close"][first_monitored] * 5.0
        labels_after = _labels(mutated)
        row = np.searchsorted(labels_before.event_idx, event_t)
        if labels_after.barrier_outcome[row] != labels_before.barrier_outcome[row]:
            changed += 1
    assert changed > 0, (
        "geen enkel label reageert op de eerste gemonitorde bar; de "
        "monitoringlus doet niets en de causaliteitstest hierboven is leeg"
    )


def test_truncating_the_series_after_the_barrier_window_changes_nothing() -> None:
    """Een label mag niet afhangen van bars ná zijn eigen verticale barrière."""
    base = _series(n=200)
    full = _labels(base)

    cutoff = 120
    truncated = {k: v[:cutoff].copy() for k, v in base.items()}
    partial = _labels(truncated)

    # Elk event dat in de afgekapte reeks nog een volledig venster heeft, moet
    # exact hetzelfde label krijgen.
    for i, t in enumerate(partial.event_idx):
        j = int(np.searchsorted(full.event_idx, t))
        assert full.event_idx[j] == t
        assert partial.barrier_outcome[i] == full.barrier_outcome[j], (
            f"event {t} krijgt een ander label wanneer de reeks na bar {cutoff} "
            "wordt afgekapt; het label kijkt voorbij zijn eigen venster"
        )
        assert partial.exit_idx[i] == full.exit_idx[j]


def test_no_event_can_be_labelled_without_a_complete_window() -> None:
    """Een afgekapt venster is systematisch naar de verticale barrière vertekend."""
    base = _series(n=60)
    labels = _labels(base)
    last_allowed = 60 - 1 - CFG.entry_lag_bars - CFG.horizon_bars
    assert int(labels.event_idx.max()) <= last_allowed
    assert int(labels.exit_idx.max()) <= 59


def test_entry_lag_zero_is_rejected_by_the_schema_not_by_convention() -> None:
    """`entry_lag_bars = 0` bestaat niet als geldige configuratie."""
    with pytest.raises(Exception) as exc:
        LabelingConfig(entry_lag_bars=0)
    assert "entry_lag_bars" in str(exc.value)


def test_a_hand_built_config_with_lag_zero_still_crashes_the_labeler() -> None:
    """Ook wie het schema omzeilt, komt er niet doorheen.

    `model_construct` slaat de validatie over — precies wat iemand doet die
    'even snel' wil testen. De labeler crasht dan alsnog, omdat een conventie
    die je per aanroep kunt uitzetten geen conventie is.
    """
    sneaky = LabelingConfig.model_construct(
        profit_target_sigma=2.0, stop_loss_sigma=2.0, horizon_bars=10,
        entry_lag_bars=0, min_sigma_obs=60,
    )
    base = _series()
    with pytest.raises(CausalityViolationError, match="entry_lag_bars"):
        label_triple_barrier(base["high"], base["low"], base["close"],
                             base["sigma"], base["side"], sneaky)


def test_both_barriers_in_one_bar_resolves_to_the_stop_loss() -> None:
    """Zonder intrabar-data is de volgorde onbekend; de bias moet conservatief.

    Wie hier de profit target kiest, bouwt een optimistische bias in het label
    die later als modelkwaliteit terugkomt.
    """
    n = 40
    close = np.full(n, 100.0)
    high = np.full(n, 100.0)
    low = np.full(n, 100.0)
    sigma = np.full(n, 0.02)
    side = np.zeros(n)
    side[5] = 1.0
    # Bar t+2 raakt beide barrières: +10 % en -10 % tegen 2 sigma = 4 %.
    high[7] = 110.0
    low[7] = 90.0

    labels = label_triple_barrier(high, low, close, sigma, side, CFG)
    assert len(labels) == 1
    assert labels.barrier_outcome[0] == -1
    assert labels.exit_idx[0] == 7
