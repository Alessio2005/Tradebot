"""D-1 POORT 5 — LABEL-HORIZON EN PURGE.

    Het label van een event op `t` wordt bepaald door bars t/m `t + H`. Ligt
    `t + H` in de testperiode, dan traint het model op de uitkomst die het moet
    voorspellen.

Dit lek is anders dan de vier voorgaande: de FEATURES zijn hier volstrekt
causaal. Het is het LABEL dat vooruit kijkt, en dat hoort het ook te doen — een
label zonder toekomst is geen label. Het lek ontstaat pas bij de SPLITSING, en
het is daarom niet met een featuretest te vinden.

PURGING EN EMBARGO ZIJN TWEE DINGEN
===================================
`López de Prado (2018) §7.4` behandelt ze als twee stappen, en dat is niet
overdreven:

* **Purging** verwijdert train-events waarvan het label het testvenster in
  loopt. Zonder purging traint het model op de uitkomst zelf.
* **Embargo** verwijdert daarnaast een marge NA de testperiode uit de
  eerstvolgende trainperiode. Dat vangt de seriële correlatie die overblijft
  wanneer het label al is gepurged: features rond de grens blijven gecorreleerd
  met testuitkomsten, ook al overlapt het label niet meer.

Purging alleen laat de correlatie staan; embargo alleen laat het label lekken.

DE HORIZON IS NIET CONSTANT — EN DAT IS DE VALKUIL
==================================================
Bij triple-barrier labeling sluit een event op de eerste barrière die wordt
geraakt. De GEREALISEERDE horizon `exit_idx - event_idx` varieert dus per event;
alleen de VERTICALE barrière is een bovengrens. Wie purge't op de gemiddelde of
de typische horizon in plaats van op het maximum, laat precies de events door
die het langst open stonden — en dat zijn de events met het meeste
overlap-risico.

Deze poort meet de gerealiseerde horizon uit `BarrierLabels.exit_idx` in plaats
van hem uit de configuratie aan te nemen, en toetst de purge daartegen.

Ref: audit §17.1; `fase_2_research_falsification.md` deliverable 5;
`validation/walk_forward.py`.
"""
from __future__ import annotations

import numpy as np
import pytest

from tradebot.labeling.phase6_barriers import BarrierLabels, label_triple_barrier
from tradebot.schemas.config import LabelingConfig, ValidationConfig, load_config
from tradebot.utils.failfast import ConfigContractError, DataContractError
from tradebot.validation.walk_forward import (
    PurgedFold,
    purged_walk_forward,
    verify_no_overlap,
)

from .d1_harness import ROOT

pytestmark = pytest.mark.lookahead

HORIZON_BARS = 10
LABEL_CFG = LabelingConfig(
    profit_target_sigma=2.0, stop_loss_sigma=2.0,
    horizon_bars=HORIZON_BARS, entry_lag_bars=1, min_sigma_obs=60,
)
N_BARS = 2000


@pytest.fixture(scope="module")
def validation_cfg() -> ValidationConfig:
    return load_config(ROOT / "conf" / "validation" / "default.yaml",
                       ValidationConfig)


@pytest.fixture(scope="module")
def labels() -> BarrierLabels:
    """Echte triple-barrier labels op een synthetische maar realistische reeks."""
    rng = np.random.default_rng(20260829)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, N_BARS)))
    wiggle = np.abs(rng.normal(0.0, 0.01, N_BARS))
    return label_triple_barrier(
        high=close * (1.0 + wiggle),
        low=close * (1.0 - wiggle),
        close=close,
        sigma=np.full(N_BARS, 0.02),
        side=np.ones(N_BARS),
        cfg=LABEL_CFG,
    )


# --------------------------------------------------------------------------- #
# De horizon is een BOVENGRENS, geen constante
# --------------------------------------------------------------------------- #
class TestTheRealisedHorizon:
    def test_no_label_reaches_past_the_vertical_barrier(
        self, labels: BarrierLabels
    ) -> None:
        """De bovengrens waarop de purge mag rekenen.

        Zou één event verder reiken dan de verticale barrière, dan is elke purge
        die op `horizon_bars` is gebaseerd te kort, en lekt precies dat event.
        """
        assert len(labels) > 0, "geen events; de fixture meet niets"
        span = labels.exit_idx - labels.event_idx
        # entry_lag_bars = 1, dus de monitoring begint op t+2 en de verticale
        # barrière valt op t + 1 + horizon_bars.
        upper = LABEL_CFG.entry_lag_bars + HORIZON_BARS
        assert int(span.max()) <= upper, (
            f"een event liep {int(span.max())} bars door tegen een verticale "
            f"barrière op {upper}. De purge-horizon in conf/validation/ is dan "
            f"structureel te kort.")

    def test_the_horizon_actually_varies(self, labels: BarrierLabels) -> None:
        """Zonder variatie is deze poort een omslachtige constante-check.

        Als élk event op de verticale barrière sloot, zou purgen op een vaste H
        triviaal correct zijn en zou de valkuil uit de moduledocstring niet
        bestaan. De labels moeten dus aantoonbaar verschillende looptijden
        hebben.
        """
        span = labels.exit_idx - labels.event_idx
        assert int(span.min()) < int(span.max()), (
            "alle events sluiten op dezelfde bar; de gerealiseerde horizon "
            "varieert niet en deze poort meet minder dan hij beweert")


# --------------------------------------------------------------------------- #
# De purge doet aantoonbaar werk
# --------------------------------------------------------------------------- #
def _leaking_events(labels: BarrierLabels, train_end: int, test_start: int) -> int:
    """Aantal train-events waarvan het LABEL de testperiode in loopt."""
    in_train = labels.event_idx < train_end
    return int((labels.exit_idx[in_train] >= test_start).sum())


class TestTheUnpurgedSplitLeaks:
    """NEGATIEVE CONTROLE, en tegelijk de meting van de omvang.

    Een naïeve splitsing op een grens `k` — train `[0, k)`, test `[k, ...)` —
    laat elk event doorlekken dat binnen H bars vóór `k` begon. Dit is geen
    theoretisch risico maar een telbaar aantal.
    """

    def test_a_naive_boundary_split_leaks_a_countable_number_of_events(
        self, labels: BarrierLabels
    ) -> None:
        boundary = N_BARS // 2
        leaked = _leaking_events(labels, train_end=boundary, test_start=boundary)
        assert leaked > 0, (
            "de naïeve splitsing lekte geen enkel event. Dan meet deze "
            "negatieve controle niets en bewijst de purge hieronder niets.")
        # Elk lekkend event begon binnen de horizon vóór de grens.
        assert leaked <= LABEL_CFG.entry_lag_bars + HORIZON_BARS + 1

    def test_purging_by_the_horizon_removes_exactly_those_events(
        self, labels: BarrierLabels
    ) -> None:
        boundary = N_BARS // 2
        horizon = LABEL_CFG.entry_lag_bars + HORIZON_BARS
        purged_train_end = boundary - horizon
        assert _leaking_events(
            labels, train_end=purged_train_end, test_start=boundary) == 0, (
            "er lekt nog een event na purging op de volledige horizon; de "
            "gebruikte H dekt de gerealiseerde looptijd niet")


# --------------------------------------------------------------------------- #
# De poort in `validation/walk_forward.py`
# --------------------------------------------------------------------------- #
class TestThePurgedWalkForwardGate:
    def test_no_train_label_reaches_into_any_test_window(
        self, validation_cfg: ValidationConfig
    ) -> None:
        cfg = validation_cfg.model_copy(update={
            "label_horizon_bars": LABEL_CFG.entry_lag_bars + HORIZON_BARS,
            "embargo_bars": LABEL_CFG.entry_lag_bars + HORIZON_BARS,
        })
        folds = list(purged_walk_forward(N_BARS, cfg))
        assert folds, "geen folds; de toets meet niets"
        verify_no_overlap(folds)          # crasht bij overlap

    def test_the_folds_cover_real_ground(
        self, validation_cfg: ValidationConfig
    ) -> None:
        """Een splitter die één minuscule fold oplevert, haalt `verify_no_overlap`
        ook. De poort moet dus ook bewijzen dat er iets te splitsen viel."""
        cfg = validation_cfg.model_copy(update={
            "label_horizon_bars": LABEL_CFG.entry_lag_bars + HORIZON_BARS,
            "embargo_bars": LABEL_CFG.entry_lag_bars + HORIZON_BARS,
        })
        folds = list(purged_walk_forward(N_BARS, cfg))
        assert len(folds) >= 3
        assert all(f.test_idx.size == cfg.test_bars for f in folds)
        assert all(f.train_idx.size >= cfg.train_bars // 2 for f in folds)

    def test_zero_embargo_is_refused(
        self, validation_cfg: ValidationConfig
    ) -> None:
        """Purging zonder embargo laat de seriële correlatie staan."""
        bad = validation_cfg.model_copy(update={
            "embargo_bars": 0, "label_horizon_bars": 1})
        with pytest.raises(ConfigContractError, match="geen purged walk-forward"):
            list(purged_walk_forward(N_BARS, bad))

    def test_an_embargo_shorter_than_the_horizon_is_refused(self) -> None:
        """Het schema dwingt dit al af. Deze test bewaakt de tweede lijn: een
        met de hand gebouwd configobject dat het schema omzeilt."""
        with pytest.raises(Exception, match=r"embargo_bars|label_horizon"):
            ValidationConfig(label_horizon_bars=11, embargo_bars=2)

    def test_too_little_data_crashes_instead_of_yielding_nothing(
        self, validation_cfg: ValidationConfig
    ) -> None:
        """Nul folds stilzwijgend teruggeven laat elke aggregatie leeg middelen
        en levert een gemiddelde over niets op dat als resultaat wordt gelezen."""
        with pytest.raises(DataContractError, match="nul bruikbare folds"):
            list(purged_walk_forward(10, validation_cfg))


class TestTheGateCanGoRed:
    """De zelfcontrole van de splitter moet aantoonbaar kunnen vuren."""

    def test_verify_no_overlap_catches_a_leaking_fold(self) -> None:
        leaking = PurgedFold(
            fold_id=0,
            train_idx=np.arange(0, 100),
            test_idx=np.arange(95, 150),          # overlapt met train
            n_purged=0, embargo_bars=5, label_horizon_bars=1,
        )
        with pytest.raises(DataContractError, match="lookahead"):
            verify_no_overlap([leaking])

    def test_verify_no_overlap_catches_a_horizon_that_just_reaches(self) -> None:
        """Het randgeval, en het gevaarlijkste: train en test raken elkaar niet,
        maar het LABEL van het laatste train-event wel.

        Zonder deze test zou een `verify_no_overlap` die alleen op index-overlap
        controleert groen zijn, en dat is precies de fout die purging bestaat om
        te voorkomen.
        """
        just_reaching = PurgedFold(
            fold_id=1,
            train_idx=np.arange(0, 100),          # laatste train-event: 99
            test_idx=np.arange(105, 150),         # geen index-overlap
            n_purged=0, embargo_bars=5,
            label_horizon_bars=10,                # 99 + 10 = 109 >= 105
        )
        with pytest.raises(DataContractError, match="lookahead"):
            verify_no_overlap([just_reaching])

    def test_the_same_fold_is_clean_with_a_shorter_horizon(self) -> None:
        """De weigering is gericht: dezelfde geometrie met een horizon die er
        wél in past, is geldig. Anders zou `verify_no_overlap` alles weigeren."""
        fine = PurgedFold(
            fold_id=2,
            train_idx=np.arange(0, 100),
            test_idx=np.arange(105, 150),
            n_purged=0, embargo_bars=5, label_horizon_bars=4,   # 99 + 4 = 103
        )
        verify_no_overlap([fine])
