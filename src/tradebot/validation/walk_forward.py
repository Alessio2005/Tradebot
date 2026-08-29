"""Purged Walk-Forward met expliciete embargo — Stage B-2, L11-ingang.

WAAROM DEZE WRAPPER BESTAAT NAAST `cv/walk_forward.py`
======================================================
`cv/walk_forward.py::WalkForwardCV` splitst correct en kent al een
`embargo_bars`. Maar hij laat twee dingen toe die een promotiegate niet mag
toelaten:

1. **`embargo_bars` mag 0 zijn** wanneer de aanroeper `require_embargo=False`
   zet. Dat is een verdedigbare ontsnapping voor een verkennende studie, maar
   niet voor de poort waarop een promotiebesluit rust.
2. **De embargo-lengte komt van de aanroeper**, niet uit `conf/`. Twee runs
   kunnen dus dezelfde code met een andere embargo draaien, en het resultaat is
   niet vergelijkbaar zonder de aanroepcode ernaast te leggen.

Deze module dwingt beide af: de embargo komt uit `ValidationConfig`, hij is
minstens de labelhorizon, en er is geen pad waarin hij nul is.

WAT PURGING EN EMBARGO ELK OPLOSSEN — ZIJ ZIJN NIET HETZELFDE
=============================================================
Bij een label met horizon `H` gebruikt de uitkomst van een event op `t` de data
tot `t + H`.

* **Purging** verwijdert train-events waarvan het LABEL het testvenster in
  loopt. Zonder purging traint het model op de uitkomst die het moet
  voorspellen.
* **Embargo** verwijdert daarnaast een marge NA de testperiode uit de
  eerstvolgende trainperiode. Dat vangt de seriële correlatie die overblijft
  wanneer het label zelf al is gepurged: features rond de grens zijn nog
  gecorreleerd met testuitkomsten, ook al overlapt het label niet.

`López de Prado (2018) §7.4` behandelt ze als twee stappen, en dat is niet
overdreven: purging alleen laat de correlatie staan, embargo alleen laat het
label lekken.

Ref: audit §17.1 (*"Purged Walk-Forward met embargo is ESSENTIAL en de ENIGE
toegestane CV in de promotiepijplijn"*), `conf/validation/default.yaml`.
"""
from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..cv.walk_forward import WalkForwardCV
from ..schemas.config import ValidationConfig
from ..utils.failfast import ConfigContractError, DataContractError, require

__all__ = ["PurgedFold", "purged_walk_forward", "verify_no_overlap"]


@dataclass(frozen=True)
class PurgedFold:
    """Eén fold met het bewijs van zijn eigen zuiverheid erbij."""

    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    #: Aantal train-events dat is verwijderd omdat het label het testvenster in liep.
    n_purged: int
    #: Aantal bars embargo tussen train en test.
    embargo_bars: int
    #: De labelhorizon waarop de purge is gebaseerd.
    label_horizon_bars: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "n_train": int(self.train_idx.size),
            "n_test": int(self.test_idx.size),
            "n_purged": self.n_purged,
            "embargo_bars": self.embargo_bars,
            "label_horizon_bars": self.label_horizon_bars,
            "train_end": int(self.train_idx.max()) if self.train_idx.size else -1,
            "test_start": int(self.test_idx.min()) if self.test_idx.size else -1,
        }


def _purge(
    train_idx: np.ndarray, test_idx: np.ndarray, horizon: int
) -> tuple[np.ndarray, int]:
    """Verwijder train-events waarvan het label tot IN het testvenster reikt.

    Een event op index `i` heeft een label dat data tot `i + horizon` gebruikt.
    Ligt `i + horizon` op of voorbij het begin van de testset, dan bevat het
    trainlabel testinformatie.
    """
    if test_idx.size == 0 or train_idx.size == 0:
        return train_idx, 0
    test_start = int(test_idx.min())
    keep = (train_idx + horizon) < test_start
    return train_idx[keep], int((~keep).sum())


def purged_walk_forward(
    n: int,
    config: ValidationConfig,
    *,
    mode: str = "rolling",
) -> Generator[PurgedFold, None, None]:
    """Yield purged walk-forward folds met embargo uit `conf/validation/`.

    Parameters
    ----------
    n
        Aantal observaties.
    config
        `train_bars`, `test_bars`, `embargo_bars` en `label_horizon_bars` komen
        hiervandaan. Geen enkele van deze waarden mag uit de aanroepcode komen.
    mode
        `"rolling"` of `"anchored"`.

    Raises
    ------
    ConfigContractError
        Wanneer de embargo nul is of de labelhorizon niet dekt. `ValidationConfig`
        dwingt dat laatste al af; deze controle staat er voor het geval iemand
        een handgemaakt configobject doorgeeft.
    DataContractError
        Wanneer er na purging geen bruikbare fold overblijft.
    """
    require(
        config.embargo_bars > 0,
        "Purged walk-forward met embargo_bars=0 is geen purged walk-forward. "
        "De laatste H train-events houden dan labels die het testvenster in "
        "lopen. Zet embargo_bars >= label_horizon_bars in "
        "conf/validation/default.yaml.",
        ConfigContractError,
        embargo_bars=config.embargo_bars,
    )
    require(
        config.embargo_bars >= config.label_horizon_bars,
        f"embargo_bars ({config.embargo_bars}) < label_horizon_bars "
        f"({config.label_horizon_bars}). De embargo moet minstens de labelhorizon "
        f"dekken, anders overlapt een trainlabel met de testperiode.",
        ConfigContractError,
    )

    cv = WalkForwardCV(
        train_size=config.train_bars,
        test_size=config.test_bars,
        step=config.test_bars,
        mode=mode,                                    # type: ignore[arg-type]
        min_train=config.train_bars,
        embargo_bars=config.embargo_bars,
        require_embargo=True,
    )

    produced = 0
    for fold in cv.split(n):
        train_idx, n_purged = _purge(
            np.asarray(fold.train_idx), np.asarray(fold.test_idx),
            config.label_horizon_bars,
        )
        if train_idx.size == 0 or fold.test_idx.size == 0:
            continue
        produced += 1
        yield PurgedFold(
            fold_id=fold.fold_id,
            train_idx=train_idx,
            test_idx=np.asarray(fold.test_idx),
            n_purged=n_purged,
            embargo_bars=config.embargo_bars,
            label_horizon_bars=config.label_horizon_bars,
        )

    require(
        produced > 0,
        f"Purged walk-forward leverde nul bruikbare folds op {n} observaties met "
        f"train_bars={config.train_bars}, test_bars={config.test_bars}, "
        f"embargo={config.embargo_bars}. Dat is een datavolume-probleem, geen "
        f"resultaat: stilzwijgend nul folds teruggeven zou elke daaropvolgende "
        f"aggregatie een lege verzameling laten middelen.",
        DataContractError,
        n_obs=n,
    )


def verify_no_overlap(folds: list[PurgedFold]) -> None:
    """Crash zodra een train-index een testlabel kan raken.

    Dit is de zelfcontrole van de splitter. Hij hoort altijd te slagen; hij
    bestaat omdat een splitter die stilzwijgend lekt, elk resultaat erna
    ongeldig maakt zonder dat iets rood wordt.
    """
    for f in folds:
        if f.train_idx.size == 0 or f.test_idx.size == 0:
            continue
        test_start = int(f.test_idx.min())
        latest_label_end = int(f.train_idx.max()) + f.label_horizon_bars
        require(
            latest_label_end < test_start,
            f"Fold {f.fold_id}: het laatste trainlabel eindigt op "
            f"{latest_label_end}, de testset begint op {test_start}. Het label "
            f"van een train-event gebruikt dus data uit de testperiode - dit is "
            f"lookahead, geen randgeval.",
            DataContractError,
            fold_id=f.fold_id,
        )
