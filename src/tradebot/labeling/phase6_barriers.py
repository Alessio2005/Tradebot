"""Triple-barrier labeling onder de `shift(2)`-conventie — Phase 6, deliverable 19.

WAAROM DIT NAAST `triple_barrier.py` STAAT EN NIET DAARIN
=========================================================
`labeling/triple_barrier.py` bevat een Numba-kernel die bit-identiek is
overgenomen uit een monoliet en die daarop is afgestemd: event-bars uit een
CUSUM-filter, bid/ask-quartetten, een adaptieve stop-loss op wick-ratio's, een
half-spread-clamp op rolling medianen, en `execution_delay_bars=1` gerekend
vanaf de OPEN van de volgende bar. Elk van die keuzes is verdedigbaar in zijn
eigen context en geen van alle is de context van deze fase.

Deze fase heeft een ander en veel smaller contract:

  * daily bars op een vast raster, geen event-bars;
  * één primaire richting per bar uit het Phase 3-baselinesignaal;
  * barrières op de causale EWMA-volatiliteit uit `conf/model/volatility.yaml`;
  * en bovenal de `shift(2)`-conventie, die de legacy-kernel niet kent.

Die kernel bendrukken tot dit contract zou hem voor beide gebruikers slechter
maken. De nieuwe functie is klein genoeg om in zijn geheel te lezen, en dat is
bij een labelfunctie de eigenschap die telt: elke off-by-one hierin is een
lookahead die pas maanden later als een te hoge AUC zichtbaar wordt.

DE TIJDAS, EXPLICIET
====================
::

    bar t      EVENT. Alles t/m de close van t mag hierin zitten: het primaire
               signaal, de EWMA-sigma, de regime-classificatie.
    bar t+1    FILL. `entry_lag_bars = 1`. De entryprijs is de CLOSE van t+1.
    bar t+2    eerste bar waarop een barrière kan raken. Vanaf hier worden
               high en low gemonitord.
    ...
    bar t+1+H  verticale barrière. Wordt geen horizontale barrière geraakt, dan
               is de exit de close van deze bar.

De eerste bar die het label kan beïnvloeden is dus `t+2`, en dat is precies wat
`shift(2)` betekent. `entry_lag_bars = 0` — handelen op de close waarop je
besluit — is door het schema uitgesloten en niet door een assertie hier: een
conventie die je per aanroep kunt uitzetten, is geen conventie.

WAT HET LABEL BETEKENT
======================
Twee grootheden, bewust gescheiden:

``barrier_outcome`` in {+1, 0, -1}
    Welke barrière eerst raakte, gezien vanuit de POSITIE (dus al vermenigvuldigd
    met de richting van het primaire signaal). +1 = profit target, -1 = stop
    loss, 0 = verticale barrière.

``meta_label`` in {0, 1}
    Het doel van het secondary model (López de Prado, AFML §3.6): had de trade
    die het primaire model wilde doen, geld opgeleverd? Dat is 1 wanneer het
    gerealiseerde rendement van de positie strikt positief is, en 0 anders.

    Het secondary model voorspelt UITSLUITEND deze kans. Het bepaalt nooit de
    richting; die komt uit het primaire model en zit al in het teken van de
    positie verwerkt. Zie `train/meta_label.py`, waar dat technisch geblokkeerd
    is in plaats van afgeraden (§12.1).

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md §12.1; López de Prado, AFML hoofdstuk 3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..schemas.config import LabelingConfig
from ..utils.failfast import CausalityViolationError, DataContractError, require

__all__ = [
    "BarrierLabels",
    "label_triple_barrier",
    "labels_to_frame",
]


@dataclass(frozen=True)
class BarrierLabels:
    """Wat de barrières opleverden, per event, met volledige herleidbaarheid."""

    #: Index van de EVENT-bar `t` in de invoerreeks. Alles wat het label
    #: bepaalt, ligt op `t+2` of later.
    event_idx: np.ndarray
    #: Index van de bar waarop de positie is gesloten (`t1` in AFML-notatie).
    exit_idx: np.ndarray
    #: +1 profit target, -1 stop loss, 0 verticale barrière.
    barrier_outcome: np.ndarray
    #: 1 wanneer de trade van het primaire model geld opleverde, anders 0.
    meta_label: np.ndarray
    #: Rendement van de POSITIE (richting al verrekend), close-to-close.
    realized_return: np.ndarray
    #: De richting die het primaire model wilde: +1 long, -1 short.
    side: np.ndarray
    #: De sigma waarop de barrières van dit event zijn geschaald.
    sigma: np.ndarray

    def __len__(self) -> int:
        return int(self.event_idx.size)

    @property
    def positive_ratio(self) -> float:
        """Aandeel `meta_label == 1`. De klassebalans die de gate toetst."""
        return float(self.meta_label.mean()) if len(self) else float("nan")

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_events": len(self),
            "positive_ratio": self.positive_ratio,
            "n_profit_target": int(np.count_nonzero(self.barrier_outcome == 1)),
            "n_stop_loss": int(np.count_nonzero(self.barrier_outcome == -1)),
            "n_vertical": int(np.count_nonzero(self.barrier_outcome == 0)),
            "mean_holding_bars": (
                float(np.mean(self.exit_idx - self.event_idx)) if len(self)
                else float("nan")
            ),
        }


def label_triple_barrier(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    sigma: np.ndarray,
    side: np.ndarray,
    cfg: LabelingConfig,
) -> BarrierLabels:
    """Label elk bar met een niet-nul richting volgens de drie barrières.

    Parameters
    ----------
    high, low, close : bar-arrays van gelijke lengte, in prijs-eenheden.
    sigma : de CAUSALE volatiliteit per bar, in return-eenheden per bar (dus
        niet geannualiseerd). `sigma[t]` mag uitsluitend informatie t/m de close
        van `t` bevatten; dat is een eigenschap van de aanroeper en wordt hier
        niet gecontroleerd omdat het niet controleerbaar is uit de waarden zelf.
    side : +1 / -1 / 0 per bar. 0 betekent "het primaire model wil hier niets"
        en levert geen event op.

    Returns
    -------
    BarrierLabels met één rij per event.

    Notes
    -----
    Er wordt NIET geïmputeerd. Een bar met een niet-eindige sigma of prijs levert
    geen event op; dat is een dataeigenschap die in de dekkingsrapportage hoort,
    niet iets om weg te vullen. Een bar waarvan het volledige barrièrevenster
    niet meer in de reeks past, levert evenmin een event op: een label dat op
    een afgekapt venster is bepaald, is systematisch naar de verticale barrière
    vertekend.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    side = np.asarray(side, dtype=np.float64)

    n = close.size
    require(
        high.size == n and low.size == n and sigma.size == n and side.size == n,
        "Triple-barrier kreeg reeksen van ongelijke lengte.",
        DataContractError,
        n_high=int(high.size), n_low=int(low.size), n_close=int(close.size),
        n_sigma=int(sigma.size), n_side=int(side.size),
    )
    require(
        cfg.entry_lag_bars >= 1,
        "entry_lag_bars = 0 betekent handelen op de close waarop je besluit. "
        "`reports/phase5_engine_diff.md` weerlegde die aanname; zij mag niet "
        "via een aanroep terugkeren.",
        CausalityViolationError,
        entry_lag_bars=cfg.entry_lag_bars,
    )
    require(
        bool(np.all(np.isin(side[np.isfinite(side)], (-1.0, 0.0, 1.0)))),
        "`side` bevat andere waarden dan -1, 0 en +1. Het primaire model levert "
        "een RICHTING; de grootte hoort bij de soevereine risicolaag.",
        DataContractError,
        unique=sorted({float(v) for v in np.unique(side[np.isfinite(side)])}),
    )

    lag = int(cfg.entry_lag_bars)
    horizon = int(cfg.horizon_bars)
    pt_mult = float(cfg.profit_target_sigma)
    sl_mult = float(cfg.stop_loss_sigma)

    events: list[int] = []
    exits: list[int] = []
    outcomes: list[int] = []
    returns: list[float] = []
    sides: list[float] = []
    sigmas: list[float] = []

    # De laatste bar waarop een VOLLEDIG venster nog past. `t + lag` is de
    # entry-bar; `t + lag + horizon` is de verticale barrière.
    last_event = n - 1 - lag - horizon

    for t in range(last_event + 1):
        s = side[t]
        if s == 0.0 or not np.isfinite(s):
            continue
        sig = sigma[t]
        if not np.isfinite(sig) or sig <= 0.0:
            continue

        entry_idx = t + lag
        entry_price = close[entry_idx]
        if not np.isfinite(entry_price) or entry_price <= 0.0:
            continue

        # Barrières in PRIJS-eenheden, gezien vanuit de positie.
        pt_return = pt_mult * sig
        sl_return = sl_mult * sig

        vertical_idx = entry_idx + horizon
        outcome = 0
        exit_idx = vertical_idx

        # Monitoring begint op `entry_idx + 1` = `t + 2` bij lag 1. De entry-bar
        # zelf telt NIET mee: op de close van die bar is de positie net
        # ingenomen en haar high/low liggen grotendeels vóór de fill.
        for j in range(entry_idx + 1, vertical_idx + 1):
            hi, lo = high[j], low[j]
            if not (np.isfinite(hi) and np.isfinite(lo)):
                break
            if s > 0:
                up = hi / entry_price - 1.0
                down = lo / entry_price - 1.0
            else:
                # Short: een DALING van de prijs is winst voor de positie.
                up = 1.0 - lo / entry_price
                down = 1.0 - hi / entry_price

            hit_pt = up >= pt_return
            hit_sl = down <= -sl_return
            if hit_pt and hit_sl:
                # Beide barrières binnen dezelfde bar. Zonder intrabar-data is
                # de volgorde ONBEKEND. De conservatieve toewijzing is de stop
                # loss: wie hier de profit target kiest, bouwt een optimistische
                # bias in het label die als modelkwaliteit terugkomt.
                outcome, exit_idx = -1, j
                break
            if hit_sl:
                outcome, exit_idx = -1, j
                break
            if hit_pt:
                outcome, exit_idx = 1, j
                break

        exit_price = close[exit_idx]
        if not np.isfinite(exit_price):
            continue
        realized = s * (exit_price / entry_price - 1.0)

        events.append(t)
        exits.append(exit_idx)
        outcomes.append(outcome)
        returns.append(float(realized))
        sides.append(float(s))
        sigmas.append(float(sig))

    outcome_arr = np.asarray(outcomes, dtype=np.int8)
    return_arr = np.asarray(returns, dtype=np.float64)
    return BarrierLabels(
        event_idx=np.asarray(events, dtype=np.int64),
        exit_idx=np.asarray(exits, dtype=np.int64),
        barrier_outcome=outcome_arr,
        meta_label=(return_arr > 0.0).astype(np.int8),
        realized_return=return_arr,
        side=np.asarray(sides, dtype=np.float64),
        sigma=np.asarray(sigmas, dtype=np.float64),
    )


def labels_to_frame(labels: BarrierLabels, index: pd.Index) -> pd.DataFrame:
    """Zet de labels om naar een frame geïndexeerd op de EVENT-timestamp.

    De index is die van bar `t`, niet van de entry of de exit. Dat is de bar
    waarop de features worden gemeten, en het is dus de bar waarop purging en
    embargo moeten aangrijpen.
    """
    require(
        len(index) > int(labels.exit_idx.max()) if len(labels) else True,
        "De index is korter dan de hoogste exit-index; labels en index horen "
        "bij verschillende reeksen.",
        DataContractError,
    )
    return pd.DataFrame(
        {
            "exit_ts": index[labels.exit_idx],
            "barrier_outcome": labels.barrier_outcome,
            "meta_label": labels.meta_label,
            "realized_return": labels.realized_return,
            "side": labels.side,
            "sigma": labels.sigma,
            "holding_bars": labels.exit_idx - labels.event_idx,
        },
        index=index[labels.event_idx],
    )
