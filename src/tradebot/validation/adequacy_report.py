"""Meet de adequaatheid van alle vijf Phase 6-modelklassen op DEZE data.

De wetenschap achter `apps/run_data_adequacy.py`. Phase 6 §3, stap 2:

    "Meet en publiceer de adequaatheid van alle vijf modelklassen VOORDAT je
    iets fit. Blijkt hier al dat H2 of H3 onhaalbaar is op dit universum, dan is
    dat de goedkoopste bevinding van de hele fase."

Elke functie hieronder TELT. Er wordt niets gefit, geen parameter geoptimaliseerd
en geen model geselecteerd. Dat is wat deze meting mag voorafgaan aan de
pre-registraties zonder hun betekenis te ondermijnen: pre-registratie beschermt
tegen *"ik mat de uitkomst en noemde het achteraf mijn voorspelling"*, en het
aantal observaties per fold is geen uitkomst.

De folds komen uit `conf/validation/default.yaml` en de drempels uit
`conf/model/adequacy.yaml`. Geen van beide staat hier als literal.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..cv.uniqueness import _calculate_average_uniqueness
from ..cv.walk_forward import WalkForwardCV
from ..labeling.phase6_barriers import label_triple_barrier
from ..schemas.config import (
    AdequacyConfig,
    LabelingConfig,
    ValidationConfig,
)
from ..utils.failfast import DataContractError, require
from .data_adequacy import (
    AdequacyVerdict,
    assess_garch,
    assess_har_rv,
    assess_hmm,
    assess_hrp,
    assess_meta_labeling,
)

__all__ = [
    "FoldGeometry",
    "measure_all",
    "measure_fold_geometry",
    "measure_intraday_coverage",
    "measure_meta_label_events",
]


@dataclass(frozen=True)
class FoldGeometry:
    """De feitelijke walk-forward indeling op deze reekslengte.

    `conf/validation/default.yaml` legt `n_splits: 6` vast als MINIMUM, niet als
    doel: de stap is `test_bars`, zodat de testvensters een partitie van de
    tijdas vormen. Op 1.743 bars levert dat meer folds dan zes op, en dat is het
    aantal dat de adequaatheid bepaalt — niet het geconfigureerde minimum.
    """

    n_bars: int
    train_sizes: tuple[int, ...]
    test_sizes: tuple[int, ...]
    n_folds: int

    def as_record(self) -> dict[str, Any]:
        return {
            "n_bars": self.n_bars,
            "n_folds": self.n_folds,
            "train_bars_min": min(self.train_sizes),
            "train_bars_max": max(self.train_sizes),
            "test_bars_min": min(self.test_sizes),
            "test_bars_max": max(self.test_sizes),
            "total_oos_bars": int(sum(self.test_sizes)),
        }


def measure_fold_geometry(n_bars: int, cfg: ValidationConfig) -> FoldGeometry:
    """Tel de folds die het geconfigureerde schema op deze lengte oplevert."""
    require(
        n_bars > cfg.train_bars + cfg.test_bars,
        "Te weinig bars voor een enkele walk-forward fold.",
        DataContractError,
        n_bars=n_bars, train_bars=cfg.train_bars, test_bars=cfg.test_bars,
    )
    cv = WalkForwardCV(
        train_size=cfg.train_bars, test_size=cfg.test_bars, step=cfg.test_bars,
        mode="rolling", min_train=cfg.train_bars, embargo_bars=cfg.embargo_bars,
    )
    trains: list[int] = []
    tests: list[int] = []
    for fold in cv.split(n_bars):
        trains.append(int(fold.train_idx.size))
        tests.append(int(fold.test_idx.size))
    require(
        len(trains) >= cfg.n_splits,
        "Minder walk-forward folds dan het geconfigureerde minimum.",
        DataContractError,
        realised=len(trains), required=cfg.n_splits,
    )
    return FoldGeometry(
        n_bars=int(n_bars),
        train_sizes=tuple(trains),
        test_sizes=tuple(tests),
        n_folds=len(trains),
    )


def measure_intraday_coverage(
    bars_per_day: Mapping[pd.Timestamp, int] | None,
    n_days_expected: int,
    cfg: AdequacyConfig,
) -> tuple[int, int]:
    """Tel de dagen met voldoende intraday-bars voor een RV-schatting.

    ``None`` betekent: de gecertificeerde store bevat de granulariteit NIET.
    Dat is een andere toestand dan "nul bars gevonden" en wordt als zodanig
    geteld — 0 van N dagen — maar het onderscheid staat in het rapport.
    """
    if bars_per_day is None:
        return n_days_expected, 0
    need = cfg.har_rv.min_bars_per_day
    covered = sum(1 for v in bars_per_day.values() if v >= need)
    return n_days_expected, covered


def measure_meta_label_events(
    ohlc: Mapping[str, pd.DataFrame],
    sigma: pd.DataFrame,
    side: pd.DataFrame,
    geometry: FoldGeometry,
    labeling: LabelingConfig,
    validation: ValidationConfig,
) -> tuple[list[int], list[float], list[float], dict[str, Any]]:
    """Tel de triple-barrier events per fold, ná purging en embargo.

    De events worden per symbool gelabeld en daarna GEPOOLD, omdat het secondary
    model op de gepoolde cross-sectie wordt getraind: één model dat de
    succeskans van het primaire signaal voorspelt, niet zes modellen. Dat is
    ook wat de klassebalans per fold betekent.

    Purging: een event op bar `t` heeft een label dat tot `t + lag + horizon`
    doorloopt. Valt dat interval over de train/test-grens, dan is het event uit
    de TRAIN-fold verwijderd. De embargo uit `conf/validation/` snijdt daar
    bovenop de eerste bars van elke testfold weg.
    """
    symbols = list(sigma.columns)
    per_symbol: dict[str, Any] = {}
    per_symbol_spans: list[tuple[np.ndarray, np.ndarray]] = []
    event_rows: list[tuple[int, int, int]] = []  # (event_idx, exit_idx, label)

    for symbol in symbols:
        frame = ohlc[symbol]
        labels = label_triple_barrier(
            high=frame["high"].to_numpy(dtype=np.float64),
            low=frame["low"].to_numpy(dtype=np.float64),
            close=frame["close"].to_numpy(dtype=np.float64),
            sigma=sigma[symbol].to_numpy(dtype=np.float64),
            side=side[symbol].to_numpy(dtype=np.float64),
            cfg=labeling,
        )
        per_symbol[symbol] = labels.as_dict()
        per_symbol_spans.append((labels.event_idx, labels.exit_idx))
        event_rows.extend(
            zip(labels.event_idx.tolist(), labels.exit_idx.tolist(),
                labels.meta_label.tolist(), strict=True)
        )

    cv = WalkForwardCV(
        train_size=validation.train_bars, test_size=validation.test_bars,
        step=validation.test_bars, mode="rolling",
        min_train=validation.train_bars, embargo_bars=validation.embargo_bars,
    )
    events = np.array([r[0] for r in event_rows], dtype=np.int64)
    exits = np.array([r[1] for r in event_rows], dtype=np.int64)
    labels_arr = np.array([r[2] for r in event_rows], dtype=np.int64)

    per_fold_counts: list[int] = []
    per_fold_effective: list[float] = []
    per_fold_ratio: list[float] = []
    for fold in cv.split(geometry.n_bars):
        train_lo, train_hi = int(fold.train_idx[0]), int(fold.train_idx[-1])
        # Purging: het LABEL-interval mag de trainrand niet overschrijden.
        in_train = (events >= train_lo) & (exits <= train_hi)
        n = int(np.count_nonzero(in_train))
        per_fold_counts.append(n)
        per_fold_ratio.append(
            float(labels_arr[in_train].mean()) if n else float("nan")
        )
        # EFFECTIEF aantal labels (AFML hoofdstuk 4). Met een event op vrijwel
        # elke bar en een horizon van tien bars delen buren negen van hun tien
        # toekomstige bars; het nominale aantal overschat de informatie-inhoud
        # dan met ongeveer die factor.
        #
        # De uniqueness wordt PER SYMBOOL berekend en daarna gesommeerd. Twee
        # symbolen die op dezelfde bar een event hebben, overlappen in de TIJD
        # maar niet in de UITKOMST: hun labels zijn verschillende trades. De
        # cross-sectionele afhankelijkheid tussen die uitkomsten is een ANDERE
        # correctie en zit in `effective_independent_series`.
        effective = 0.0
        for symbol_events, symbol_exits in per_symbol_spans:
            mask = ((symbol_events >= train_lo) & (symbol_exits <= train_hi))
            if not np.any(mask):
                continue
            uniqueness = _calculate_average_uniqueness(
                symbol_events[mask].astype(np.int32),
                # `t1` is exclusief in de kernel; de exit-bar telt mee.
                (symbol_exits[mask] + 1).astype(np.int32),
                geometry.n_bars,
            )
            effective += float(uniqueness.sum())
        per_fold_effective.append(effective)
    return per_fold_counts, per_fold_effective, per_fold_ratio, per_symbol


def measure_all(
    *,
    returns: pd.DataFrame,
    ohlc: Mapping[str, pd.DataFrame],
    sigma: pd.DataFrame,
    side: pd.DataFrame,
    intraday_bars_per_day: Mapping[pd.Timestamp, int] | None,
    observed_intraday_granularity: str,
    adequacy: AdequacyConfig,
    validation: ValidationConfig,
    labeling: LabelingConfig,
    hmm_state_counts: Sequence[int] = (2, 3),
) -> tuple[dict[str, AdequacyVerdict], dict[str, Any]]:
    """Meet alle vijf de modelklassen en geef de oordelen plus de ruwe cijfers."""
    geometry = measure_fold_geometry(len(returns), validation)

    verdicts: dict[str, AdequacyVerdict] = {}
    details: dict[str, Any] = {"fold_geometry": geometry.as_record()}

    # --- GARCH, per symbool ------------------------------------------------
    for symbol in returns.columns:
        verdicts[f"garch:{symbol}"] = assess_garch(
            geometry.train_sizes, adequacy, symbol=str(symbol))

    # --- HAR-RV ------------------------------------------------------------
    n_days, n_covered = measure_intraday_coverage(
        intraday_bars_per_day, geometry.n_bars, adequacy)
    verdicts["har_rv"] = assess_har_rv(
        n_days, n_covered, adequacy,
        observed_granularity=observed_intraday_granularity)

    # --- M2 HMM, per aantal toestanden -------------------------------------
    for k in hmm_state_counts:
        verdicts[f"hmm_k{k}"] = assess_hmm(geometry.train_sizes, k, adequacy)

    # --- Meta-labeling ------------------------------------------------------
    counts, effective, ratios, per_symbol = measure_meta_label_events(
        ohlc, sigma, side, geometry, labeling, validation)
    verdicts["meta_labeling"] = assess_meta_labeling(
        counts, ratios, adequacy, effective_events_per_fold=effective)
    details["meta_labeling"] = {
        "events_per_fold": counts,
        "effective_events_per_fold": effective,
        "positive_ratio_per_fold": ratios,
        "per_symbol": per_symbol,
    }

    # --- HRP ----------------------------------------------------------------
    verdicts["hrp"] = assess_hrp(returns, adequacy, n_folds=geometry.n_folds)

    return verdicts, details
