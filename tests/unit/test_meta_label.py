# tests/unit/test_meta_label.py
"""Het secondary model en zijn poorten. Deliverables 20 en 21.

Wat hier wordt afgedwongen is niet dat CatBoost werkt — dat doet CatBoost — maar
de drie dingen waar een meta-labeling-pipeline stilletjes op stukloopt:

  * dat het model GEEN richting kan leren, omdat er geen route is om het een
    eigen doelvector te geven;
  * dat de purging op `t1` gebeurt en niet op een embargo-aantal, want een
    triple-barrier-label van 10 bars overleeft een embargo van 5;
  * dat de AUC de uniqueness meeweegt, want zonder die weging telt elk van tien
    overlappende buren als een volwaardige waarneming.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.cv.walk_forward import WalkForwardCV
from tradebot.labeling.phase6_barriers import BarrierLabels
from tradebot.reporting.phase6_meta_labeling import render_meta_label_report
from tradebot.schemas.config import MetaLabelConfig, meta_label_config
from tradebot.train.meta_label import (
    META_LABEL_GRID,
    MetaLabelDataset,
    build_dataset,
    fit_secondary_model,
    purged_training_index,
    shuffled_targets,
    walk_forward_predictions,
)
from tradebot.utils.failfast import DataContractError
from tradebot.validation.feature_importance import (
    mdi_importance,
    sfi_importance,
    weighted_auc,
)

CFG = meta_label_config()
#: Klein genoeg om snel te fitten, groot genoeg voor twaalf folds.
FAST = MetaLabelConfig(iterations=40, thread_count=1)
CV = WalkForwardCV(train_size=200, test_size=50, step=50, mode="rolling",
                   min_train=200, embargo_bars=5)
N_BARS = 500
HORIZON = 10


def _labels(n_events: int, seed: int, offset: int = 0) -> BarrierLabels:
    rng = np.random.default_rng(seed)
    event = np.arange(n_events) + offset
    return BarrierLabels(
        event_idx=event,
        exit_idx=event + HORIZON,
        barrier_outcome=rng.choice([-1, 0, 1], n_events),
        meta_label=(rng.random(n_events) < 0.5).astype(np.int64),
        realized_return=rng.normal(0.0, 0.02, n_events),
        side=rng.choice([-1.0, 1.0], n_events),
        sigma=np.full(n_events, 0.03),
    )


def _features(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(n, 4)),
        columns=["vol_a", "vol_b", "mom_a", "mom_b"],
        index=pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC"))


def _dataset(n_events: int = 460, seed: int = 3) -> MetaLabelDataset:
    return build_dataset(
        {"AAA": _features(N_BARS, seed), "BBB": _features(N_BARS, seed + 1)},
        {"AAA": _labels(n_events, seed), "BBB": _labels(n_events, seed + 1)})


class TestTheGridIsThePreregisteredSpace:
    def test_three_depths_times_two_rates_is_six_trials(self) -> None:
        assert len(META_LABEL_GRID) == 6
        assert len({s.label for s in META_LABEL_GRID}) == 6

    def test_a_duplicated_depth_is_refused(self) -> None:
        with pytest.raises(Exception, match="dubbelen"):
            MetaLabelConfig(depths=(3, 3))


class TestDirectionIsTechnicallyBlocked:
    def test_there_is_no_target_argument(self) -> None:
        """De pre-registratie noemt directionele voorspelling 'technisch
        geblokkeerd'. Dat is hier letterlijk: `build_dataset` leidt het doel af
        uit de barrièrelabels en neemt er geen."""
        import inspect
        assert "target" not in inspect.signature(build_dataset).parameters

    def test_a_continuous_target_is_refused(self) -> None:
        data = _dataset()
        with pytest.raises(DataContractError, match="niet binair"):
            MetaLabelDataset(
                features=data.features, target=data.side * 0.017,
                event_bar=data.event_bar, exit_bar=data.exit_bar,
                side=data.side, symbol=data.symbol,
                uniqueness=data.uniqueness)

    def test_an_event_without_a_direction_is_refused(self) -> None:
        data = _dataset()
        side = data.side.copy()
        side[5] = 0.0
        with pytest.raises(DataContractError, match="richting"):
            MetaLabelDataset(
                features=data.features, target=data.target,
                event_bar=data.event_bar, exit_bar=data.exit_bar, side=side,
                symbol=data.symbol, uniqueness=data.uniqueness)

    def test_the_control_route_only_accepts_a_permutation(self) -> None:
        """`walk_forward_predictions` heeft één ingang voor een ander doel, en
        die bestaat voor de negatieve controle. Hij accepteert geen doel met
        een andere klassebalans — anders was het alsnog een eigen doelvector."""
        data = _dataset()
        wrong = np.ones_like(data.target)
        with pytest.raises(DataContractError, match="permutatie"):
            walk_forward_predictions(
                data, CV, META_LABEL_GRID[0], FAST, n_bars=N_BARS,
                embargo_bars=5, target=wrong)


class TestPurging:
    def test_labels_that_reach_into_the_test_window_are_dropped(self) -> None:
        """De kern: een event 3 bars vóór het testvenster draagt een label van
        10 bars en RAAKT dat venster dus. Een embargo van 5 zou hem laten
        staan; de purge op `t1` niet."""
        data = _dataset()
        fold = next(f for f in CV.split(N_BARS) if f.fold_id == 1)
        rows, counts = purged_training_index(data, fold, embargo_bars=5)
        start = int(fold.test_idx[0])
        kept_before = data.event_bar[rows] < start
        assert np.all(data.exit_bar[rows][kept_before] < start)
        assert counts["n_purged_by_overlap"] > 0

    def test_no_training_event_starts_inside_the_test_window(self) -> None:
        data = _dataset()
        for fold in CV.split(N_BARS):
            rows, _ = purged_training_index(data, fold, embargo_bars=5)
            inside = (
                (data.event_bar[rows] >= int(fold.test_idx[0]))
                & (data.event_bar[rows] <= int(fold.test_idx[-1])))
            assert not inside.any()

    def test_the_embargo_zone_after_the_window_is_empty(self) -> None:
        data = _dataset()
        for fold in CV.split(N_BARS):
            rows, _ = purged_training_index(data, fold, embargo_bars=5)
            end = int(fold.test_idx[-1])
            zone = ((data.event_bar[rows] > end)
                    & (data.event_bar[rows] <= end + 5))
            assert not zone.any()

    def test_no_training_event_comes_from_after_the_test_window(self) -> None:
        """Het schema is WALK-FORWARD en niet purged K-fold.

        Dit is de test die er niet was toen `purged_training_index` het hele
        complement van de testfold als trainmateriaal nam. Zonder hem was elke
        andere purge-test groen terwijl ~70 % van elke fit uit de TOEKOMST van
        zijn eigen testvenster kwam.
        """
        data = _dataset()
        for fold in CV.split(N_BARS):
            rows, _ = purged_training_index(data, fold, embargo_bars=5)
            assert np.all(data.event_bar[rows] < int(fold.test_idx[0]))

    def test_the_training_window_is_the_rolling_one_and_not_everything_before(
            self) -> None:
        """`train_bars` doet er toe.

        Onder K-fold werd het venster genegeerd; de fit zag dan alles wat niet
        in de testfold zat. Hier hoort geen event van vóór de linkerrand van het
        rollende venster in de fit te zitten -- anders is `train_bars: 500` uit
        de pre-registratie een dood getal.
        """
        data = _dataset()
        late = [f for f in CV.split(N_BARS) if f.fold_id >= 2]
        assert late, "te weinig folds om een rollend venster te toetsen"
        for fold in late:
            rows, counts = purged_training_index(data, fold, embargo_bars=5)
            assert np.all(data.event_bar[rows] >= int(fold.train_idx[0]))
            assert counts["n_before_train_window"] > 0

    def test_the_fit_uses_far_fewer_events_than_the_whole_panel(self) -> None:
        """Een grofmazige maar harde bovengrens op de trainomvang.

        Het venster is `train_bars` bars maal het aantal symbolen; alles wat daar
        ruim boven ligt, betekent dat er ergens toch buiten het venster wordt
        gekeken.
        """
        data = _dataset()
        n_symbols = len(set(data.symbol.tolist()))
        for fold in CV.split(N_BARS):
            rows, _ = purged_training_index(data, fold, embargo_bars=5)
            assert rows.size <= len(fold.train_idx) * n_symbols

    def test_a_longer_embargo_never_keeps_more(self) -> None:
        data = _dataset()
        fold = next(iter(CV.split(N_BARS)))
        short, _ = purged_training_index(data, fold, embargo_bars=5)
        long, _ = purged_training_index(data, fold, embargo_bars=40)
        assert long.size <= short.size


class TestUniqueness:
    def test_overlapping_labels_get_less_than_one(self) -> None:
        """Met een horizon van 10 en een event op elke bar delen buren negen
        van hun tien bars. De uniqueness hoort dan ver onder 1 te liggen."""
        data = _dataset()
        assert data.uniqueness.max() <= 1.0
        assert float(np.median(data.uniqueness)) < 0.2

    def test_the_effective_sample_is_far_below_the_nominal(self) -> None:
        data = _dataset()
        assert data.effective_n < 0.3 * len(data)

    def test_a_lone_event_is_fully_unique(self) -> None:
        labels = BarrierLabels(
            event_idx=np.array([10]), exit_idx=np.array([20]),
            barrier_outcome=np.array([1]), meta_label=np.array([1]),
            realized_return=np.array([0.01]), side=np.array([1.0]),
            sigma=np.array([0.03]))
        other = BarrierLabels(
            event_idx=np.array([100, 200]), exit_idx=np.array([110, 210]),
            barrier_outcome=np.array([1, -1]), meta_label=np.array([1, 0]),
            realized_return=np.array([0.01, -0.01]), side=np.array([1.0, -1.0]),
            sigma=np.array([0.03, 0.03]))
        data = build_dataset(
            {"AAA": _features(300, 1), "BBB": _features(300, 2)},
            {"AAA": labels, "BBB": other})
        assert np.allclose(data.uniqueness, 1.0)


class TestWeightedAuc:
    def test_a_perfect_ranking_is_one(self) -> None:
        target = np.array([0, 0, 1, 1])
        assert weighted_auc(target, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0

    def test_a_reversed_ranking_is_zero(self) -> None:
        target = np.array([0, 0, 1, 1])
        assert weighted_auc(target, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0

    def test_a_constant_score_is_exactly_a_half(self) -> None:
        target = np.array([0, 1, 0, 1])
        assert weighted_auc(target, np.full(4, 0.5)) == 0.5

    def test_it_matches_sklearn_without_weights(self) -> None:
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(11)
        target = (rng.random(400) < 0.4).astype(int)
        score = rng.random(400) + 0.3 * target
        assert weighted_auc(target, score) == pytest.approx(
            roc_auc_score(target, score))

    def test_weights_change_the_answer(self) -> None:
        """Zonder dit zou de weging decoratief zijn."""
        target = np.array([0, 0, 1, 1])
        score = np.array([0.1, 0.9, 0.2, 0.8])
        assert weighted_auc(target, score) != weighted_auc(
            target, score, np.array([1.0, 0.01, 1.0, 0.01]))

    def test_one_class_crashes(self) -> None:
        with pytest.raises(DataContractError):
            weighted_auc(np.ones(10, dtype=int), np.random.random(10))


class TestTheModelNeverSeesTheTestWindow:
    def test_changing_the_test_rows_does_not_change_the_model(self) -> None:
        """De eigenschap die 'scaler per fold gefit' beschermt, direct gemeten:
        vervang de testrijen door ruis en eis dat de voorspellingen op de
        TRAINrijen niet bewegen."""
        data = _dataset()
        fold = next(iter(CV.split(N_BARS)))
        rows, _ = purged_training_index(data, fold, embargo_bars=5)
        model = fit_secondary_model(
            data.features.iloc[rows], data.target[rows],
            data.uniqueness[rows], META_LABEL_GRID[0], FAST)

        polluted = data.features.copy()
        test_rows = np.flatnonzero(
            (data.event_bar >= int(fold.test_idx[0]))
            & (data.event_bar <= int(fold.test_idx[-1])))
        polluted.iloc[test_rows] = 999.0
        other = fit_secondary_model(
            polluted.iloc[rows], data.target[rows], data.uniqueness[rows],
            META_LABEL_GRID[0], FAST)
        probe = data.features.iloc[rows].to_numpy(dtype="float64")
        assert np.array_equal(model.predict_proba(probe),
                              other.predict_proba(probe))

    def test_a_single_class_training_window_crashes(self) -> None:
        data = _dataset()
        with pytest.raises(DataContractError, match="één klasse"):
            fit_secondary_model(
                data.features.iloc[:50], np.zeros(50, dtype=np.int64),
                data.uniqueness[:50], META_LABEL_GRID[0], FAST)


class TestWalkForwardAndImportance:
    @pytest.fixture(scope="class")
    def folds(self):
        return walk_forward_predictions(
            _dataset(), CV, META_LABEL_GRID[0], FAST, n_bars=N_BARS,
            embargo_bars=5)

    def test_every_fold_reports_its_purge(self, folds) -> None:
        assert folds
        for fold in folds:
            record = fold.as_record()
            assert record["purge_n_kept"] <= record["purge_n_candidates"]
            assert record["purge_n_inside_test"] > 0

    def test_random_features_give_an_auc_near_a_half(self, folds) -> None:
        """De features zijn ruis en het label is een muntworp. Een AUC ver van
        0,5 zou betekenen dat er informatie lekt via de opzet zelf."""
        target = np.concatenate([f.target for f in folds])
        score = np.concatenate([f.probability for f in folds])
        weight = np.concatenate([f.uniqueness for f in folds])
        assert abs(weighted_auc(target, score, weight) - 0.5) < 0.1

    def test_mdi_sums_to_one_per_fold(self, folds) -> None:
        result = mdi_importance(folds, ["vol_a", "vol_b", "mom_a", "mom_b"])
        assert result.per_fold.sum(axis=1) == pytest.approx(1.0)
        assert len(result.ranked()) == 4

    def test_sfi_scores_every_feature_out_of_sample(self) -> None:
        result = sfi_importance(
            _dataset(), CV, META_LABEL_GRID[0], FAST, n_bars=N_BARS,
            embargo_bars=5)
        assert result.method == "SFI"
        assert result.mean.size == 4
        assert np.all((result.mean > 0.2) & (result.mean < 0.8))

    def test_shuffled_targets_keep_the_class_balance(self) -> None:
        data = _dataset()
        for shuffled in shuffled_targets(data, seed=1, n_replicates=3):
            assert shuffled.sum() == data.target.sum()
            assert not np.array_equal(shuffled, data.target)


# ─────────────────────────────────────────────────────────────────────────────
# HET RAPPORT MOET HET TEGENOVERGESTELDE KUNNEN ZEGGEN
#
# De renderer droeg drie conclusies als VASTE TEKST: dat alle zes AUC's onder
# 0,50 lagen, dat de negatieve controle schoon was, en dat SFI voor élke feature
# rond 0,50 uitkwam. Alle drie stonden er voordat de campagne één keer had
# gedraaid. Een rapport dat zijn uitkomst al kent, is hetzelfde defect als een
# toets die niet rood kan worden (§0.10) — alleen een stap later in de keten.
#
# Deze klasse is de negatieve controle op het RAPPORT: zij voert payloads in
# waarin de uitkomst de andere kant op wijst, en eist dat de tekst meebeweegt.
# ─────────────────────────────────────────────────────────────────────────────
def _payload(*, aucs, shuffled_auc=0.5012, sfi_mean=0.501,
             min_train_events=442.7, promoted=0, deltas=None, exposed=None):
    """Een H3-payload met precies die knoppen die de conclusies bepalen."""
    deltas = [-0.01] * len(aucs) if deltas is None else deltas
    exposed = [900] * len(aucs) if exposed is None else exposed
    trials = [
        {
            "label": f"catboost-d{3 + i}-lr0.03", "oos_auc": auc,
            "interval_nominal": [auc - 0.02, auc + 0.02, 0.01],
            "interval_conservative": [auc - 0.10, auc + 0.10, 0.05],
            "operating_point": {"threshold": 0.5, "precision": 0.49,
                                "recall": 0.5, "kept_fraction": 0.5,
                                "base_rate": 0.49, "n_kept": 10,
                                "n_total": 20},
            "operating_curve": [{"threshold": 0.5, "precision": 0.49,
                                 "recall": 0.5, "kept_fraction": 0.5,
                                 "n_kept": 10}],
            "arm": {"net_sharpe_oos": 0.1, "turnover_notional_oos": 1000,
                    "fees_oos": 12.0, "n_exposed_bars_oos": exposed[i],
                    "impact_status": "IMPACT_UNCALIBRATED",
                    "spread_status": "SPREAD_ASSUMED", "half_spread_bps": 2.5,
                    "risk_policy_hash": "abc123"},
            "net_sharpe_delta": deltas[i],
            "verdict": {"status": "ARCHIVED", "binding": [], "rationale": "x"},
        }
        for i, auc in enumerate(aucs)
    ]
    by_status = {"ARCHIVED": len(trials) - promoted}
    if promoted:
        by_status["PROMOTED"] = promoted
    scenario = {
        "assumptions": {"effective_n": 246.2},
        "minimum_detectable_effect": 0.1029, "informative": False,
    }
    return {
        "git_sha": "deadbeef", "preregistration_id": "56395fa2",
        "preregistration_frozen_utc": "2026-08-25T19:52:00Z",
        "ledger_total_at_freeze": 2776,
        "preregistration": {"content": {
            "primary_metric": "oos_auc", "planned_trials": len(trials),
            "stop_criteria": [
                {"name": "negative_control_leaks", "threshold": 0.55},
                {"name": "events_below_adequacy", "threshold": 100.0},
                {"name": "auc_below_threshold", "threshold": 0.58},
            ],
            "parameters": {"power_analysis": {
                "threshold_auc": 0.58, "threshold_source": "§24",
                "n_symbols": 6, "measured_mean_pairwise_correlation": 0.7485,
                "effective_independent_series": 1.2651,
                "scenarios": {"nominal": scenario,
                              "uniqueness_corrected": scenario,
                              "uniqueness_and_cross_section": scenario},
            }},
        }},
        "universe": {"symbols": ["A", "B"], "n_bars": 1743,
                     "period_start": "2020-03-26", "period_end": "2026-08-23"},
        "track": "xs_momentum_risk_parity",
        "labeling": {"profit_target_sigma": 2.0, "stop_loss_sigma": 2.0,
                     "horizon_bars": 10, "entry_lag_bars": 1},
        "probability_threshold": 0.5, "embargo_bars": 5,
        "ledger_before": 2776, "ledger_after": 2776,
        "trials_booked_at_freeze": 6,
        "campaign": {
            "n_trials": len(trials), "by_status": by_status,
            "unfiltered": {"net_sharpe_oos": 0.11,
                           "turnover_notional_oos": 1100, "fees_oos": 13.0,
                           "n_exposed_bars_oos": 950,
                           "impact_status": "IMPACT_UNCALIBRATED",
                           "spread_status": "SPREAD_ASSUMED",
                           "half_spread_bps": 2.5,
                           "risk_policy_hash": "abc123"},
            "shuffled_auc": shuffled_auc,
            "shuffled_auc_replicates": [shuffled_auc, 0.49, 0.50],
            "n_events": 10330, "positive_ratio": 0.496,
            "uniqueness_ratio": 0.162,
            "min_effective_events_per_fold": min_train_events,
            "min_effective_test_events_per_fold": 38.4,
            "effective_n_nominal": 3000.0, "effective_n_conservative": 102.6,
            "best_label": "catboost-d3-lr0.03",
            "mdi": {"method": "MDI", "features": [
                {"name": "vol_a", "mean": 0.30, "std": 0.05},
                {"name": "mom_a", "mean": 0.20, "std": 0.04}]},
            "sfi": {"method": "SFI", "features": [
                {"name": "vol_a", "mean": sfi_mean, "std": 0.02},
                {"name": "mom_a", "mean": 0.499, "std": 0.02}]},
            "trials": trials,
        },
    }


class TestTheReportReadsItsOwnNumbers:
    def test_it_says_below_when_every_auc_is_below(self) -> None:
        text = render_meta_label_report(_payload(aucs=[0.47, 0.48, 0.49]))
        assert "**Alle 3 liggen ONDER 0,50**" in text
        assert "BOVEN 0,50" not in text

    def test_it_says_above_when_every_auc_is_above(self) -> None:
        text = render_meta_label_report(_payload(aucs=[0.61, 0.62, 0.63]))
        assert "**Alle 3 liggen BOVEN 0,50**" in text
        assert "ONDER 0,50" not in text

    def test_a_mixed_field_is_counted_and_not_rounded_to_a_story(self) -> None:
        text = render_meta_label_report(_payload(aucs=[0.47, 0.62, 0.49]))
        assert "**2 van de 3 ligt onder 0,50**" in text

    def test_a_leaking_control_is_announced_in_the_heading(self) -> None:
        text = render_meta_label_report(
            _payload(aucs=[0.47, 0.48], shuffled_auc=0.61))
        assert "EN ZIJ LEKT" in text
        assert "en zij is schoon" not in text
        assert "falsificeert de RUN" in text

    def test_a_clean_control_still_reads_as_clean(self) -> None:
        text = render_meta_label_report(
            _payload(aucs=[0.47, 0.48], shuffled_auc=0.5012))
        assert "en zij is schoon" in text
        assert "EN ZIJ LEKT" not in text

    def test_a_binding_adequacy_gate_is_not_reported_as_slack(self) -> None:
        text = render_meta_label_report(
            _payload(aucs=[0.47, 0.48], min_train_events=61.0))
        assert "bindt hier WEL" in text
        assert "bindt hier NIET" not in text
        assert "ruim boven de eis" not in text

    def test_sfi_that_departs_from_a_half_is_not_called_flat(self) -> None:
        text = render_meta_label_report(
            _payload(aucs=[0.47, 0.48], sfi_mean=0.72))
        assert "dicht bij 0,50 uitkomt" not in text
        assert "wijkt voor minstens één feature merkbaar van 0,50 af" in text

    def test_a_promotion_does_not_print_that_there_are_none(self) -> None:
        text = render_meta_label_report(
            _payload(aucs=[0.61, 0.62], promoted=1))
        assert "Er zijn geen promotieclaims." not in text
        assert "vervalt zodra de impact is gekalibreerd" in text

    def test_a_positive_engine_delta_is_named_and_not_left_hanging(
            self) -> None:
        text = render_meta_label_report(
            _payload(aucs=[0.47, 0.48], deltas=[-0.30, +0.0449]))
        assert "1 van de 2 specs verbetert de netto Sharpe" in text
        assert "Geen enkele spec verbetert de netto Sharpe" not in text
        assert "+0.0449" in text

    def test_all_negative_deltas_are_reported_as_such(self) -> None:
        text = render_meta_label_report(
            _payload(aucs=[0.47, 0.48], deltas=[-0.30, -0.10]))
        assert "Geen enkele spec verbetert de netto Sharpe" in text
        assert "verbetert de netto Sharpe:" not in text

    def test_the_thinnest_exposure_is_reported(self) -> None:
        text = render_meta_label_report(
            _payload(aucs=[0.47, 0.48], exposed=[114, 753]))
        assert "114" in text

    def test_the_mde_comes_from_the_preregistration_and_is_not_a_constant(
            self) -> None:
        payload = _payload(aucs=[0.47, 0.48])
        (payload["preregistration"]["content"]["parameters"]["power_analysis"]
         ["scenarios"]["uniqueness_and_cross_section"]) = {
            "assumptions": {"effective_n": 900.0},
            "minimum_detectable_effect": 0.0421, "informative": True}
        text = render_meta_label_report(payload)
        assert "0.042" in text
        assert "0.103" not in text
