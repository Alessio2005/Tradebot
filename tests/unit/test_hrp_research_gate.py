"""HRP is technisch geblokkeerd voor productie — fase-6 no-go 15, Stage C-1.

De no-go luidde: *"HRP is productie-toegankelijk zonder bewijs."* Gemeten op
2026-08-29 was die conditie **actief**: `portfolio/hrp.py` bevatte geen `raise`,
geen vlag en geen markering. `hrp_weights(returns)` gaf gewichten terug die
elke aanroeper in een boek kon zetten.

Deliverable 23 eist dat de gate een **crash** is en geen vlag. Dit bestand
bewijst dat, langs twee onafhankelijke lijnen:

1. **Onderzoek vereist een expliciete verklaring.** Zonder `HrpResearchGate`
   crasht de aanroep. Er is geen default-instantie in enige handtekening.
2. **Productie is onbereikbaar.** De uitkomst draagt permanent
   `NOT_ADMISSIBLE_AS_PROMOTION_EVIDENCE`, en `registry/promotion.py` weigert
   hem daarop — via `reject_vectorized_evidence`, die de MARKERING toetst en
   niet de engine.

De tweede lijn is de belangrijkste. Een gate die alleen bij de ingang staat,
laat zich omzeilen door het resultaat door te geven; deze reist mee met het
resultaat zelf.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from tradebot.backtest.vectorized import (
    EVIDENCE_KEY,
    NOT_ADMISSIBLE,
    is_vectorized_evidence,
    reject_vectorized_evidence,
)
from tradebot.portfolio.hrp import (
    ADMISSION_CRITERIA,
    HrpAllocation,
    HRPOptimizer,
    HrpResearchGate,
    hrp_weights,
)
from tradebot.registry.catalog import ModelCatalog, ModelRecord
from tradebot.registry.promotion import promote
from tradebot.utils.failfast import DataContractError

VALID_REASON = (
    "deliverable 24: HRP vergelijken met Inverse Volatility op "
    "turnover-gecorrigeerd OOS-rendement"
)


@pytest.fixture(scope="module")
def gate() -> HrpResearchGate:
    return HrpResearchGate(reason=VALID_REASON,
                           preregistration_id="prereg-hrp-h4")


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    """Zes assets met hoge onderlinge correlatie — dit universum.

    `fase_6` §0.7: gemiddelde paarsgewijze correlatie 0,735, één cluster. De
    fixture bootst dat na, zodat de tests niet groen worden op een universum dat
    dit platform niet heeft.
    """
    rng = np.random.default_rng(20260829)
    n, k = 400, 6
    common = rng.normal(0.0, 0.02, n)
    cols = {f"SYM{i}": common * 0.85 + rng.normal(0.0, 0.011, n) for i in range(k)}
    return pd.DataFrame(cols)


# --------------------------------------------------------------------------- #
# 1. De gate crasht
# --------------------------------------------------------------------------- #
class TestTheGateIsACrashNotAFlag:
    def test_calling_without_a_gate_is_a_type_error(self, returns) -> None:
        """`research_gate` is keyword-only zonder default: Python zelf weigert."""
        with pytest.raises(TypeError):
            hrp_weights(returns)                      # type: ignore[call-arg]

    @pytest.mark.parametrize("bogus", [None, "ja", 1, True, object()])
    def test_anything_that_is_not_a_gate_crashes(self, returns, bogus) -> None:
        with pytest.raises(DataContractError, match="RESEARCH ONLY"):
            hrp_weights(returns, research_gate=bogus)  # type: ignore[arg-type]

    def test_the_crash_names_both_admission_criteria(self, returns) -> None:
        """Wie de gate raakt, moet niet hoeven zoeken wat hem zou openen."""
        with pytest.raises(DataContractError) as exc:
            hrp_weights(returns, research_gate=None)   # type: ignore[arg-type]
        message = str(exc.value)
        assert "turnover" in message
        assert "STABIEL" in message or "stabiel" in message

    def test_the_optimizer_demands_the_gate_at_construction(self) -> None:
        """Niet pas bij optimize(): een object dat er onschuldig uitziet en pas
        diep in een aanroepketen crasht, is een slechtere gate."""
        with pytest.raises(DataContractError, match="geblokkeerd"):
            HRPOptimizer(research_gate="onderzoek")    # type: ignore[arg-type]
        with pytest.raises(TypeError):
            HRPOptimizer()                             # type: ignore[call-arg]


class TestTheTokenCannotBeEmpty:
    def test_a_short_reason_is_refused(self) -> None:
        """`"test"` is geen reden. De ondergrens maakt het ritueel duurder dan
        even nadenken."""
        with pytest.raises(DataContractError, match="inhoudelijke reden"):
            HrpResearchGate(reason="onderzoek", preregistration_id="p")

    def test_whitespace_is_not_a_reason(self) -> None:
        with pytest.raises(DataContractError, match="inhoudelijke reden"):
            HrpResearchGate(reason=" " * 80, preregistration_id="p")

    def test_a_missing_preregistration_is_refused(self) -> None:
        """Een HRP-run is een trial; zonder pre-registratie telt hij niet in M."""
        with pytest.raises(DataContractError, match="preregistration_id"):
            HrpResearchGate(reason=VALID_REASON, preregistration_id="")

    def test_the_token_is_immutable(self, gate: HrpResearchGate) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            gate.reason = "iets anders"                # type: ignore[misc]


# --------------------------------------------------------------------------- #
# 2. Onderzoek blijft mogelijk
# --------------------------------------------------------------------------- #
class TestResearchStillWorks:
    """Een gate die ALLES weigert, maakt deliverable 24 onmogelijk. De weigering
    moet gericht zijn."""

    def test_a_valid_gate_produces_weights(self, returns, gate) -> None:
        alloc = hrp_weights(returns, research_gate=gate)
        assert isinstance(alloc, HrpAllocation)
        assert len(alloc.weights) == returns.shape[1]
        assert alloc.weights.sum() == pytest.approx(1.0)
        assert (alloc.weights > 0).all()

    def test_the_optimizer_works_with_a_gate(self, returns, gate) -> None:
        opt = HRPOptimizer(research_gate=gate)
        alloc = opt.optimize(returns)
        assert opt.allocation is alloc
        assert opt.weights is not None

    def test_the_tree_order_is_exposed_for_stability_measurement(
        self, returns, gate
    ) -> None:
        """No-go 11 eist dat de boomordening STABIEL is over folds. Dat is niet
        te meten als de ordening niet naar buiten komt."""
        alloc = hrp_weights(returns, research_gate=gate)
        assert sorted(alloc.tree_order) == list(range(returns.shape[1]))

    def test_the_high_correlation_of_this_universe_travels_with_the_result(
        self, returns, gate
    ) -> None:
        """De context zonder welke elke HRP-uitkomst verkeerd wordt gelezen."""
        alloc = hrp_weights(returns, research_gate=gate)
        assert alloc.audit["mean_abs_pairwise_corr"] > 0.5, (
            "de fixture heeft geen hoog-gecorreleerd universum; de tests meten "
            "dan iets anders dan dit platform")

    def test_a_single_asset_degenerates_cleanly(self, gate) -> None:
        one = pd.DataFrame({"SYM0": np.random.default_rng(1).normal(0, 0.02, 50)})
        alloc = hrp_weights(one, research_gate=gate)
        assert alloc.weights.iloc[0] == 1.0

    def test_an_empty_matrix_crashes(self, gate) -> None:
        with pytest.raises(DataContractError, match="zonder kolommen"):
            hrp_weights(pd.DataFrame(), research_gate=gate)


# --------------------------------------------------------------------------- #
# 3. Productie blijft onbereikbaar — de belangrijkste helft
# --------------------------------------------------------------------------- #
class TestProductionIsUnreachable:
    def test_the_result_is_permanently_marked(self, returns, gate) -> None:
        alloc = hrp_weights(returns, research_gate=gate)
        assert alloc.evidence_class == NOT_ADMISSIBLE
        assert is_vectorized_evidence(alloc), (
            "de bestaande weigeringsmachinerie herkent HRP-output niet; er zou "
            "dan een TWEEDE mechanisme nodig zijn, en dat is precies hoe een "
            "gate uit elkaar groeit")

    def test_the_marking_survives_serialisation(self, returns, gate) -> None:
        """De vorm waarin een resultaat de promotion gate feitelijk bereikt is
        een platte dict uit een JSON-artefact, geen dataclass."""
        record = hrp_weights(returns, research_gate=gate).as_record()
        assert record[EVIDENCE_KEY] == NOT_ADMISSIBLE
        assert is_vectorized_evidence(record)
        assert record["research_reason"] == VALID_REASON
        assert record["admission_criteria_not_yet_met"] == list(ADMISSION_CRITERIA)

    def test_the_evidence_class_cannot_be_edited(self, returns, gate) -> None:
        alloc = hrp_weights(returns, research_gate=gate)
        with pytest.raises(dataclasses.FrozenInstanceError):
            alloc.evidence_class = "FINE"              # type: ignore[misc]

    def test_the_promotion_gate_refuses_it(self, returns, gate) -> None:
        record = hrp_weights(returns, research_gate=gate).as_record()
        with pytest.raises(DataContractError, match="promotiebewijs"):
            reject_vectorized_evidence(record, context="hrp")

    def test_promote_refuses_a_model_carrying_hrp_evidence(
        self, returns, gate, tmp_path
    ) -> None:
        """De end-to-end weigering, via het echte promotiepad."""
        catalog = ModelCatalog(tmp_path / "catalog.json")
        catalog.register(ModelRecord(
            symbol="BTCUSDT", side="LONG", version=1, stage="research",
            artifact_path="m.joblib", git_sha="abc", dvc_hash="d",
            feature_hash="f",
            # Metrics die de legacy-drempels RUIM halen, zodat de test niet
            # groen wordt omdat de screening toevallig faalde.
            metrics={"oos_logloss": 0.40, "sharpe": 2.0},
            extra=hrp_weights(returns, research_gate=gate).as_record(),
        ))
        with pytest.raises(DataContractError, match="promotiebewijs"):
            promote(catalog, "BTCUSDT", "LONG", "staging")


class TestTheAdmissionCriteriaAreCodeNotProse:
    """Een toekomstige promotie moet ze AANRAKEN in plaats van eromheen te
    schrijven."""

    def test_both_criteria_are_declared(self) -> None:
        assert len(ADMISSION_CRITERIA) == 2

    def test_they_name_turnover_and_stability(self) -> None:
        joined = " ".join(ADMISSION_CRITERIA).lower()
        assert "turnover" in joined
        assert "inverse volatility" in joined
        assert "stabiel" in joined

    def test_the_engine_and_config_hash_are_named(self) -> None:
        """Turnover-gecorrigeerde Sharpe moet uit de authoritative engine komen,
        niet uit een vectorized benadering."""
        joined = " ".join(ADMISSION_CRITERIA)
        assert "backtest/engine.py" in joined
        assert "1b60cb664fbf9a2a" in joined
