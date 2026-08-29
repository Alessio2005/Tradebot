# src/tradebot/registry/promotion.py
"""Deployment-stages — research → staging → production.

TWEE ASSEN, EN HET GAT ERTUSSEN (Phase 7/8 Stage B-4)
=====================================================
Dit bestand modelleert waar een model DRAAIT. `registry/lifecycle.py` modelleert
hoeveel BEWIJS ervoor bestaat (`REGISTERED -> TESTED -> CANDIDATE -> PAPER ->
CHAMPION`). Die twee assen liepen tot Stage B niet gelijk, en in dat gat kon een
model productie in glippen op niets meer dan:

    staging -> prod : sharpe >= 0.5 AND max_dd <= 0.25 AND n_obs >= 200

Geen DSR, geen SPA, geen lookahead-suite, geen pre-registratie, geen `M`. Drie
in-sample-getallen die elk overgefit kunnen zijn, en precies de vrijheidsgraad
die de hele L11-laag hoort weg te nemen. Dat is fase-no-go 5 (*"een promotieclaim
zonder de Stage B-gates"*) via een tweede deur.

**`prod` vereist vanaf Stage B een geslaagd `GateResult`.** De drempels hieronder
blijven bestaan als SCREENING voor `staging` — een goedkope voorfilter die
hopeloze kandidaten tegenhoudt — maar zij zijn geen promotiebewijs en kunnen dat
ook niet worden. Zie `validation/gates.py`.

Alle promoties worden append-only in de catalog vastgelegd als een nieuwe
`ModelRecord` met een bijgewerkt `stage`-veld.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..backtest.vectorized import reject_vectorized_evidence
from ..utils.failfast import DataContractError, require
from .catalog import ModelCatalog, ModelRecord

if TYPE_CHECKING:                      # pragma: no cover - typing only
    from ..validation.gates import GateResult

logger = logging.getLogger(__name__)

__all__ = ["RESEARCH_TO_STAGING", "STAGING_TO_PROD", "PromotionGates", "promote"]

# Screening-drempels voor research -> staging. GEEN promotiebewijs: zie de
# moduledocstring. Zij zijn in-sample-grootheden zonder correctie voor `M`.
RESEARCH_TO_STAGING: dict[str, float] = {
    "oos_logloss_max": 0.69,
    "sharpe_min": 0.0,
}
STAGING_TO_PROD: dict[str, float] = {
    "sharpe_min": 0.5,
    "max_dd_max": 0.25,
    "n_obs_min": 200,
}


class PromotionGates:
    """Checks model metrics against gate thresholds.

    Parameters
    ----------
    research_to_staging : dict of {metric: threshold} for first promotion gate.
    staging_to_prod : dict of {metric: threshold} for second promotion gate.
    """

    def __init__(
        self,
        research_to_staging: dict[str, float] | None = None,
        staging_to_prod: dict[str, float] | None = None,
    ) -> None:
        self.r2s = research_to_staging or RESEARCH_TO_STAGING
        self.s2p = staging_to_prod or STAGING_TO_PROD

    def can_promote_to_staging(self, metrics: dict[str, float]) -> bool:
        logloss = metrics.get("oos_logloss", float("inf"))
        sharpe = metrics.get("sharpe", -float("inf"))
        return logloss < self.r2s["oos_logloss_max"] and sharpe > self.r2s["sharpe_min"]

    def can_promote_to_prod(self, metrics: dict[str, float]) -> bool:
        sharpe = metrics.get("sharpe", -float("inf"))
        max_dd = metrics.get("max_dd", float("inf"))
        n_obs = metrics.get("n_obs", 0)
        return (
            sharpe >= self.s2p["sharpe_min"]
            and max_dd <= self.s2p["max_dd_max"]
            and n_obs >= self.s2p["n_obs_min"]
        )


def promote(
    catalog: ModelCatalog,
    symbol: str,
    side: str,
    target_stage: str,
    gates: PromotionGates | None = None,
    *,
    gate_result: GateResult | None = None,
) -> ModelRecord | None:
    """Promote the latest model for (symbol, side) to target_stage if gates pass.

    Parameters
    ----------
    catalog : ModelCatalog to read from and write to.
    symbol : asset symbol.
    side : "LONG" or "SHORT".
    target_stage : "staging" or "prod".
    gates : screening thresholds for research -> staging (None = defaults).
    gate_result
        VERPLICHT voor `target_stage="prod"`. Het `GateResult` uit
        `validation.gates.run_promotion_gates`. Ontbreekt hij, dan CRASHT deze
        functie; hij weigert niet stil en logt geen waarschuwing. Een productie-
        promotie zonder de vijf poorten is geen zwak besluit maar een besluit
        dat niet genomen had mogen kunnen worden.

    Returns
    -------
    New ModelRecord with updated stage, or None if gates fail.

    Raises
    ------
    DataContractError
        Bij `prod` zonder `gate_result`, of met een `gate_result` dat niet is
        geslaagd.
    """
    gates = gates or PromotionGates()
    records = catalog.query(symbol=symbol, side=side, latest=True)
    if not records:
        logger.warning("No model found for %s/%s to promote.", symbol, side)
        return None

    latest = records[0]

    # PHASE 5 (audit sectie 16.1, fase-opdracht paragraaf 15): de vectorized
    # engine is uitsluitend toegestaan voor hypothese-screening. Een resultaat
    # dat daar vandaan komt, draagt NOT_ADMISSIBLE_AS_PROMOTION_EVIDENCE, en
    # de gate WEIGERT hem technisch in plaats van hem te loggen.
    #
    # De controle staat hier en niet in de aanroeper, omdat dit de enige plek
    # is waar een model van stage verandert. Een gate die je kunt overslaan
    # door een andere entrypoint te kiezen, is geen gate.
    reject_vectorized_evidence(
        latest.metrics, context=f"promote({symbol}/{side} -> {target_stage})")
    reject_vectorized_evidence(
        latest.extra, context=f"promote({symbol}/{side} -> {target_stage})")

    if target_stage == "staging":
        if not gates.can_promote_to_staging(latest.metrics):
            logger.info(
                "Promotion blocked (research→staging): %s/%s metrics=%s",
                symbol, side, latest.metrics,
            )
            return None

    elif target_stage == "prod":
        # STAGE B-4: de L11-poort staat VÓÓR de legacy-drempels, niet ernaast.
        # Zou hij erna staan, dan bepaalt `sharpe >= 0.5` nog steeds wie er
        # überhaupt aan de statistische toets toekomt.
        require(
            gate_result is not None,
            f"promote({symbol}/{side} -> prod) zonder GateResult. Productie "
            f"vereist de vijf poorten uit validation/gates.py: pre-registratie, "
            f"data-adequaatheid, lookahead-suite, DSR en SPA. De drempels in "
            f"STAGING_TO_PROD zijn in-sample-screening zonder correctie voor M "
            f"en tellen niet als bewijs (fase-no-go 5).",
            DataContractError,
            symbol=symbol, side=side,
        )
        assert gate_result is not None
        require(
            gate_result.passed,
            f"promote({symbol}/{side} -> prod) met een GateResult dat niet is "
            f"geslaagd: verdict={gate_result.verdict}, gefaalde poorten="
            f"{list(gate_result.failed_gates)}. Er is geen deelscore en geen "
            f"override.",
            DataContractError,
            symbol=symbol, side=side, verdict=gate_result.verdict,
        )
        if latest.stage != "staging":
            logger.warning(
                "Cannot promote directly to prod from stage=%s (must pass staging first).",
                latest.stage,
            )
            return None
        if not gates.can_promote_to_prod(latest.metrics):
            logger.info(
                "Promotion blocked (staging→prod): %s/%s metrics=%s",
                symbol, side, latest.metrics,
            )
            return None
    else:
        raise ValueError(f"Unknown target_stage: {target_stage!r}")

    promoted = ModelRecord(
        symbol=latest.symbol,
        side=latest.side,
        version=latest.version,
        stage=target_stage,
        artifact_path=latest.artifact_path,
        git_sha=latest.git_sha,
        dvc_hash=latest.dvc_hash,
        feature_hash=latest.feature_hash,
        metrics=latest.metrics,
        extra={**latest.extra, "promoted_from": latest.stage},
    )
    catalog.register(promoted)
    logger.info("Promoted %s/%s v%d to stage=%s.", symbol, side, promoted.version, target_stage)
    return promoted
