# src/tradebot/registry/promotion.py
"""Model lifecycle promotion — research → staging → production.

Gate rules (blueprint §8.5):
  research  → staging : oos_logloss < 0.69 AND sharpe > 0.0
  staging   → prod    : sharpe >= 0.5 AND max_dd <= 0.25 AND n_obs >= 200

All promotions are recorded in the catalog as a new ModelRecord entry
with an updated ``stage`` field (append-only, immutable history).
"""
from __future__ import annotations

import logging

from ..backtest.vectorized import reject_vectorized_evidence
from .catalog import ModelCatalog, ModelRecord

logger = logging.getLogger(__name__)

__all__ = ["RESEARCH_TO_STAGING", "STAGING_TO_PROD", "PromotionGates", "promote"]

# Default gate thresholds
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
) -> ModelRecord | None:
    """Promote the latest model for (symbol, side) to target_stage if gates pass.

    Parameters
    ----------
    catalog : ModelCatalog to read from and write to.
    symbol : asset symbol.
    side : "LONG" or "SHORT".
    target_stage : "staging" or "prod".
    gates : promotion gate rules (None = use defaults).

    Returns
    -------
    New ModelRecord with updated stage, or None if gates fail.
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
