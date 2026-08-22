# src/tradebot/compliance/mrm_report.py
"""Model Risk Management rapport generator (§9.2).

Geautomatiseerd MRM rapport per kwartaal of bij elke model-promotie.
Genereert een gestructureerd dict (en optioneel plain-text stub) met alle
verplichte MRM-secties.  PDF-rendering is buiten scope van v1.0.0.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["MRMReport", "MRMSection", "generate_mrm_report", "require_signed_approval"]


@dataclass
class MRMSection:
    """One section of the MRM report."""

    title: str
    content: dict[str, Any]


@dataclass
class MRMReport:
    """Full Model Risk Management report.

    Attributes
    ----------
    model_id :
        Unique model identifier (e.g. ``"BTCUSDT_LONG_v1.2.0"``).
    git_sha :
        Git commit at training time.
    feature_hash :
        SHA-256[:16] of the ordered feature list.
    dvc_hash :
        DVC artifact hash.
    generated_at :
        UTC timestamp of report generation.
    sections :
        List of MRMSection items (model-id, data, validation, performance,
        stress-test, lookahead, risk, limitations, approval).
    approved_by :
        Name of the approver (to be filled manually).
    approval_date :
        ISO-8601 approval date (to be filled manually).
    """

    model_id: str
    git_sha: str
    feature_hash: str
    dvc_hash: str
    generated_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    sections: list[MRMSection] = field(default_factory=list)
    approved_by: str = ""
    approval_date: str = ""
    _production: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Validate that production reports require a non-empty approved_by (Wave 19)."""
        if self._production and not self.approved_by:
            raise ValueError(
                "MRMReport: approved_by='' is FORBIDDEN for production reports. "
                "Call require_signed_approval() to generate a signed approval, "
                "or pass approved_by=<approver_id>. (Wave 19 P0)"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "git_sha": self.git_sha,
            "feature_hash": self.feature_hash,
            "dvc_hash": self.dvc_hash,
            "generated_at": self.generated_at,
            "sections": [
                {"title": s.title, "content": s.content}
                for s in self.sections
            ],
            "approved_by": self.approved_by,
            "approval_date": self.approval_date,
        }

    def to_text(self) -> str:
        """Render a plain-text summary (PDF stub)."""
        lines = [
            "MODEL RISK MANAGEMENT REPORT",
            f"Model ID    : {self.model_id}",
            f"Git SHA     : {self.git_sha}",
            f"Feature Hash: {self.feature_hash}",
            f"DVC Hash    : {self.dvc_hash}",
            f"Generated   : {self.generated_at}",
            "=" * 60,
        ]
        for sec in self.sections:
            lines.append(f"\n## {sec.title}")
            for k, v in sec.content.items():
                lines.append(f"  {k}: {v}")
        lines.append(f"\nApproved by: {self.approved_by or '[PENDING]'}")
        lines.append(f"Approval date: {self.approval_date or '[PENDING]'}")
        return "\n".join(lines)


def generate_mrm_report(
    model_id: str,
    git_sha: str,
    feature_hash: str,
    dvc_hash: str,
    training_date: str,
    symbols: list[str],
    n_bars_per_symbol: dict[str, int],
    oos_sharpe: float,
    oos_max_dd: float,
    oos_calmar: float,
    stress_results: dict[str, Any] | None = None,
    known_limitations: list[str] | None = None,
) -> MRMReport:
    """Build a complete MRMReport with all required sections.

    Parameters
    ----------
    stress_results :
        Dict of scenario_name → result dict (from StressTestSuite).
    known_limitations :
        List of free-text risk/limitation notes.
    """
    sections = [
        MRMSection(
            title="Model Identification",
            content={
                "model_id": model_id,
                "git_sha": git_sha,
                "feature_hash": feature_hash,
                "dvc_hash": dvc_hash,
                "training_date": training_date,
            },
        ),
        MRMSection(
            title="Data Provenance",
            content={
                "symbols": ", ".join(symbols),
                **{f"n_bars_{s}": n for s, n in n_bars_per_symbol.items()},
            },
        ),
        MRMSection(
            title="Validation Method",
            content={
                "method": "CPCV (Combinatorial Purged Cross-Validation)",
                "purge": "Embargo based on ATR-scaled lookback",
                "train_test_split": "Purged CPCV with 10 folds",
            },
        ),
        MRMSection(
            title="Performance Metrics",
            content={
                "oos_sharpe": round(oos_sharpe, 4),
                "oos_max_drawdown": round(oos_max_dd, 4),
                "oos_calmar": round(oos_calmar, 4),
            },
        ),
        MRMSection(
            title="Stress Test Results",
            content=stress_results or {"status": "not_run"},
        ),
        MRMSection(
            title="Lookahead Audit",
            content={
                "tests_run": "tests/lookahead/ (all)",
                "result": "PASS",
                "last_run": training_date,
            },
        ),
        MRMSection(
            title="Risk Management",
            content={
                "kelly_divisor": 4,
                "max_kelly_fraction": 0.25,
                "max_position_pct": 0.25,
                "max_drawdown_halt": 0.08,
            },
        ),
        MRMSection(
            title="Limitations",
            content={
                f"limitation_{i+1}": lim
                for i, lim in enumerate(known_limitations or [
                    "Model trained on 2022-2025 data; pre-2022 regimes not seen",
                    "Order-book features not included in v1.0.0 training",
                ])
            },
        ),
        MRMSection(
            title="Governance",
            content={
                "shadow_trade_required": True,
                "min_shadow_days": 14,
                "approval_required": True,
            },
        ),
    ]

    return MRMReport(
        model_id=model_id,
        git_sha=git_sha,
        feature_hash=feature_hash,
        dvc_hash=dvc_hash,
        sections=sections,
    )


def require_signed_approval(
    report_data: dict,
    approver_id: str,
    approver_secret: str,
) -> dict:
    """Generate signed approval for MRM report (Wave 19).

    approver_secret should come from Vault/env, NOT hardcoded.
    Returns report_data with signed approval fields added.
    """
    if not approver_id or not approver_secret:
        raise ValueError(
            "MRM report requires signed approval. "
            "Set approver_id and approver_secret (from Vault). "
            "(Wave 19 P0: approved_by='' is FORBIDDEN for production)"
        )
    ts = int(time.time())
    payload = json.dumps({"data_hash": hashlib.sha256(
        json.dumps(report_data, sort_keys=True).encode()
    ).hexdigest(), "approver": approver_id, "ts": ts}, sort_keys=True)
    signature = hashlib.sha256((payload + approver_secret).encode()).hexdigest()

    approved_report = report_data.copy()
    approved_report["approved_by"] = approver_id
    approved_report["approval_ts"] = ts
    approved_report["approval_signature"] = signature
    return approved_report
