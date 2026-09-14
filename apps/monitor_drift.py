# apps/monitor_drift.py
"""Stage 5 — Nightly feature drift monitoring.

Entry-point: ``tb-monitor-drift`` (defined in pyproject.toml).

Hydra config: conf/monitor_drift.yaml
  baseline_path : artefacts/features/baseline_features.parquet
  current_path  : artefacts/features/latest_features.parquet
  output_dir    : reports/drift/
  psi_critical  : 0.20
  alert_slack   : true
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="monitor_drift", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    import numpy as np
    import pandas as pd

    from tradebot.monitoring import AlertSeverity, check_feature_drift, send_alert

    baseline_path = Path(cfg.baseline_path)
    current_path  = Path(cfg.current_path)

    if not baseline_path.exists():
        raise FileNotFoundError(f"baseline_path not found: {baseline_path}")
    if not current_path.exists():
        raise FileNotFoundError(f"current_path not found: {current_path}")

    df_base = pd.read_parquet(baseline_path)
    df_cur  = pd.read_parquet(current_path)

    num_cols = df_base.select_dtypes(include=[np.number]).columns.tolist()

    reference = {col: df_base[col].dropna().values for col in num_cols}
    current   = {col: df_cur[col].dropna().values  for col in num_cols if col in df_cur.columns}

    results = check_feature_drift(reference, current)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "drift_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump([asdict(r) for r in results], fh, indent=2, default=str)
    logger.info("Drift report written: %s", report_path)

    critical = [r for r in results if r.is_critical]
    if critical and bool(cfg.get("alert_slack", True)):
        summary = ", ".join(f"{r.feature}(PSI={r.psi_value:.3f})" for r in critical[:5])
        send_alert(
            title="CRITICAL Feature Drift Detected",
            message=f"{len(critical)} features exceeded PSI=0.20: {summary}",
            severity=AlertSeverity.CRITICAL,
            metadata={"n_critical": len(critical), "n_features": len(results)},
        )

    if critical:
        logger.critical("%d CRITICAL drift features — see %s", len(critical), report_path)
        raise SystemExit(1)

    logger.info("Drift monitoring passed (%d features checked).", len(results))


if __name__ == "__main__":
    main()
