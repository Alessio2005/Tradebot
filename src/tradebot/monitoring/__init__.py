# src/tradebot/monitoring/__init__.py
"""Monitoring sub-package — drift detection, feature health, calibration, alerts."""
from __future__ import annotations

from .alerts import Alert, AlertRouter, AlertSeverity, send_alert
from .cross_account import CrossAccountConfig, CrossAccountMonitor
from .drift import PSI_CRITICAL, PSI_MODERATE, DriftResult, check_feature_drift, psi
from .feature_health import FeatureHealthReport, check_feature_health
from .prob_calibration import CalibrationResult, expected_calibration_error

__all__ = [
    "PSI_CRITICAL",
    "PSI_MODERATE",
    "Alert",
    "AlertRouter",
    "AlertSeverity",
    "CalibrationResult",
    "CrossAccountConfig",
    "CrossAccountMonitor",
    "DriftResult",
    "FeatureHealthReport",
    "check_feature_drift",
    "check_feature_health",
    "expected_calibration_error",
    "psi",
    "send_alert",
]
