# src/tradebot/compliance/__init__.py
"""Governance & MRM compliance layer."""
from .champion_challenger import (
    ChampionChallenger,
    ChampionChallengerConfig,
    DMTestResult,
)
from .circuit_log import CircuitLog, CircuitLogEntry
from .mrm_report import MRMReport, MRMSection, generate_mrm_report
from .position_report import (
    DailyPositionReport,
    PositionSnapshot,
    generate_position_report,
)
from .shadow_trader import ShadowRecord, ShadowTrader

__all__ = [
    "ShadowTrader", "ShadowRecord",
    "ChampionChallenger", "ChampionChallengerConfig", "DMTestResult",
    "MRMReport", "MRMSection", "generate_mrm_report",
    "DailyPositionReport", "PositionSnapshot", "generate_position_report",
    "CircuitLog", "CircuitLogEntry",
]
