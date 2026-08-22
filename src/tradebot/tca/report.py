# src/tradebot/tca/report.py
"""TCA summary report generation."""
from __future__ import annotations

import logging

import pandas as pd

from .post_trade import PostTradeRecord

logger = logging.getLogger(__name__)

__all__ = ["TCAReport", "build_tca_report"]


class TCAReport:
    """Aggregated TCA statistics over a list of PostTradeRecords."""

    def __init__(self, records: list[PostTradeRecord]) -> None:
        self.records = records
        self._df: pd.DataFrame | None = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert records to a flat DataFrame."""
        if self._df is not None:
            return self._df

        rows = []
        for r in self.records:
            rows.append({
                "symbol":             r.symbol,
                "execution_time":     r.execution_time,
                "signed_qty":         r.signed_qty,
                "execution_price":    r.execution_price,
                "expected_cost_bps":  r.expected_cost.total_bps,
                "actual_is_bps":      r.is_decomp.total_bps,
                "delay_bps":          r.is_decomp.delay_component,
                "impact_bps":         r.is_decomp.market_impact,
                "cost_vs_expected":   r.cost_vs_expected_bps,
                "alpha_capture":      r.alpha_capture,
            })
        self._df = pd.DataFrame(rows)
        return self._df

    def summary(self) -> pd.Series:
        """Return key TCA statistics as a Series."""
        df = self.to_dataframe()
        if df.empty:
            return pd.Series(dtype=float)

        return pd.Series({
            "n_trades":              len(df),
            "mean_is_bps":           float(df["actual_is_bps"].mean()),
            "median_is_bps":         float(df["actual_is_bps"].median()),
            "mean_impact_bps":       float(df["impact_bps"].mean()),
            "mean_delay_bps":        float(df["delay_bps"].mean()),
            "mean_cost_vs_exp_bps":  float(df["cost_vs_expected"].mean()),
            "pct_worse_than_exp":    float((df["cost_vs_expected"] > 0).mean()) * 100,
            "mean_alpha_capture":    float(df["alpha_capture"].mean()),
        })

    def by_symbol(self) -> pd.DataFrame:
        """Return per-symbol TCA breakdown."""
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()
        return df.groupby("symbol").agg(
            n_trades=("signed_qty", "count"),
            mean_is_bps=("actual_is_bps", "mean"),
            mean_impact=("impact_bps", "mean"),
            mean_alpha_capture=("alpha_capture", "mean"),
        )


def build_tca_report(records: list[PostTradeRecord]) -> TCAReport:
    """Factory function to build a TCAReport from a list of records."""
    return TCAReport(records)
