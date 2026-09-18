"""apps/paper_monitor.py — Live Streamlit dashboard for paper trading.

Reads output from apps/paper_trade_runner.py (artefacts/paper_trade/).
Auto-refreshes every 3 seconds.

Usage:
    streamlit run apps/paper_monitor.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
_OUT  = _ROOT / "artefacts" / "paper_trade"

st.set_page_config(
    page_title="Paper Trade Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Auto-refresh every 3s ─────────────────────────────────────────────────────
REFRESH_INTERVAL = 3  # seconds

# ── Load helpers ──────────────────────────────────────────────────────────────

def _load_state() -> dict[str, Any] | None:
    p = _OUT / "state.json"
    if not p.exists():
        return None
    try:
        state: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
        return state
    except Exception:
        return None


def _load_equity_curve() -> pd.DataFrame:
    p = _OUT / "equity_curve.jsonl"
    if not p.exists():
        return pd.DataFrame(columns=["ts", "equity", "bar"])
    rows = []
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except Exception:
        pass
    if not rows:
        return pd.DataFrame(columns=["ts", "equity", "bar"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="mixed")
    return df.sort_values("ts").reset_index(drop=True)


def _load_audit() -> pd.DataFrame:
    p = _OUT / "audit.jsonl"
    if not p.exists():
        return pd.DataFrame()
    rows = []
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except Exception:
        pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "event_ts" in df.columns:
        df["event_ts"] = pd.to_datetime(df["event_ts"], utc=True, format="mixed")
    return df


# ── Page render ───────────────────────────────────────────────────────────────

def render() -> None:
    state = _load_state()
    eq_df = _load_equity_curve()
    audit_df = _load_audit()

    # ── Header ────────────────────────────────────────────────────────────────
    col_title, col_status = st.columns([3, 1])
    with col_title:
        st.title("Paper Trade Monitor")
        if state:
            st.caption(
                f"Mode: **{state.get('mode', 'paper_replay')}**  |  "
                f"Started: {state.get('start_ts', 'N/A')[:19]}  |  "
                f"Bar time: {state.get('current_bar_ts', 'N/A')[:19]}"
            )
    with col_status:
        if state is None:
            st.error("Waiting for runner...")
        elif state.get("cb_active"):
            st.error("CIRCUIT BREAKER TRIPPED")
        else:
            st.success("Running")

    if state is None:
        st.info(
            "No state data yet. Start the runner:\n\n"
            "```\npython apps/paper_trade_runner.py\n```"
        )
        time.sleep(2)
        st.rerun()
        return

    # ── Top KPIs ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    equity   = state.get("equity", 0.0)
    init_eq  = state.get("initial_equity", 200_000.0)
    pnl_usdt = state.get("pnl_usdt", 0.0)
    pnl_pct  = state.get("pnl_pct", 0.0)
    # CHIEF-4 (2026-05-28): the "Max Drawdown" KPI used to read drawdown_pct,
    # which is INSTANTANEOUS (peak-to-current).  It returns 0 the moment equity
    # recovers — masking real intraday dips visible on the chart.  We now read
    # max_drawdown_pct (historic worst) with a fall-back to a curve-derived
    # value so old state.json files still render correctly.
    dd_max  = state.get("max_drawdown_pct")
    dd_cur  = state.get("current_drawdown_pct", state.get("drawdown_pct", 0.0))
    dd_intra = state.get("intraday_drawdown_pct")
    if dd_max is None:
        # Fallback for legacy state.json: derive from the equity curve.
        if not eq_df.empty and len(eq_df) > 1:
            _eq = eq_df["equity"].values
            _peak = np.maximum.accumulate(_eq)
            dd_max = float(((_peak - _eq) / np.where(_peak > 0, _peak, 1)).max() * 100.0)
        else:
            dd_max = dd_cur
    sharpe   = state.get("rolling_sharpe_32d", None)
    n_trades = state.get("n_trades", 0)
    n_bars   = state.get("bars_processed", 0)

    k1.metric("NAV (USDT)", f"${equity:,.0f}", f"${pnl_usdt:+,.0f}")
    k2.metric("Return", f"{pnl_pct:+.2f}%")
    # CHIEF-4: show historic max DD as primary KPI, current DD as secondary.
    _delta_label = (
        f"now {dd_cur:.2f}% · today {dd_intra:.2f}%"
        if dd_intra is not None else f"now {dd_cur:.2f}%"
    )
    k3.metric("Max Drawdown", f"{dd_max:.2f}%", _delta_label,
               delta_color="off")
    k4.metric(
        "Rolling Sharpe",
        f"{sharpe:.2f}" if sharpe is not None else "—"
    )
    k5.metric("Trades", n_trades)
    k6.metric("Bars processed", f"{n_bars:,}")

    st.divider()

    # ── Equity curve + Drawdown ───────────────────────────────────────────────
    col_eq, col_dd = st.columns([2, 1])

    with col_eq:
        st.subheader("Equity Curve")
        if not eq_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=eq_df["ts"], y=eq_df["equity"],
                mode="lines", name="NAV",
                line=dict(color="#00CC96", width=2),
                fill="tozeroy", fillcolor="rgba(0,204,150,0.08)",
            ))
            fig.add_hline(
                y=init_eq, line_dash="dash",
                line_color="gray", annotation_text="Initial equity",
            )
            fig.update_layout(
                height=320, margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="USDT", xaxis_title="",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No equity data yet — waiting for first bars.")

    with col_dd:
        st.subheader("Drawdown")
        if not eq_df.empty and len(eq_df) > 1:
            eq_arr = eq_df["equity"].values
            peak   = np.maximum.accumulate(eq_arr)
            dd_arr = (peak - eq_arr) / np.where(peak > 0, peak, 1) * 100
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=eq_df["ts"], y=-dd_arr,
                mode="lines", name="Drawdown",
                line=dict(color="#EF553B", width=1.5),
                fill="tozeroy", fillcolor="rgba(239,85,59,0.12)",
            ))
            fig2.update_layout(
                height=320, margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="%", xaxis_title="",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Not enough bars for drawdown chart.")

    st.divider()

    # ── Positions + Trade log ─────────────────────────────────────────────────
    col_pos, col_trades = st.columns([1, 2])

    with col_pos:
        st.subheader("Open Positions")
        positions = state.get("positions", {})
        prices    = state.get("prices", {})
        if positions:
            rows = []
            for sym, info in positions.items():
                qty      = info.get("qty", 0.0)
                notional = info.get("notional", info.get("notional_usdt", 0.0))
                side_str = info.get("side", "FLAT")
                if abs(qty) < 1e-10:
                    continue
                rows.append({
                    "Symbol": sym,
                    "Side": side_str,
                    "Qty": f"{qty:+.4f}",
                    "Notional ($)": f"{notional:,.0f}",
                    "Last px": f"{prices.get(sym, 0):,.4f}",
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            else:
                st.info("No open positions.")
        else:
            st.info("No open positions.")

    with col_trades:
        st.subheader(f"Trade Log ({n_trades} fills)")
        if not audit_df.empty:
            cols_show = [c for c in [
                "event_ts", "symbol", "side", "qty_base",
                "fill_price", "notional_usdt", "signal_prob",
                "post_trade_cost_bps",
            ] if c in audit_df.columns]
            display = audit_df[cols_show].copy()
            if "event_ts" in display.columns:
                display["event_ts"] = display["event_ts"].dt.strftime("%Y-%m-%d %H:%M")
            if "notional_usdt" in display.columns:
                display["notional_usdt"] = display["notional_usdt"].apply(
                    lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
                )
            if "fill_price" in display.columns:
                display["fill_price"] = display["fill_price"].apply(
                    lambda x: f"{x:,.4f}" if pd.notna(x) else "—"
                )
            st.dataframe(
                display.tail(50).iloc[::-1],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No fills yet.")

    # ── Per-symbol P&L bar chart ──────────────────────────────────────────────
    if not audit_df.empty and "symbol" in audit_df.columns and "notional_usdt" in audit_df.columns:
        st.divider()
        st.subheader("Gross Notional by Symbol")
        sym_notional = (
            audit_df.groupby("symbol")["notional_usdt"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig3 = go.Figure(go.Bar(
            x=sym_notional["symbol"],
            y=sym_notional["notional_usdt"],
            marker_color="#636EFA",
        ))
        fig3.update_layout(
            height=220, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="USDT notional",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.caption(
        f"Auto-refreshing every {REFRESH_INTERVAL}s  |  "
        f"Data: {_OUT}  |  "
        f"Last update: {pd.Timestamp.now().strftime('%H:%M:%S')}"
    )
    time.sleep(REFRESH_INTERVAL)
    st.rerun()


if __name__ == "__main__":
    render()
else:
    render()
