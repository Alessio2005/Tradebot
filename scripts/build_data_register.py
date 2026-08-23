"""Genereer docs/DATA_REGISTER.md uit de PIT-store — Phase 1, stap 11.

Dit document is vanaf nu de ENIGE geldige bron voor de vraag "welke data mag ik
gebruiken". Per dataset: bron, granulariteit, periode, `data_hash`, bekende
gaps, gemeten uitschieters, en de research tracks die hem mogen gebruiken.

Read-only ten opzichte van de store; schrijft alleen het register.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.data.pit_store import PitStore
from tradebot.data.validation import detect_gaps, detect_outliers
from tradebot.schemas.config import DataConfig, load_config
from tradebot.utils.hashing import dataframe_content_hash

#: Welke research track welke dataset mag gebruiken (audit sectie 7.1).
TRACK_ACCESS = {
    "ohlcv": "Alpha Research, Volatility Research (daily GARCH/EWMA), Regime Research",
    "funding": "Alpha Research (carry), Execution & TCA Research",
    "open_interest": "Alpha Research (positionering), Regime Research",
}


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT)
    ).decode().strip()


def dvc_dir_hash() -> str:
    p = ROOT / "data" / "pit_store.dvc"
    if not p.is_file():
        return "(niet DVC-tracked)"
    import yaml
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    return str(d["outs"][0]["md5"])


def main() -> int:
    cfg = load_config(ROOT / "conf" / "data" / "default.yaml", DataConfig)
    store = PitStore(ROOT / cfg.pit_store_root)
    refs = store.partitions()
    series: dict[tuple[str, str, str, str], list] = {}
    for r in refs:
        series.setdefault((r.asset_class, r.dataset, r.symbol, r.granularity), []).append(r)

    L: list[str] = []
    A = L.append
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    A("# DATA REGISTER")
    A("")
    A("> **De enige geldige bron voor de vraag: welke data mag ik gebruiken?**")
    A(">")
    A("> Een onderzoeksresultaat dat geen `data_hash` uit dit register citeert,")
    A("> is per definitie `INVALID` (audit sectie 7.2).")
    A("")
    A("**Gegenereerd:** " + stamp + "  ")
    A("**Generator:** `scripts/build_data_register.py`  ")
    A("**git_sha:** `" + git_sha() + "`  ")
    A("**DVC dir-hash van de PIT-store:** `" + dvc_dir_hash() + "`  ")
    A("**PIT-store root:** `" + cfg.pit_store_root.as_posix() + "`")
    A("")
    A("---")
    A("")
    A("## 1. Gecertificeerde datasets")
    A("")
    A("| Asset class | Dataset | Symbool | Gran. | Rijen | Van | Tot | `data_hash` |")
    A("|---|---|---|---|---:|---|---|---|")

    rows_by_dataset: dict[str, int] = {}
    details: list[tuple] = []
    for (ac, ds, sym, gran) in sorted(series):
        df = store.load(ac, ds, sym, gran)
        h = dataframe_content_hash(df)
        first = pd.Timestamp(int(df["event_ts_ns"].iloc[0]), unit="ns", tz="UTC")
        last = pd.Timestamp(int(df["event_ts_ns"].iloc[-1]), unit="ns", tz="UTC")
        A(f"| {ac} | {ds} | {sym} | {gran} | {len(df):,} | "
          f"{first.date()} | {last.date()} | `{h}` |")
        rows_by_dataset[ds] = rows_by_dataset.get(ds, 0) + len(df)
        gaps = detect_gaps(df, asset_class=ac, symbol=sym, granularity=gran)
        rep = (detect_outliers(df, symbol=sym, granularity=gran,
                               max_abs_log_return=cfg.max_abs_log_return)
               if "close" in df.columns else None)
        details.append((ac, ds, sym, gran, len(df), h, gaps, rep,
                        len(series[(ac, ds, sym, gran)])))
    A("")
    A("**Totaal:** " + ", ".join(f"{k}: {v:,} rijen" for k, v in
                                 sorted(rows_by_dataset.items())) +
      f" — {len(series)} reeksen, {len(refs)} partities.")
    A("")
    A("---")
    A("")
    A("## 2. Bekende gaps")
    A("")
    total_gaps = sum(len(g) for *_, g, _, _ in details)
    if total_gaps == 0:
        A("**Nul ontbrekende bars over alle reeksen.** Gemeten met")
        A("`data/validation/gaps.py` op de bar-cadans van elke granulariteit.")
        A("De gap-ledger (`artefacts/governance/gap_ledger.jsonl`) is leeg.")
    else:
        A("| Dataset | Symbool | Gaps | Ontbrekende bars |")
        A("|---|---|---:|---:|")
        for ac, ds, sym, gran, n, h, gaps, rep, nparts in details:
            if gaps:
                A(f"| {ds} | {sym} | {len(gaps)} | "
                  f"{sum(g.n_missing for g in gaps)} |")
    A("")
    A("Er wordt **nooit** geinterpoleerd. `gap_policy` staat in")
    A("`conf/data/default.yaml`; bij `reject` breekt een gat de ingestion.")
    A("")
    A("---")
    A("")
    A("## 3. Gemeten uitschieters")
    A("")
    A("Drempel: `max_abs_log_return = " + str(cfg.max_abs_log_return) + "` "
      "(uit `conf/data/`), `allow_price_jumps = " + str(cfg.allow_price_jumps) + "`.")
    A("")
    A("| Symbool | Bars | Sprongen > drempel | Grootste \\|log-return\\| | Zwaarste dag |")
    A("|---|---:|---:|---:|---|")
    for ac, ds, sym, gran, n, h, gaps, rep, nparts in details:
        if rep is None:
            continue
        worst = rep.worst_jumps[0][0][:10] if rep.worst_jumps else "-"
        A(f"| {sym} | {n:,} | {rep.n_price_jumps} | "
          f"{rep.max_abs_log_return:.4f} | {worst} |")
    A("")
    A("`allow_price_jumps = true` betekent: **gemeten, beoordeeld en als echte")
    A("marktgebeurtenis geaccepteerd**. Crypto kent dagen met bewegingen van")
    A("tientallen procenten (12 maart 2020, 19 mei 2021, november 2022); die")
    A("weggooien of winsoriseren zou de staartverdeling vervalsen, en juist die")
    A("staart bepaalt de drawdown. `high < low` en `volume == 0` blijven")
    A("onvoorwaardelijk fataal.")
    A("")
    A("---")
    A("")
    A("## 4. Toegang per research track (sectie 7.1)")
    A("")
    A("| Dataset | Mag gebruikt worden door |")
    A("|---|---|")
    for ds in sorted(rows_by_dataset):
        A(f"| `{ds}` | {TRACK_ACCESS.get(ds, '(niet toegewezen)')} |")
    A("")
    A("---")
    A("")
    A("## 5. Semantiek van de tijdkolommen")
    A("")
    A("| Dataset | `event_ts_ns` | `asof_ts_ns` |")
    A("|---|---|---|")
    A("| `ohlcv` | openingstijd van de bar | **sluitingstijd** — een daily bar "
      "over 3 jan is pas op 4 jan 00:00 UTC compleet |")
    A("| `funding` | settlement-moment | settlement-moment — een rate die om "
      "08:00 UTC settelt is **pas dan** bekend |")
    A("| `open_interest` | snapshot-moment | snapshot-moment |")
    A("")
    A("Elke koppeling tussen deze reeksen loopt via")
    A("`utils.time.asof_join(direction=\"backward\")` met een **verplichte**")
    A("tolerance. Bewezen truncatie-invariant in")
    A("`tests/lookahead/test_asof_join_crypto.py`.")
    A("")
    A("---")
    A("")
    A("## 6. Wat hier NIET staat")
    A("")
    A("| Ontbrekend | Gevolg |")
    A("|---|---|")
    A("| OHLCV 1m/5m | Realized Variance en HAR-RV zijn niet te schatten; de "
      "Volatility Research track blijft beperkt tot daily (EWMA, GARCH). |")
    A("| Orderboek L1/L2 | De eta-kalibratie (Phase 5) kan niet op echte "
      "depth-data. Open Question 1 blijft onbeantwoordbaar. |")
    A("| Liquidaties | Bybit publiceert geen historische liquidatie-feed via "
      "de publieke REST-API. |")
    A("| Historische delistings | Het universum bevat uitsluitend nog-actieve "
      "symbolen; zie het exit-rapport voor de omvang van de survivorship bias. |")
    A("")

    out = ROOT / "docs" / "DATA_REGISTER.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"{len(series)} series, {len(refs)} partitions, "
          f"{sum(rows_by_dataset.values()):,} rows")
    (ROOT / "artefacts" / "governance").mkdir(parents=True, exist_ok=True)
    (ROOT / "artefacts" / "governance" / "data_hashes.json").write_text(
        json.dumps({f"{ac}/{ds}/{sym}/{gran}": h
                    for ac, ds, sym, gran, _, h, _, _, _ in details},
                   indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
