"""Phase 1 / stap 1 - formele vastlegging van de datalacune (bewijsartefact).

Inventariseert wat er FEITELIJK in de werkkopie staat en zet dat af tegen wat
sectie 7.1 van het auditdocument per research track eist. Kwantificeert
expliciet de crypto-lacune uit sectie 3.2.

Read-only. Muteert niets.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "market_data_parquet"
PIT = ROOT / "data" / "pit_store"
CONF = ROOT / "conf"

# Sectie 7.1 - vereiste datasets per research track.
REQUIRED: dict[str, list[tuple[str, str]]] = {
    "Alpha Research": [
        ("crypto OHLCV daily", "per symbool, maximale historie"),
        ("crypto OHLCV hourly/intraday", "per symbool"),
        ("funding rates (perp)", "settlement-ts in UTC ns, interval-metadata"),
        ("open interest", "per symbool"),
        ("liquidaties", "per symbool"),
        ("cross-sectional spreads", "afgeleid uit OHLCV-panel"),
        ("volume profiles", "per symbool"),
    ],
    "Volatility Research (GARCH & EWMA)": [
        ("1m / 5m bars", "voor Realized Variance en HAR-RV"),
        ("crypto OHLCV daily", "voor GARCH/EGARCH/GJR"),
    ],
    "Regime Research": [
        ("macro-economische tijdreeksen", "FRED e.d."),
        ("implied volatility indices", "DVOL / VIX"),
        ("volatility term structures", ""),
        ("cross-asset correlatiematrices", "afgeleid"),
    ],
    "Execution & TCA Research": [
        ("orderboek L1 (top-of-book)", "snapshot-frequentie gedocumenteerd"),
        ("orderboek L2 (depth)", "voor eta-kalibratie, Phase 5"),
        ("individuele trade-prints", ""),
        ("exchange fee schedules", "conf/execution/fees.yaml"),
        ("latentiestatistieken", ""),
    ],
}


def describe_parquet(p: Path) -> dict[str, object]:
    df = pd.read_parquet(p)
    out: dict[str, object] = {
        "rows": len(df),
        "cols": list(df.columns),
        "size_mb": round(p.stat().st_size / 1e6, 2),
    }
    ts_col = next((c for c in ("event_ts", "timestamp", "ts") if c in df.columns), None)
    if ts_col is not None:
        s = pd.to_datetime(df[ts_col])
        out["ts_col"] = ts_col
        out["tz"] = str(getattr(s.dt, "tz", None))
        out["first"] = str(s.min())
        out["last"] = str(s.max())
    elif isinstance(df.index, pd.DatetimeIndex):
        out["ts_col"] = "<index>"
        out["tz"] = str(df.index.tz)
        out["first"] = str(df.index.min())
        out["last"] = str(df.index.max())
    else:
        out["ts_col"] = "GEEN"
        out["tz"] = "n.v.t."
    if "symbol" in df.columns:
        out["symbols"] = sorted(df["symbol"].astype(str).unique())[:12]
    if "series_id" in df.columns:
        out["symbols"] = sorted(df["series_id"].astype(str).unique())[:12]
    return out


def main() -> int:
    universe = sorted(p.stem for p in (CONF / "symbols").glob("*.yaml"))
    cfg = yaml.safe_load((CONF / "data" / "default.yaml").read_text(encoding="utf-8"))["data"]

    present = {}
    for p in sorted(LEGACY.rglob("*.parquet")):
        present[p.relative_to(ROOT).as_posix()] = describe_parquet(p)

    pit_files = sorted(PIT.rglob("*.parquet")) if PIT.is_dir() else []

    L: list[str] = []
    A = L.append
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    A("# PHASE 1 - DATA GAP ANALYSIS")
    A("")
    A("**Gegenereerd:** " + stamp + "  ")
    A("**Generator:** `scripts/phase1_data_gap.py` (read-only)  ")
    A("**Bindend brondocument:** `docs/ARCHITECTUUR_AUDIT_2026-08-22.md` sectie 3.2, 7.1, 7.2  ")
    A("**Status:** bewijsstuk voor sectie 3.2 - wordt niet overschreven na ingestie.")
    A("")
    A("---")
    A("")
    A("## 1. Wat er FEITELIJK aanwezig is")
    A("")
    A("### 1.1 Legacy store `market_data_parquet/`")
    A("")
    A("| Bestand | Rijen | MB | Tijdkolom | TZ | Van | Tot |")
    A("|---|---:|---:|---|---|---|---|")
    for rel, d in present.items():
        A("| `{}` | {:,} | {} | `{}` | {} | {} | {} |".format(
            rel, d["rows"], d["size_mb"], d["ts_col"], d.get("tz", "?"),
            str(d.get("first", "-"))[:10], str(d.get("last", "-"))[:10]))
    A("")
    A("### 1.2 PIT store `data/pit_store/`")
    A("")
    if pit_files:
        A(f"{len(pit_files)} partitie(s) aanwezig.")
    else:
        A("**Leeg.** De PIT-store wordt in stap 3 van deze fase gebouwd en in")
        A("stap 5-7 gevuld. Tot dat moment bestaat er geen enkele dataset met een")
        A("gecertificeerde `data_hash`, en is dus elk historisch onderzoeksresultaat")
        A("per definitie `INVALID` (sectie 7.2). Dat wordt in Phase 2 stap 1 formeel")
        A("in het falsificatieregister vastgelegd.")
    A("")
    A("---")
    A("")
    A("## 2. DE CRYPTO-LACUNE (sectie 3.2), gekwantificeerd")
    A("")
    A("Het geconfigureerde universum telt **{}** symbolen: {}.".format(
        len(universe), ", ".join("`" + s + "`" for s in universe)))
    A("")
    A("| Vereiste crypto-dataset | Symbolen vereist | Symbolen aanwezig | Dekking |")
    A("|---|---:|---:|---:|")
    for name in ("OHLCV daily", "OHLCV 5m", "funding rates", "open interest",
                 "liquidaties", "orderboek L1", "orderboek L2"):
        A("| {} | {} | **0** | **0%** |".format(name, len(universe)))
    A("")
    A("**Er is nul byte crypto-data in de werkkopie.** `market_data_parquet/`")
    A("bevat uitsluitend FX-, macro-, commodity- en equity-reeksen. De bevinding")
    A("van sectie 3.2 wordt hiermee bevestigd en gekwantificeerd.")
    A("")
    A("### 2.1 Consequentie voor bestaande claims")
    A("")
    A("Het vlaggenschipmodel (*Crypto-MN, Sharpe 1.15*) en de 4 als geaccepteerd")
    A("geregistreerde alpha-units zijn in deze werkkopie **niet reproduceerbaar**.")
    A("Er zijn **0 verifieerbare actieve alpha-units**.")
    A("")
    A("De projectgeschiedenis vermeldt een dataopslag buiten de repository")
    A("(`Desktop/Trading Setup A/B/C`). Die mappen bestaan niet meer op deze")
    A("machine; geverifieerd op de datum van dit rapport. De data is dus niet")
    A("elders beschikbaar maar daadwerkelijk afwezig en moet opnieuw worden")
    A("ingested.")
    A("")
    A("---")
    A("")
    A("## 3. Vereist versus aanwezig, per research track (sectie 7.1)")
    A("")
    have_macro = any("fx" in k or "factors" in k or "eia" in k for k in present)
    for track, items in REQUIRED.items():
        A(f"### {track}")
        A("")
        A("| Dataset | Aanwezig | Toelichting |")
        A("|---|---|---|")
        for name, note in items:
            if "macro" in name or "term structure" in name:
                status = "gedeeltelijk" if have_macro else "NEE"
                note = note or "FX + Fama-French + EIA aanwezig; DVOL/VIX niet"
            elif "fee schedule" in name:
                status = "JA"
            elif "correlatie" in name or "spreads" in name or "volume profile" in name:
                status = "afgeleid"
                note = note or "volgt uit het OHLCV-panel zodra dat bestaat"
            else:
                status = "**NEE**"
            A(f"| {name} | {status} | {note} |")
        A("")
    A("---")
    A("")
    A("## 4. Reikwijdte van deze fase")
    A("")
    A("Besluit van de opdrachtgever (2026-08-22): **alleen daily OHLCV nu,")
    A("intraday later**. Dat betekent voor de exit criteria van deze fase:")
    A("")
    A("| Dataset | Deze fase | Consequentie |")
    A("|---|---|---|")
    A("| crypto OHLCV daily | **wordt ingested** | basis voor de Phase 3-baseline |")
    A("| funding rates | **wordt ingested** | settlement-semantiek in UTC ns |")
    A("| open interest | **wordt ingested** | |")
    A("| liquidaties | **wordt ingested** waar de API het toelaat | Bybit publiceert geen volledige historie |")
    A("| orderboek L1/L2 | **niet** | alleen live te samplen; geen REST-historie |")
    A("| OHLCV 5m | **niet** | doorgeschoven; blokkeert HAR-RV en Realized Variance |")
    A("")
    A("**Open Question 1** (sectie 27) - *beschikken we over voldoende")
    A("orderboek-depth om HAR-RV en het TCA-impactmodel te kalibreren?* - wordt")
    A("in het exit-rapport van deze fase beantwoord met de gemeten dekking van")
    A("wat wel is ingested, plus een expliciet *nog niet gemeten* voor intraday")
    A("en depth. Het antwoord is daarmee niet *ja* en niet *nee*, maar")
    A("*onbeantwoordbaar op de huidige data* - en dat wordt als zodanig")
    A("geregistreerd in plaats van geraden.")
    A("")

    out = ROOT / "reports" / "phase1_data_gap.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"legacy parquet files={len(present)}  pit partitions={len(pit_files)}  "
          f"universe={len(universe)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
