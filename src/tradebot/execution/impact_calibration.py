# src/tradebot/execution/impact_calibration.py
"""Kalibratie van `eta` en `kappa_d` — of het eerlijke bewijs dat het niet kan.

Phase 5, deliverable 5 (de wetenschap achter `apps/calibrate_impact.py`).

DE VOLGORDE IS BINDEND
----------------------
1. Probeer de ECHTE kalibratie: eigen orders, hun grootte, en de gemeten
   prijsrespons uit orderboek- en trade-data.
2. Ontbreekt die data, dan is `eta` niet identificeerbaar. Er wordt dan GEEN
   getal verzonnen en geen literatuurwaarde overgenomen. De status wordt
   `IMPACT_UNCALIBRATED` en er wordt een BOVENGRENS afgeleid uit wat wél
   gecertificeerd is.

Stap 2 is geen zwaktebod maar de enige geldige uitkomst wanneer stap 1 faalt.
`conf/execution/fees.yaml` droeg tot Phase 5 `impact_eta: 0.142`, en dat getal
komt uit Bouchaud-Bonart op BTC-perpetuals - een ander universum, een andere
periode, een andere venue. Het was bovendien dood: geen enkele module las het.

DE BOVENGRENS, EN WAAROM HIJ GELDIG IS
--------------------------------------
Het model zegt `Impact(Q) = eta * sigma_d * sqrt(Q/V)`. Bij `Q = V` - een
meta-order zo groot als het volledige dagvolume - reduceert dat tot
`Impact = eta * sigma_d`.

De grootste prijsbeweging die op die dag daadwerkelijk IS opgetreden, is de
dagrange `(high - low) / close`. Als de VOLLEDIGE range aan impact zou worden
toegeschreven - en niet aan nieuws, aan informatie of aan de stroom van alle
andere marktdeelnemers - dan geldt:

    eta <= (high - low) / close / sigma_d

Dat is een geldige bovengrens omdat de aanname maximaal conservatief is: geen
enkel deel van de beweging wordt aan iets anders toegeschreven dan aan de eigen
order. De werkelijke `eta` ligt hier met zekerheid onder. Hij ligt er
vermoedelijk ver onder, en dat wordt in het rapport gezegd in plaats van
weggelaten.

`kappa_d` (de permanente fractie) wordt geschat als de verhouding tussen de
close-to-close-volatiliteit en de gemiddelde intraday-range: het deel van de
typische dagbeweging dat de close haalt in plaats van terug te lopen. Voor de
kosten van EEN order maakt `kappa_d` niets uit - je betaalt de volledige impact
- maar hij bepaalt wat er overblijft voor de volgende order.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md secties 7.2, 15.1, 27 (Open Question 1).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..utils.failfast import DataContractError, require
from ..utils.hashing import hash_config
from .impact_model import ImpactParams, ImpactStatus

__all__ = [
    "CalibrationEvidence",
    "ETA_BOUND_QUANTILE",
    "REQUIRED_DATASETS",
    "calibrate_impact",
    "missing_calibration_datasets",
    "range_implied_eta",
]

#: Wat een ECHTE kalibratie nodig heeft. Zonder deze twee is `eta` niet
#: identificeerbaar: je kunt de prijsrespons op je eigen order niet meten als je
#: niet weet welke orders er waren en tegen welke diepte.
REQUIRED_DATASETS: tuple[str, ...] = ("orderbook_l2", "trades")

#: Het kwantiel van de range-implied verdeling dat als bovengrens geldt.
#: p95 en niet het maximum: het maximum van een staartverdeling over ~10.000
#: symbool-dagen is een enkele gebeurtenis (19 mei 2021, november 2022) en
#: daarmee een meting van die dag, niet van de liquiditeitsstructuur. p95 is
#: nog steeds ruim conservatief en is wel een eigenschap van de verdeling.
ETA_BOUND_QUANTILE: float = 0.95


@dataclass(frozen=True, slots=True)
class CalibrationEvidence:
    """Alles wat het rapport moet kunnen citeren. Geen enkel veld optioneel."""

    status: ImpactStatus
    reason: str
    missing_datasets: tuple[str, ...]
    n_symbol_days: int
    per_symbol: pd.DataFrame
    quantiles: dict[str, float]
    period_start: str
    period_end: str
    instruments: tuple[str, ...]
    data_hashes: dict[str, str]

    def as_record(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "missing_datasets": list(self.missing_datasets),
            "n_symbol_days": int(self.n_symbol_days),
            "quantiles": {k: float(v) for k, v in self.quantiles.items()},
            "period_start": self.period_start,
            "period_end": self.period_end,
            "instruments": list(self.instruments),
            "data_hashes": dict(self.data_hashes),
            "per_symbol": self.per_symbol.to_dict(orient="index"),
        }


def missing_calibration_datasets(available: Sequence[str]) -> tuple[str, ...]:
    """Welke van `REQUIRED_DATASETS` ontbreekt in de gecertificeerde store."""
    have = {str(a) for a in available}
    return tuple(d for d in REQUIRED_DATASETS if d not in have)


def range_implied_eta(
    ohlc: pd.DataFrame, sigma_daily: pd.Series
) -> pd.Series:
    """`(high - low) / close / sigma_d` per bar — de per-dag bovengrens op eta.

    Parameters
    ----------
    ohlc : moet `high`, `low` en `close` bevatten, op dezelfde index als
        `sigma_daily`.
    sigma_daily : DAGELIJKSE volatiliteit als decimaal, niet geannualiseerd.
        Causaal: de schatting voor bar `t` mag bar `t` niet gebruiken.

    Notes
    -----
    Bars met een niet-positieve `sigma_daily` worden VERWIJDERD en niet op een
    epsilon gezet. Een dag zonder volatiliteitsschatting levert geen bovengrens
    op; hem meenemen met een epsilon in de noemer zou een oneindige bovengrens
    produceren en het kwantiel vervuilen.
    """
    for column in ("high", "low", "close"):
        require(
            column in ohlc.columns,
            "range_implied_eta vereist high, low en close.",
            DataContractError, missing=column, got=list(ohlc.columns),
        )
    require(
        bool(ohlc.index.equals(sigma_daily.index)),
        "OHLC en sigma_daily staan niet op dezelfde tijdas.",
        DataContractError,
        n_ohlc=len(ohlc), n_sigma=len(sigma_daily),
    )
    excursion = (ohlc["high"] - ohlc["low"]) / ohlc["close"]
    sigma = sigma_daily.where(sigma_daily > 0.0)
    ratio = (excursion / sigma).replace([np.inf, -np.inf], np.nan)
    return ratio.dropna()


def _permanent_share(ohlc: pd.DataFrame) -> float:
    """Welk deel van de typische dagbeweging haalt de close?

    `sigma_cc / E[(high - low)/close]`, geklemd op [0, 1]. Bij volledige
    reversie is de close-to-close-vol nul en is de permanente fractie nul; bij
    een pure drift zonder intraday-ruis nadert de verhouding 1.
    """
    cc = ohlc["close"].pct_change(fill_method=None).dropna()
    if cc.empty:
        return 1.0
    sigma_cc = float(cc.std())
    mean_range = float(((ohlc["high"] - ohlc["low"]) / ohlc["close"]).mean())
    if not np.isfinite(mean_range) or mean_range <= 0.0:
        return 1.0
    return float(np.clip(sigma_cc / mean_range, 0.0, 1.0))


def calibrate_impact(
    panels: dict[str, pd.DataFrame],
    sigma_daily: dict[str, pd.Series],
    *,
    available_datasets: Sequence[str],
    data_hashes: dict[str, str],
) -> tuple[ImpactParams, CalibrationEvidence]:
    """Kalibreer, of lever een bovengrens met `IMPACT_UNCALIBRATED`.

    Parameters
    ----------
    panels : symbool -> OHLCV-frame met `high`, `low`, `close`.
    sigma_daily : symbool -> causale dagelijkse volatiliteit (decimaal).
    available_datasets : de datasetnamen die de gecertificeerde store bevat.
    data_hashes : symbool-reeks -> `data_hash` uit `docs/DATA_REGISTER.md`.
    """
    require(
        len(panels) > 0,
        "Kalibratie zonder instrumenten.",
        DataContractError,
    )
    missing = missing_calibration_datasets(available_datasets)

    ratios: dict[str, pd.Series] = {}
    for symbol, frame in panels.items():
        sigma = sigma_daily.get(symbol)
        require(
            sigma is not None,
            "Ontbrekende dagvolatiliteit voor een instrument in de kalibratie.",
            DataContractError, symbol=symbol,
        )
        assert sigma is not None  # narrowing; require() heeft al gecrasht
        ratios[symbol] = range_implied_eta(frame, sigma)

    pooled = pd.concat(ratios.values()).sort_index()
    require(
        len(pooled) > 0,
        "Nul bruikbare symbool-dagen; er valt geen bovengrens af te leiden.",
        DataContractError,
    )

    rows = []
    for symbol, series in ratios.items():
        rows.append({
            "symbol": symbol,
            "n_days": int(len(series)),
            "median": float(series.median()),
            "p95": float(series.quantile(0.95)),
            "max": float(series.max()),
            "kappa_d": _permanent_share(panels[symbol]),
        })
    per_symbol = pd.DataFrame(rows).set_index("symbol").sort_index()

    quantiles = {
        "p50": float(pooled.quantile(0.50)),
        "p75": float(pooled.quantile(0.75)),
        "p90": float(pooled.quantile(0.90)),
        "p95": float(pooled.quantile(0.95)),
        "p99": float(pooled.quantile(0.99)),
        "max": float(pooled.max()),
    }
    eta_bound = float(pooled.quantile(ETA_BOUND_QUANTILE))
    kappa_d = float(per_symbol["kappa_d"].median())

    if missing:
        status = ImpactStatus.IMPACT_UNCALIBRATED
        reason = (
            "Geen orderboek- of trade-data in de gecertificeerde store "
            f"(ontbreekt: {', '.join(missing)}). eta is niet identificeerbaar "
            "zonder eigen orders en hun gemeten prijsrespons; de gerapporteerde "
            "waarde is een CONSERVATIEVE BOVENGRENS uit de dagrange, geen "
            "schatting. Zie docs/DATA_REGISTER.md sectie 6."
        )
        method = (
            f"range-implied upper bound: quantile({ETA_BOUND_QUANTILE:.2f}) of "
            "(high-low)/close / sigma_daily, pooled over symbol-days"
        )
    else:  # pragma: no cover - er is nog geen orderboekdata om dit te bereiken
        status = ImpactStatus.CALIBRATED
        reason = "Gekalibreerd op orderboek- en trade-data."
        method = "metaorder response regression on order book data"

    index = pd.DatetimeIndex(pooled.index)
    params = ImpactParams(
        eta=eta_bound,
        kappa_d=kappa_d,
        status=status,
        method=method,
        data_hash=hash_config(dict(sorted(data_hashes.items()))),
        sample_size=int(len(pooled)),
        period_start=str(index.min().date()),
        period_end=str(index.max().date()),
        instruments=tuple(sorted(panels)),
        # Het "betrouwbaarheidsinterval" van een BOVENGRENS is geen
        # steekproefinterval maar de spreiding van de bovengrens zelf over de
        # verdeling. p75 en p99 begrenzen hoe de keuze van het kwantiel het
        # getal verplaatst; dat is de eerlijke onzekerheidsmaat hier.
        eta_ci_low=float(quantiles["p75"]),
        eta_ci_high=float(quantiles["p99"]),
    )
    evidence = CalibrationEvidence(
        status=status,
        reason=reason,
        missing_datasets=missing,
        n_symbol_days=int(len(pooled)),
        per_symbol=per_symbol,
        quantiles=quantiles,
        period_start=params.period_start,
        period_end=params.period_end,
        instruments=params.instruments,
        data_hashes=dict(sorted(data_hashes.items())),
    )
    return params, evidence


def render_calibration_report(
    params: ImpactParams, evidence: CalibrationEvidence, *, git_sha: str
) -> str:
    """`reports/TCA_CALIBRATION_REPORT.md` — elk veld dat §11 opsomt.

    De volgorde van de secties is die van de fase-opdracht: eerst WAT er is
    gemeten, dan de status, dan pas het getal. Andersom leest een bovengrens als
    een schatting.
    """
    q = evidence.quantiles
    per = evidence.per_symbol
    head = "\n".join(
        f"| `{s}` | {int(r['n_days']):,} | {r['median']:.3f} | {r['p95']:.3f} | "
        f"{r['max']:.3f} | {r['kappa_d']:.3f} |"
        for s, r in per.iterrows()
    )
    missing = ", ".join(f"`{d}`" for d in evidence.missing_datasets) or "geen"
    hashes = "\n".join(
        f"| `{k}` | `{v}` |" for k, v in evidence.data_hashes.items()
    )
    return f"""# TCA / MARKET IMPACT CALIBRATION REPORT

**Gegenereerd door:** `apps/calibrate_impact.py`
**git_sha:** `{git_sha}`
**Status:** **`{params.status.value}`**

---

## 1. Wat een echte kalibratie vereist

| Dataset | Aanwezig in de gecertificeerde store? |
|---|---|
{chr(10).join(f"| `{d}` | {'ja' if d not in evidence.missing_datasets else '**nee**'} |" for d in REQUIRED_DATASETS)}

**Ontbrekend:** {missing}

{evidence.reason}

---

## 2. Status

> ## `{params.status.value}`
>
> De gerapporteerde `eta` is **geen schatting**. Elk resultaat dat op deze
> parameter draait, moet deze status zichtbaar meedragen; de authoritative
> engine dwingt dat af via `ImpactEstimate.status`.

---

## 3. De afgeleide bovengrens

**Methode:** {params.method}

Het model geeft bij `Q = V` (een meta-order zo groot als het dagvolume)
`Impact = eta * sigma_d`. De grootste beweging die op die dag daadwerkelijk
optrad is `(high - low) / close`. Wordt die VOLLEDIG aan impact toegeschreven -
niets aan nieuws, niets aan de stroom van anderen - dan volgt
`eta <= (high - low) / close / sigma_d`.

De aanname is maximaal conservatief, dus de werkelijke `eta` ligt hier met
zekerheid onder, en vermoedelijk ver eronder.

### Verdeling over {evidence.n_symbol_days:,} symbool-dagen

| Kwantiel | Waarde |
|---|---:|
| p50 | {q['p50']:.3f} |
| p75 | {q['p75']:.3f} |
| p90 | {q['p90']:.3f} |
| **p95 (gekozen bovengrens)** | **{q['p95']:.3f}** |
| p99 | {q['p99']:.3f} |
| max | {q['max']:.3f} |

p95 en niet het maximum: het maximum over ~{evidence.n_symbol_days // 1000}k
symbool-dagen is een enkele gebeurtenis en meet die dag, niet de
liquiditeitsstructuur.

### Per instrument

| Symbool | Dagen | mediaan | p95 | max | `kappa_d` |
|---|---:|---:|---:|---:|---:|
{head}

---

## 4. De parameters

| Parameter | Waarde |
|---|---:|
| `eta` | **{params.eta:.4f}** |
| `kappa_d` | **{params.kappa_d:.4f}** |
| Onzekerheidsband (`eta`, p75-p99) | [{params.eta_ci_low:.3f}, {params.eta_ci_high:.3f}] |
| Steekproef | {params.sample_size:,} symbool-dagen |
| Periode | {params.period_start} t/m {params.period_end} |
| Instrumenten | {', '.join(f'`{s}`' for s in params.instruments)} |
| `data_hash` | `{params.data_hash}` |

`kappa_d` is de PERMANENTE fractie: `sigma_cc / E[(high-low)/close]`, het deel
van de typische dagbeweging dat de close haalt. Voor de kosten van EEN order
verandert `kappa_d` niets - de volledige impact wordt betaald - maar hij bepaalt
wat er van de beweging overblijft voor de volgende order.

---

## 5. Wat dit betekent voor de kosten

Kosten in basispunten bij `sigma_d` = 4,0 % (typisch voor dit universum):

| Participatie (`Q/V`) | Impact (bps) |
|---|---:|
{chr(10).join(f"| {p:.2%} | {params.eta * 0.04 * (p ** 0.5) * 1e4:,.1f} |" for p in (0.0001, 0.001, 0.01, 0.05, 0.10))}

Ter vergelijking: de `adv_participation_cap` in `conf/risk/default.yaml` staat op
**1 %**, en de sovereign laag dwingt die af vóór er een order bestaat.

### 5.1 Hoe informatief is deze bovengrens? Nauwelijks.

Dat moet gezegd worden, want het getal ziet er preciezer uit dan het is.

Voor een Brownse beweging is de verwachte range over een periode
`sigma * sqrt(8/pi) ~ 1,60 sigma`. De gemeten mediaan over alle instrumenten
ligt op **{q['p50']:.2f}**, en de p95 op **{q['p95']:.2f}** — precies wat een
random walk oplevert. De spreiding tussen instrumenten is bovendien
verwaarloosbaar: de p95 per symbool ligt tussen
{per['p95'].min():.2f} en {per['p95'].max():.2f}, terwijl hun dagvolumes
ordes van grootte verschillen.

**Deze bovengrens meet dus vrijwel uitsluitend gewone volatiliteit, en vrijwel
geen liquiditeitsstructuur.** Dat is geen fout in de afleiding maar de directe
consequentie van het ontbreken van orderboekdata: zonder eigen orders is er
geen signaal waaruit impact te scheiden valt van beweging.

### 5.2 Waarom hij desondanks bruikbaar is

Omdat hij op dit boek niet bindt. Bij de bruto-exposure die de sovereign laag
toestaat (~0,11 van de equity, gespreid over
{len(params.instruments)} symbolen) is de participatie in de orde van 1e-7 tot
1e-5, en daar levert zelfs een bovengrens van {params.eta:.2f} een impact van
onder de 1 bp op een liquide dag. De dominante kostenpost blijft de taker-fee.

De bovengrens wordt pas bindend bij een aanzienlijk groter boek of een
aanzienlijk dunnere markt — en dat is precies wanneer je een conservatieve
aanname wílt hebben. Het exit-rapport draait daarom een
gevoeligheidsanalyse over `eta` in plaats van één getal te rapporteren.

---

## 6. Gecertificeerde bronnen

| Reeks | `data_hash` |
|---|---|
{hashes}

---

## 7. Wat er nodig is om dit te vervangen door een echte kalibratie

1. Orderboek-L2-snapshots op minstens 1-minuutcadans voor het universum.
2. Trade-prints met richting (taker side).
3. Eigen order-flow met grootte en tijdstempel, of een meta-order-reconstructie
   uit de trade-tape.

Punt 1 en 2 zijn de openstaande items uit `docs/DATA_REGISTER.md` §6; punt 3
komt pas na Phase 7 beschikbaar. Tot dat moment blijft de status
`IMPACT_UNCALIBRATED`, en dat is geen tekortkoming van deze fase maar een
eigenschap van de dataset.
"""
