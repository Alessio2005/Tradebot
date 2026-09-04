# MASTER-PROMPT: PHASE 10 — HERSTART OP DAGBARS, VOLATILITEIT ALS TOESTAND

> **Fase:** 10, volgt op `fase_9_opschoning_en_consolidatie.md` · **Prioriteit:** P0
> **Brainstormgrondslag:** `brainstormen.md` (Phase 0 — repository understanding & quant architecture brainstorm)
> **Bindende brondocumenten:** `docs/PROJECT_STATE.md`, `docs/FALSIFICATION_REGISTER.md`, `docs/ARCHITECTURAL_DECISIONS.md` (AD-13 t/m AD-21), `docs/DEFERRED_ISSUES.md`, `reports/phase5_exit_report.md` §10, `reports/GARCH_VS_EWMA_COMPETITION.md` §2.3 en §9, `reports/M0_VS_HMM_BENCHMARK.md` §10
> **Voorwaarde:** deze fase voert drie mandaatbesluiten uit die de eigenaar heeft genomen. Zij mag geen vierde besluit nemen.
>
> **Voor uitvoerders:** stappen gebruiken checkbox-syntaxis (`- [ ]`). Elke stap eindigt in een commit. Werk de stappen in volgorde af; Stage A blokkeert alles.

---

## ROL EN CONTEXT

Je acteert als **Quant Research Engineer & Platform Architect**. Je erft een repository van **290 modules / 71.025 LOC in `src/`**, **145 testbestanden / 29.151 LOC**, tien DVC-stages en negen afgeronde fasen.

Het platform is gebouwd rond één principe: *een resultaat telt pas wanneer het de poort is gepasseerd die het had kunnen tegenhouden.* Negen fasen hebben nul modellen gepromoveerd en twintig hypothesen gefalsifieerd. Dat is geen mislukking maar de opbrengst: de repository weet nu wat er níet werkt, en dat weet zij met bewijs.

**De regel die deze fase regeert:**

> **Deze fase voegt geen voorspeller toe. Zij verandert de meetbasis en repareert één architectuurfout — en zij doet allebei zo dat een negatieve uitkomst een geldig, goedkoop antwoord is.**

**Relevante lagen (Target Architecture §19):** L2 (volatiliteit), L3 (features), L4 (alpha), L7 (risk), L10 (backtest), L11 (validatie), L12 (governance).

---

## DE CORRECTIE DIE DEZE FASE DRAAGT

Dit hoofdstuk gaat vóór alle andere, want het herschrijft wat de vorige fasen met volatiliteitsmodellen deden.

### Wat er nu gebeurt

Volatiliteitsmodellen worden in deze repository op twee manieren gebruikt, en **beide zijn de verkeerde**:

1. **Als puntvoorspeller van σ², beoordeeld op forecast-nauwkeurigheid.** Dat is de H1-campagne: 48 gefitte combinaties, beoordeeld met QLIKE tegen een variantieproxy. Die opzet vraagt een proxy, en de proxy is precies waar H1 op is vastgelopen (`mean(proxy)/mean(r²)` = 1,27–2,24).
2. **Als schaalvermenigvuldiger op de exposure.** Dat is AD-15: `a_geconditioneerd[t] = a_basis[t] × (1 − p_hoog[t])`, waarbij `p_hoog` uit M0, M1 of M2 komt.

`regime/conditioning.py::_high_state` kiest de toestand op **variantie** en gooit de gefitte `means` weg — `regime/markov.py` schat ze wel (`means: np.ndarray  # (k, d)`, regel 161), maar niets leest ze.

### Waarom (2) niet kán werken — dit is gemeten, niet beredeneerd

AD-16 heeft het al vastgesteld en het is de scherpste meting in het hele regimespoor:

> De soevereine laag schaalt het boek met `w_t = min(max_leverage, σ_target / σ_boek)`. Vermenigvuldig elke exposure met dezelfde `c`, dan deelt `w_t` er weer door. **Gemeten:** elke exposure halveren verplaatste de gemiddelde bruto notional van 8.300 naar 8.283.

Daaruit volgt een ontwerpregel die nergens als zodanig is opgeschreven, en die vanaf deze fase bindend is:

> **REGEL V.** Een toestand die uitsluitend de **omvang** van het boek raakt, is per constructie redundant. L7 herschaalt al op σ̂, en de toestand is een grovere versie van datzelfde getal. Een volatiliteitstoestand kan alleen informatie toevoegen wanneer zij de **samenstelling** raakt — welke posities er zijn en met welk teken — want dat is precies wat vol-targeting niet doet.

### Wat een volatiliteitsmodel wél hoort te doen

Een volatiliteitsmodel bepaalt de volatiliteit en kent op grond daarvan een **discrete toestand** toe. Die toestand conditioneert een **besluit**, niet een schaal:

```
vol-model  →  σ̂_t  →  toestandstoewijzing  →  s_t ∈ {0, 1, 2}  →  BESLUIT
```

Dat heeft drie gevolgen die deze fase uitwerkt:

* **De evaluatie verandert.** Een toestandstoewijzer wordt niet met QLIKE beoordeeld maar op drie meetbare eigenschappen: *separatie* (verschillen de toestanden in de grootheid die er toe doet?), *bezetting* (heeft elke toestand genoeg bars?) en *economische waarde* (verandert het besluit, en wordt de netto uitkomst beter?). **Geen daarvan vraagt een variantieproxy.** De blokkade die H1 heeft geveld, bestaat op dit spoor niet.
* **De vergelijking tussen vol-modellen wordt goedkoop.** De vraag is niet meer "voorspelt GARCH σ² beter dan EWMA" — die is zonder intraday-RV onbeslisbaar (§B-2) — maar "wijst GARCH een ándere toestand toe dan EWMA, en verandert dat het besluit?" Dat is een telling, geen toets. Zie stap 8.
* **De afbeelding moet REGEL V respecteren.** Een `(1 − p_hoog)`-vermenigvuldiging doet dat niet. Zie stap 7.

### En wat de toestand hier wél en niet blijkt te dragen

Gemeten met de bestaande `regime/buckets.py::classify_vol_buckets` op de gecertificeerde PIT-store, drempels uit `conf/model/regime.yaml`, 10.400 symbool-bars:

| Toestand | n | aandeel | ann. rendement | **ann. volatiliteit** | t-stat |
|---|---:|---:|---:|---:|---:|
| LAAG | 1.720 | 16,5 % | +49,6 % | **62,6 %** | 1,72 |
| NORMAAL | 8.181 | 78,7 % | +9,5 % | **79,2 %** | 0,57 |
| HOOG | 499 | 4,8 % | +136,4 % | **137,0 %** | 1,16 |

Lees de tabel op twee manieren, want zij zegt twee verschillende dingen.

**De toestand scheidt volatiliteit uitstekend.** 62,6 % tegen 137,0 % is een factor 2,2, en dat is wat een volatiliteitsmodel hoort te kunnen. Hier is niets mis mee.

**De toestand scheidt richting niet.** Alle drie de t-statistieken liggen onder 2, en de tekens spreken elkaar per symbool tegen: DOTUSDT doet +131,6 bp/dag in HOOG, SOLUSDT −45,1 bp. Op deze data is er **geen bewijs** dat een vol-toestand richtingsinformatie draagt, en een `0 = bearish / 2 = bullish`-lezing wordt door dit paneel niet gedragen. De toestand wordt daarom geïdentificeerd op **variantie** — de grootheid die betrouwbaar wordt geschat — en haar semantiek wordt **gemeten en gerapporteerd**, nooit aangenomen.

**En één bevinding die de huidige bedrading rechtstreeks raakt:** AD-15 de-grost op `p_hoog`, dus het verlaagt de exposure het sterkst in de toestand met het **hoogste** gemeten gemiddelde rendement. Dat is niet significant, maar het is ook niet het teken dat de afbeelding veronderstelt. De afbeelding codeert een aanname over richting die niemand heeft gemeten.

---

## DE DRIE MANDAATBESLUITEN

Genomen door de eigenaar. Deze fase voert ze uit en toetst ze niet.

| | Besluit | Wat het sluit |
|---|---|---|
| **B-1** | **Barresolutie: dagbars.** De handelsklok volgt de meetklok. `live/feed.py::FeedConfig.bar_seconds = 5` vervalt. | De factor **131,5 op sigma** tussen de twee ketens; het verschil 5 vs. 6 namen; exit-criterium D7 wordt voor het eerst *gedefinieerd* |
| **B-2** | **Geen intraday alpha.** Geen order flow, orderboek-imbalance, tick-runs, volume-clock bars, intraday seizoenaliteit, 5m realized variance. | H1's heropeningsconditie, DI-18, HAR-RV, `bars/`, F2 en F19 |
| **B-3** | **De ledger wordt gereset.** `M = 2776` is niet langer de trial-teller voor dit programma. | De rekenkundige onbereikbaarheid van de promotiepoort |

---

## NULMETING (gemeten 2026-09-04 tegen `5ae4a00` — reproduceer vóór je iets wijzigt)

Deze getallen zijn de basis waartegen deze fase wordt afgerekend. Wijkt jouw meting af, dan is dát je eerste bevinding en die schrijf je op vóór je verdergaat.

### De vier muren

| Grootheid | Gemeten | Bron |
|---|---:|---|
| Effectieve breedte N_eff, 6 namen | **1,271** (ρ̄ = 0,7442) | eigen meting op `data/pit_store/`, formule `alpha/cm_carry.py:291` |
| N_eff per jaar | 1,40 · 1,20 · 1,41 · 1,34 · **1,19** · **1,17** | idem, 2021–2026 |
| Ann. Sharpe nodig voor t = 2 over 4,78 jaar | **0,915** | `t = SR·√T` |
| Ann. Sharpe nodig voor DSR ≥ 0,95 bij M = 2776, N = 1615 | **2,47** | `backtest/metrics.py::deflated_sharpe` |
| Beste gemeten track (L0, `long_only_equal_weight`) | **0,156** | `artefacts/baseline/phase3_baseline.json` |

### De vier-lagen-ladder (netto Sharpe)

| Track | L0 vectorized | L1 + risk | L2 + latency | L3 + execution |
|---|---:|---:|---:|---:|
| `xs_momentum_equal_weight` | +0,098 | +0,215 | −0,052 | −0,103 |
| `xs_momentum_risk_parity` | −0,227 | −0,005 | −0,263 | −0,331 |
| `long_only_equal_weight` | +0,039 | +0,119 | +0,136 | −0,695 |
| `long_only_risk_parity` | +0,012 | +0,120 | +0,130 | −0,721 |

`long_only_equal_weight` halteert op 2022-05-10 bij 8,31 % drawdown: **1.559 van 1.743 bars gehalteerd**. Bron: `artefacts/baseline/phase5_revaluation.json`.

### De vol-toestand

Zie de tabel in **DE CORRECTIE** hierboven. Reproduceerbaar met `regime/buckets.py` en `conf/model/regime.yaml`.

### De gedragsvingerafdruk

`python -m pytest -q` → **exact 4 failures**, alle vier pre-geregistreerde killgates op `cm_carry` en `cm_tsmom`. Minder dan vier betekent dat een killgate is uitgeschakeld, niet dat er iets is opgelost.

### De DSR-drempel als functie van M (N = 1615, dagbars)

| M | 2776 | 500 | 250 | 100 | 50 | **25** | 12 | 6 | 4 | 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vereiste ann. Sharpe | 2,47 | 2,24 | 2,14 | 1,99 | 1,87 | **1,74** | 1,58 | 1,40 | 1,28 | 1,03 |

### Wat een zoektocht uit RUIS oplevert (SE ann. Sharpe = 1/√4,78 = 0,458)

| M cellen | 2 | 4 | 6 | 12 | 25 | 55 | 100 | 2776 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E[max Sharpe] | 0,24 | 0,48 | 0,60 | 0,76 | 0,91 | **1,06** | 1,16 | 1,62 |

**Dit is tijdens de brainstorm empirisch bevestigd.** Een diagnostisch grid van 11 formatievensters × 5 houdlagen = **55 cellen** op de authoritative unit leverde een hoogste bruto Sharpe van **+1,075** op, tegen een verwacht ruismaximum van **+1,058**. De bijbehorende IC's lagen tussen −0,024 en +0,050 bij een standaardfout van 0,011. **Er zat niets in, en toch produceerde een halve middag verkennen een getal boven de DSR-eis bij M = 2.**

---

## FENCES — WAT DEZE FASE NIET AANRAAKT

1. **De vijf promotiepoorten.** `validation/gates.py` krijgt geen `force=`, geen `override=`, geen `warn_only=`, geen deelscore. Eén invoerwaarde (`M`) verandert; geen poort.
2. **`FALSIFICATION_REGISTER.md` F1 t/m F20.** Alleen `heropening uitgesloten door mandaat` mag worden toegevoegd. Geen regel wordt gewijzigd, verzwakt of verwijderd.
3. **De lookahead-suite.** 628 tests, `tests/lookahead/`. Onaangeraakt.
4. **De dubbele boekhouding en de engine.** `backtest/accounting.py`, `backtest/engine.py`. Onaangeraakt behalve de toevoeging in stap 10.
5. **`alpha/cm_carry.py` en `alpha/cm_tsmom.py`.** Dragen pre-geregistreerde killgates.
6. **De haltketen.** `risk/kill_switches.py`, `live/circuit_breaker.py`, `monitoring/sharpe_monitor.py`, `monitoring/execution_drift.py`. Resolutie-onafhankelijk en bewezen; blijft ook wanneer stap 15 `live/` vervangt.
7. **De PIT-store en zijn hashes.** Read-only.

---

## CONCRETE DELIVERABLES

| # | Deliverable | Stap |
|---|---|---|
| D1 | AD-22, AD-23, AD-24 in `docs/ARCHITECTURAL_DECISIONS.md` | 1 |
| D2 | `scripts/check_mandate_consistency.py` + CI-poort | 2 |
| D3 | `artefacts/governance/ledger_reset.json` + bevroren `M_new` | 3 |
| D4 | `registry/trial_budget.py` + poort op overschrijding | 4 |
| D5 | `regime/state.py` — het `VolState`-contract | 5 |
| D6 | `reports/phase10_state_diagnostics.md` + artefact | 6 |
| D7 | `regime/state_mapping.py` — de afbeelding die REGEL V respecteert | 7 |
| D8 | `reports/phase10_state_agreement.md` — EWMA vs. GARCH, proxyvrij | 8 |
| D9 | Gerealiseerde-bezettingspoort in `validation/data_adequacy.py` | 9 |
| D10 | Gepaarde standaardfouten in `backtest/phase5_baseline.py` | 10 |
| D11 | Pre-registratie + uitkomst H-10.1 (beslisfrequentie) | 11 |
| D12 | Pre-registratie + uitkomst H-10.2 (ongebalanceerd paneel) | 12 |
| D13 | Pre-registratie + uitkomst H-10.3 (toestandsconditionering) | 13 |
| D14 | `reports/phase10_chain_a_score.md` — keten A gescoord, niet-promoveerbaar | 14 |
| D15 | `reports/phase10_intraday_surface.md` — verdict per module | 15 |
| D16 | Besluit + uitvoering `live/` (vervangen of aansluiten) | 16 |
| D17 | Harde omvangslimiet in `conf/risk/default.yaml` | 17 |
| D18 | `reports/phase10_exit_report.md` | slot |

---

## GLOBALE RANDVOORWAARDEN

Deze gelden impliciet bij **elke** stap hieronder.

* **Python** uit `docs/RUNBOOK.md` par. 0 (`D:\venv\tradebot\Scripts\python.exe`). Meten op een andere interpreter is in Phase 9 zeven keer een foutmeting gebleken.
* **`ruff` is gepind op 0.15.12** (`requirements-dev.lock`, DI-16). Niet upgraden in deze fase.
* **R-1** Causaliteit is absoluut: een feature voor bar `t` gebruikt uitsluitend data ≤ `t−1`.
* **R-4** 800 LOC per bestand, afgedwongen door `scripts/check_file_size.py` met een cap per bestand. Een nieuw bestand boven de cap is rood.
* **R-5** Determinisme: gelijke `cfg + seed` ⇒ bit-identieke output.
* **R-6** Apps zijn stateless en ≤ 80 LOC; alle logica in `src/tradebot/`.
* **Elke drempel komt uit `conf/`**, nooit als literal in een handtekening. Bewaakt door `scripts/check_hardcoded_params.py`.
* **Elke nieuwe variant is een trial.** Ook een verkenning. Zie stap 4.
* **Na elke stap:** `python -m pytest -q` toont **exact 4 failures**. Een vijfde is een regressie; een derde is een uitgeschakelde killgate.

---

## STAPSGEWIJZE UITVOERING

# STAGE A — MANDAAT EN LEDGER

*Blokkeert alles. Zonder A is elk resultaat uit B en C ongeldig.*

---

### Stap 1: De drie mandaatbesluiten vastleggen

**Files:**
- Modify: `docs/ARCHITECTURAL_DECISIONS.md` (append AD-22, AD-23, AD-24)
- Modify: `docs/PROJECT_STATE.md` §5 (besluiten 1–3 verwijderen, verwijzen naar AD-22/23/24)

**Interfaces:**
- Produces: de identifiers `AD-22`, `AD-23`, `AD-24`, waarnaar stap 2, 14, 15 en 16 verwijzen.

- [ ] **Stap 1.1 — Reproduceer de nulmeting.** Draai:

```bash
D:/venv/tradebot/Scripts/python.exe -m pytest -q
D:/venv/tradebot/Scripts/python.exe apps/run_phase5_baseline.py
```

Verwacht: exact 4 failures; `artefacts/baseline/phase5_revaluation.json` reproduceert de laddertabel uit de NULMETING. Wijkt het af, **stop en rapporteer**.

- [ ] **Stap 1.2 — Schrijf AD-22 (barresolutie).** Volg de structuur van AD-15: `Fase / Status / Bewaakt door`, dan `Besluit`, `Waarom`, `Het afgewezen alternatief`. Het besluit: dagbars als meet- én handelsresolutie; `bar_seconds = 5` vervalt. Het afgewezen alternatief: de twee resoluties naast elkaar houden en een adapter bouwen — afgewezen omdat een pariteitstest over twee resoluties de configuratie meet in plaats van het gedrag.

- [ ] **Stap 1.3 — Schrijf AD-23 (geen intraday alpha).** Neem de lijst uitgesloten informatiebronnen letterlijk op: order flow, orderboek-imbalance, tick-runs, volume-clock bars, intraday seizoenaliteit, 5m realized variance. Vermeld expliciet dat dit de informatiebron sluit en niet één toepassing ervan.

- [ ] **Stap 1.4 — Schrijf AD-24 (ledger-reset).** Neem de zeven regels R1–R7 uit stap 3 integraal op. Vermeld de gemeten grond: `seed_total = 2363` is een reconstructie uit een verloren logboek en volgens `registry/trial_counter.py` een ondergrens.

- [ ] **Stap 1.5 — Commit.**

```bash
git add docs/ARCHITECTURAL_DECISIONS.md docs/PROJECT_STATE.md
git commit -m "docs(mandate): AD-22 dagbars, AD-23 geen intraday alpha, AD-24 ledger-reset"
```

---

### Stap 2: De gesloten sporen, met een poort die het afdwingt

De registers dragen nu heropeningscondities die onder AD-23 nooit kunnen intreden. Een conditie die niet kan intreden en toch als "heropenbaar" te boek staat, nodigt uit tot een hertest die het mandaat verbiedt.

**Files:**
- Create: `scripts/check_mandate_consistency.py`
- Create: `tests/unit/test_mandate_consistency.py`
- Modify: `docs/FALSIFICATION_REGISTER.md` (H1-sectie; F2- en F19-regels)
- Modify: `docs/DEFERRED_ISSUES.md` (DI-18 naar Gesloten)
- Modify: `.github/workflows/inventory.yml`

**Interfaces:**
- Produces: `check_mandate_consistency.main(argv) -> int` (0 = schoon, 1 = inconsistent), en `INTRADAY_MARKERS: tuple[str, ...]`.

- [ ] **Stap 2.1 — Schrijf de falende test.**

```python
# tests/unit/test_mandate_consistency.py
"""AD-23 sluit een informatiebron. Een register dat nog een heropening op die
bron belooft, is daarmee intern tegenstrijdig -- en dat is precies de
tegenstrijdigheid waaruit later een verboden hertest ontstaat."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_mandate_consistency.py"


def test_scanner_reports_clean_repository() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--strict"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_scanner_goes_red_on_an_unmarked_intraday_reopening(tmp_path: Path) -> None:
    """De negatieve controle. Zonder deze test toetst de poort haar eigen vorm."""
    register = tmp_path / "FALSIFICATION_REGISTER.md"
    register.write_text(
        "| F99 | verzonnen unit | bewijs | heropening zodra er 5m realized "
        "variance is |\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--strict", "--register", str(register)],
        capture_output=True, text=True, cwd=REPO,
    )
    assert result.returncode == 1
    assert "F99" in result.stdout
```

- [ ] **Stap 2.2 — Draai en zie hem falen.**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_mandate_consistency.py -q`
Verwacht: FAIL — het script bestaat niet.

- [ ] **Stap 2.3 — Schrijf de scanner.**

```python
# scripts/check_mandate_consistency.py
"""AD-23-poort: elke heropeningsconditie die intraday-data vereist, moet als
door het mandaat uitgesloten gemarkeerd staan.

Rood worden is de bedoeling. Een poort die niet rood kan worden, bewijst niets;
`tests/unit/test_mandate_consistency.py` toont dat deze het wel kan.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Woorden die een intraday-informatiebron aanduiden. Bewust ruim: een vals
#: positief kost een expliciete markering, een vals negatief kost een hertest
#: die het mandaat verbiedt.
INTRADAY_MARKERS: tuple[str, ...] = (
    "5m", "intraday", "realized variance", "maker-execution", "maker execution",
    "orderboek", "order flow", "tick", "L2-data", "volume-clock",
)

#: De markering die een regel expliciet onder AD-23 plaatst.
EXCLUDED = "heropening uitgesloten door mandaat"

ROW = re.compile(r"^\|\s*(F\d+|H\d+|DI-\d+)\s*\|")


def scan(register: Path) -> list[str]:
    offenders: list[str] = []
    for line in register.read_text(encoding="utf-8").splitlines():
        match = ROW.match(line)
        if not match:
            continue
        lowered = line.lower()
        if EXCLUDED in lowered:
            continue
        if any(marker.lower() in lowered for marker in INTRADAY_MARKERS):
            offenders.append(match.group(1))
    return offenders


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--register", type=Path,
        default=REPO / "docs" / "FALSIFICATION_REGISTER.md",
    )
    args = parser.parse_args(argv)

    offenders = scan(args.register)
    for identifier in offenders:
        print(f"{identifier}: heropeningsconditie vereist intraday-data en is "
              f"niet gemarkeerd als uitgesloten door AD-23")
    if offenders and args.strict:
        return 1
    print(f"mandaatconsistentie: {len(offenders)} openstaande regel(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Stap 2.4 — Markeer F2 en F19 in het register.** Voeg aan beide regels toe: `heropening uitgesloten door mandaat (AD-23)`. **Wijzig geen bestaand bewijs en geen bestaande formulering** — append binnen de kolom.

- [ ] **Stap 2.5 — Sluit H1 met §3.1-bewijs.** Voeg aan de H1-sectie van `FALSIFICATION_REGISTER.md` een subsectie toe met deze tabel uit `reports/GARCH_VS_EWMA_COMPETITION.md` §9:

| Symbool | h | Variant | p (rogers_satchell) | p (garman_klass) | p (parkinson) | **p (squared_return)** |
|---|---|---|---:|---:|---:|---:|
| AVAXUSDT | 1 | `garch(1,1)-t` | 0,0076 | 0,0047 | 0,0104 | **0,517** |
| AVAXUSDT | 1 | `gjr_garch(1,1,1)-t` | 0,0075 | 0,0070 | 0,0202 | **0,752** |
| AVAXUSDT | 1 | `egarch(1,1,1)-t` | 0,0106 | 0,0079 | 0,0200 | **0,514** |
| AVAXUSDT | 5 | `garch(1,1)-t` | 0,0121 | 0,0116 | 0,0264 | **0,368** |

Met deze tekst eronder, letterlijk:

> Op de drie range-proxies lijkt het verschil significant; op `squared_return` — de enige proxy die per constructie zuiver is voor de voorspelde grootheid `E[r_t²|F_{t-1}] = σ_t²` — verdampt het volledig. Onder AD-23 komt de intraday realized variance die dit zou oplossen er nooit. **H1 is daarmee gesloten en niet geblokkeerd**, en EWMA(0,94) is de productie-estimator bij besluit. Dit is expliciet geen bewering dat GARCH slechter is; het is de constatering dat het verschil op dagdata niet meetbaar is. De vraag "wijst GARCH een andere TOESTAND toe" is een andere vraag en staat open — zie stap 8.

- [ ] **Stap 2.6 — Verplaats DI-18 naar `## Gesloten`** met AD-23 als grondslag. Doe hetzelfde met het contract van `volatility/har_rv.py` in `docs/CODE_REGISTER.md` klasse D: de grondslag `DI-18` wordt `AD-23 — geen afnemer meer`.

- [ ] **Stap 2.7 — Voeg de poort toe aan CI.** In `.github/workflows/inventory.yml`, naast de bestaande poorten:

```yaml
      - name: Mandaatconsistentie (AD-23)
        run: python scripts/check_mandate_consistency.py --strict
```

- [ ] **Stap 2.8 — Draai en zie hem slagen.**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_mandate_consistency.py -q`
Verwacht: 2 passed.

- [ ] **Stap 2.9 — Commit.**

```bash
git add scripts/check_mandate_consistency.py tests/unit/test_mandate_consistency.py \
        docs/FALSIFICATION_REGISTER.md docs/DEFERRED_ISSUES.md docs/CODE_REGISTER.md \
        .github/workflows/inventory.yml
git commit -m "governance(AD-23): sluit H1, DI-18, F2 en F19; poort op mandaatconsistentie"
```

---

### Stap 3: De ledger-reset

De gevaarlijkste stap in deze fase. Hij verlaagt een drempel, en dat is precies het soort handeling waar deze repository haar hele apparaat tegen heeft opgetuigd.

**Het protocol — zeven regels, integraal in AD-24:**

> **R1** `M_new` is niet nul. Het is het vooraf geregistreerde, bevroren aantal geplande trials.
> **R2** `FALSIFICATION_REGISTER.md` blijft onverkort bindend, **en dat is de prijs van de reset.** De kennis uit de oude 2776 trials lekt wél door — wie weet dat crypto-XS-momentum faalt, kiest een andere kandidaat dan wie dat niet weet. Dat is selectiedruk en zij verdwijnt niet met de teller. Het register is het geheugen dat haar neutraliseert.
> **R3** De oude ledger wordt gearchiveerd, niet verwijderd. Een AD-14-amendement (`n_trials = 0`) legt de reset vast.
> **R4** Wie een oude fit hergebruikt, erft zijn trials. Het CPCV-ensemble draagt **2.400** Optuna-trials (200 × 6 symbolen × 2 zijden).
> **R5** Eén reset. Een tweede maakt `M` een parameter in plaats van een meting.
> **R6** `M_new` wordt bevroren vóór de eerste fit.
> **R7** De vijf poorten blijven ongewijzigd.

**Files:**
- Create: `src/tradebot/registry/ledger_reset.py`
- Create: `tests/unit/test_ledger_reset.py`
- Create: `apps/freeze_ledger_reset.py`
- Create: `artefacts/governance/ledger_reset.json` (output)
- Modify: `src/tradebot/registry/trial_counter.py`

**Interfaces:**
- Consumes: `trial_counter.TrialCount`, `trial_counter.M_UNCERTAINTY_NOTE`
- Produces: `ledger_reset.freeze_reset(*, m_new: int, rationale: str, git_sha: str, out: Path, ledger_path: Path | None = None) -> ResetRecord`; `ledger_reset.active_trial_count(*, reset_path: Path, ledger_path: Path) -> ActiveCount`; `ledger_reset.ResetAlreadyExists`

- [ ] **Stap 3.1 — Schrijf de falende test.** Let op de derde: dat is de negatieve controle die bewijst dat de reset géén blanco cheque is.

```python
# tests/unit/test_ledger_reset.py
"""De reset verlaagt een drempel. Deze tests bewijzen dat hij dat precies een
keer doet, dat hij de oude telling niet wist, en dat de verlaagde drempel nog
steeds weigert wat hij hoort te weigeren."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tradebot.backtest.metrics import deflated_sharpe
from tradebot.registry.ledger_reset import (
    ResetAlreadyExists,
    active_trial_count,
    freeze_reset,
)
from tradebot.utils.failfast import DataContractError

LEDGER = Path("artefacts/governance/hypothesis_ledger.json")


def test_reset_is_frozen_and_carries_its_grounds(tmp_path: Path) -> None:
    out = tmp_path / "ledger_reset.json"
    record = freeze_reset(
        m_new=25,
        rationale="AD-24; nieuw programma op dagbars, kandidaatverzameling "
                  "disjunct van F1-F20",
        git_sha="deadbee",
        out=out,
    )
    assert record.m_new == 25
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["m_new"] == 25
    assert payload["git_sha"] == "deadbee"
    assert payload["rationale"]
    assert payload["frozen_utc"]


def test_a_second_reset_is_refused(tmp_path: Path) -> None:
    """R5. Twee resets maken M een parameter in plaats van een meting."""
    out = tmp_path / "ledger_reset.json"
    freeze_reset(m_new=25, rationale="eerste", git_sha="deadbee", out=out)
    with pytest.raises(ResetAlreadyExists):
        freeze_reset(m_new=4, rationale="tweede", git_sha="deadbee", out=out)


def test_the_old_ledger_stays_readable(tmp_path: Path) -> None:
    """R3. De reset archiveert; hij wist niet."""
    out = tmp_path / "ledger_reset.json"
    freeze_reset(m_new=25, rationale="r", git_sha="deadbee", out=out)
    count = active_trial_count(reset_path=out, ledger_path=LEDGER)
    assert count.total == 25
    assert count.archived_total == 2776


def test_the_reset_does_not_make_the_gate_permissive() -> None:
    """De negatieve controle op de reset zelf.

    Bij M_new = 25 is de DSR-eis 1,74 geannualiseerd. Het beste dat deze
    repository ooit heeft gemeten is 0,156. Een reset die 0,156 zou doorlaten,
    zou geen reset zijn maar een uitschakeling.
    """
    per_bar_best_measured = 0.156 / (365 ** 0.5)
    assert deflated_sharpe(per_bar_best_measured, 25, 1615) < 0.95


def test_m_new_must_be_positive(tmp_path: Path) -> None:
    """R1. Een programma zonder trials heeft geen kandidaten."""
    with pytest.raises(DataContractError):
        freeze_reset(
            m_new=0, rationale="r", git_sha="deadbee",
            out=tmp_path / "ledger_reset.json",
        )
```

- [ ] **Stap 3.2 — Draai en zie hem falen.**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_ledger_reset.py -q`
Verwacht: FAIL — `ModuleNotFoundError: tradebot.registry.ledger_reset`.

- [ ] **Stap 3.3 — Implementeer de module.**

```python
# src/tradebot/registry/ledger_reset.py
"""De eenmalige ledger-reset — AD-24.

WAAROM DIT EEN EIGEN MODULE IS
==============================
`trial_counter.py` telt trials. Deze module doet iets anders en gevaarlijkers:
hij verlaagt een drempel. Dat hoort niet verstopt te zitten in de teller, want
dan is de handeling niet meer zichtbaar in een diff.

DE VEILIGHEID ZIT NIET IN DE REKENKUNDE
=======================================
Bij `M_new = 25` staat de DSR-eis op 1,74 geannualiseerd. Dat is hoger dan het
VERWACHTE maximum van 2.776 pure ruistrekkingen (1,62) -- maar de kans dat dat
maximum boven 1,74 uitkomt is **18 %**. Een ongeregistreerde zoektocht van die
omvang haalt dus ongeveer een op de zes keer een resultaat dat deze poort
passeert.

De reset is daarom uitsluitend geldig zolang `M_new` het WERKELIJKE aantal
geprobeerde varianten telt, en die belofte wordt niet door dit bestand gedragen
maar door de pre-registratie, het bevriezen vóór de eerste fit, en het
falsificatieregister (R2).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..utils.failfast import DataContractError, require

__all__ = ["ResetAlreadyExists", "ResetRecord", "active_trial_count", "freeze_reset"]


class ResetAlreadyExists(RuntimeError):
    """R5 -- er is er precies een."""


@dataclass(frozen=True)
class ResetRecord:
    m_new: int
    rationale: str
    git_sha: str
    frozen_utc: str
    archived_total: int


@dataclass(frozen=True)
class ActiveCount:
    total: int
    archived_total: int


def _archived_total(ledger_path: Path) -> int:
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    return int(payload["seed_total"]) + sum(
        int(entry["n_trials"]) for entry in payload["entries"]
    )


def freeze_reset(
    *, m_new: int, rationale: str, git_sha: str, out: Path,
    ledger_path: Path | None = None,
) -> ResetRecord:
    """Bevries de reset. Onherhaalbaar; het bestand is het slot."""
    require(
        m_new > 0,
        "M_new moet positief zijn. Een programma zonder geplande trials heeft "
        "geen kandidaten, en een DSR met M = 0 is geen correctie (R1).",
        DataContractError, m_new=m_new,
    )
    require(
        bool(rationale.strip()),
        "Een reset zonder grondslag is niet auditbaar (R3).",
        DataContractError,
    )
    if out.exists():
        raise ResetAlreadyExists(
            f"{out} bestaat al. R5: er is precies EEN reset. Een tweede maakt "
            f"M een parameter in plaats van een meting."
        )
    ledger = ledger_path or Path("artefacts/governance/hypothesis_ledger.json")
    archived = _archived_total(ledger)
    record = ResetRecord(
        m_new=int(m_new),
        rationale=rationale,
        git_sha=git_sha,
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        archived_total=archived,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "m_new": record.m_new,
                "rationale": record.rationale,
                "git_sha": record.git_sha,
                "frozen_utc": record.frozen_utc,
                "archived_total": record.archived_total,
                "protocol": "AD-24 R1-R7",
                "note": (
                    "De oude telling is GEARCHIVEERD, niet gewist. "
                    "FALSIFICATION_REGISTER.md F1-F20 blijft onverkort bindend; "
                    "dat is de prijs van deze reset (R2)."
                ),
            },
            indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )
    return record


def active_trial_count(*, reset_path: Path, ledger_path: Path) -> ActiveCount:
    """De `M` waarmee wordt gemeten na de reset, plus wat er is gearchiveerd."""
    require(
        reset_path.exists(),
        "Geen bevroren reset gevonden; meten met een impliciete M is precies "
        "de vrijheidsgraad die de DSR hoort weg te nemen (R6).",
        DataContractError, path=str(reset_path),
    )
    payload = json.loads(reset_path.read_text(encoding="utf-8"))
    return ActiveCount(
        total=int(payload["m_new"]),
        archived_total=int(payload["archived_total"]),
    )
```

- [ ] **Stap 3.4 — Draai en zie hem slagen.**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_ledger_reset.py -q`
Verwacht: 5 passed.

- [ ] **Stap 3.5 — Schrijf de app (≤ 80 LOC, R-6).**

```python
# apps/freeze_ledger_reset.py
"""Bevries de eenmalige ledger-reset van AD-24. Draai dit EEN keer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tradebot.registry.ledger_reset import ResetAlreadyExists, freeze_reset  # noqa: E402
from tradebot.registry.lineage import get_git_sha  # noqa: E402

OUT = Path("artefacts/governance/ledger_reset.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-new", type=int, required=True)
    parser.add_argument("--rationale", type=str, required=True)
    args = parser.parse_args()
    try:
        record = freeze_reset(
            m_new=args.m_new, rationale=args.rationale,
            git_sha=get_git_sha(short=True), out=OUT,
        )
    except ResetAlreadyExists as exc:
        print(f"GEWEIGERD: {exc}")
        return 1
    print(f"M_new = {record.m_new} bevroren op {record.frozen_utc}; "
          f"gearchiveerd: {record.archived_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Stap 3.6 — Voer de reset uit.** `M_new = 25`, met de begroting uit stap 4 als grond.

```bash
D:/venv/tradebot/Scripts/python.exe apps/freeze_ledger_reset.py \
  --m-new 25 \
  --rationale "AD-24: herstart op dagbars (AD-22) met een kandidaatverzameling die per AD-23 disjunct is van de intraday-ruimte waarin het merendeel van de 2776 trials is uitgevoerd. F1-F20 blijft onverkort bindend (R2)."
```

- [ ] **Stap 3.7 — Voeg het AD-14-amendement toe aan de ledger** (`apps/ledger_append.py`, `n_trials = 0`, `amends` verwijst naar de reset). Controleer daarna dat `hypothesis_ledger.json` integraal leesbaar blijft.

- [ ] **Stap 3.8 — Commit.**

```bash
git add src/tradebot/registry/ledger_reset.py tests/unit/test_ledger_reset.py \
        apps/freeze_ledger_reset.py artefacts/governance/ledger_reset.json \
        artefacts/governance/hypothesis_ledger.json
git commit -m "governance(AD-24): eenmalige ledger-reset naar M_new=25, met negatieve controle"
```

---

### Stap 4: Het trial-budget als poort

Bij `M_new = 25` kost elke verdubbeling van `M` ongeveer 0,13–0,17 Sharpe aan drempelhoogte. Dat maakt trials een schaars goed, en een schaars goed hoort een teller met een limiet te hebben.

**Files:**
- Create: `src/tradebot/registry/trial_budget.py`
- Create: `tests/unit/test_trial_budget.py`
- Modify: `.github/workflows/inventory.yml`

**Interfaces:**
- Consumes: `ledger_reset.active_trial_count`
- Produces: `trial_budget.assert_within_budget(planned: int, *, reset_path: Path) -> None`; `trial_budget.remaining(*, reset_path: Path, ledger_path: Path, booked: int) -> int`

- [ ] **Stap 4.1 — Schrijf de falende test.**

```python
# tests/unit/test_trial_budget.py
"""Het budget is de enige rem op de manoeuvre uit stap 3."""
from __future__ import annotations

from pathlib import Path

import pytest

from tradebot.registry.ledger_reset import freeze_reset
from tradebot.registry.trial_budget import assert_within_budget, remaining
from tradebot.utils.failfast import DataContractError

LEDGER = Path("artefacts/governance/hypothesis_ledger.json")


def _reset(tmp_path: Path, m_new: int = 25) -> Path:
    out = tmp_path / "ledger_reset.json"
    freeze_reset(m_new=m_new, rationale="test", git_sha="deadbee", out=out)
    return out


def test_a_plan_inside_the_budget_passes(tmp_path: Path) -> None:
    assert_within_budget(6, reset_path=_reset(tmp_path))


def test_a_plan_over_the_budget_crashes(tmp_path: Path) -> None:
    with pytest.raises(DataContractError):
        assert_within_budget(26, reset_path=_reset(tmp_path))


def test_inheriting_optuna_trials_blows_the_budget(tmp_path: Path) -> None:
    """R4, becijferd. Het CPCV-ensemble draagt 200 x 6 x 2 = 2400 trials.
    Wie het hergebruikt, tilt M van 25 naar 2425 en de DSR-eis van 1,74 naar
    2,46 -- en dat hoort te crashen in plaats van stilzwijgend door te gaan."""
    with pytest.raises(DataContractError):
        assert_within_budget(2400, reset_path=_reset(tmp_path))


def test_remaining_counts_down(tmp_path: Path) -> None:
    path = _reset(tmp_path)
    assert remaining(reset_path=path, ledger_path=LEDGER, booked=0) == 25
    assert remaining(reset_path=path, ledger_path=LEDGER, booked=6) == 19
```

- [ ] **Stap 4.2 — Draai en zie hem falen.**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_trial_budget.py -q`
Verwacht: FAIL — `ModuleNotFoundError`.

- [ ] **Stap 4.3 — Implementeer.**

```python
# src/tradebot/registry/trial_budget.py
"""Het trial-budget -- de rem op AD-24.

Bij M_new = 25 kost elke VERDUBBELING van M ongeveer 0,13 tot 0,17 Sharpe aan
drempelhoogte:

    M      25    50    100   250   500   2776
    eis   1,74  1,87  1,99  2,14  2,24  2,47

Dat is de wisselkoers waarin een onderzoeksplan zich hoort uit te drukken. Deze
module maakt die koers afdwingbaar in plaats van adviserend.
"""
from __future__ import annotations

from pathlib import Path

from ..utils.failfast import DataContractError, require
from .ledger_reset import active_trial_count

__all__ = ["assert_within_budget", "remaining"]


def assert_within_budget(planned: int, *, reset_path: Path) -> None:
    """Crash wanneer een gepland aantal trials het bevroren budget overschrijdt."""
    budget = active_trial_count(
        reset_path=reset_path,
        ledger_path=Path("artefacts/governance/hypothesis_ledger.json"),
    ).total
    require(
        planned <= budget,
        "Het geplande aantal trials overschrijdt het bevroren budget. Elke "
        "verdubbeling van M kost ~0,15 Sharpe aan drempelhoogte; een plan dat "
        "over het budget gaat, verhoogt zijn eigen lat. Herzie het plan, of "
        "leg een nieuw budget vast met een eigen grondslag.",
        DataContractError, planned=planned, budget=budget,
    )


def remaining(*, reset_path: Path, ledger_path: Path, booked: int) -> int:
    """Wat er van het budget over is na `booked` geboekte trials."""
    budget = active_trial_count(
        reset_path=reset_path, ledger_path=ledger_path
    ).total
    return budget - int(booked)
```

- [ ] **Stap 4.4 — Draai en zie hem slagen.**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_trial_budget.py -q`
Verwacht: 4 passed.

- [ ] **Stap 4.5 — Commit.**

```bash
git add src/tradebot/registry/trial_budget.py tests/unit/test_trial_budget.py
git commit -m "governance(AD-24): trial-budget met een poort op overschrijding"
```

---

# STAGE B — DE VOLATILITEITSTOESTAND

*De correctie, uitgevoerd. Elke stap hier is diagnostisch tenzij anders vermeld: er wordt niets uit geselecteerd en niets uit gepromoveerd.*

---

### Stap 5: Het `VolState`-contract

Er bestaat nu geen type voor "toestand". `regime/buckets.py` levert een `VolBucket(IntEnum)`, `regime/markov.py` levert kansen, `alpha/macro_regime.py` levert een `RegimeLabel`-enum, en `conditioning.py` platwalst alles tot één float. Dat is de reden dat de toestand nergens als toestand kan worden getoetst.

**Files:**
- Create: `src/tradebot/regime/state.py`
- Create: `tests/unit/test_vol_state.py`
- Create: `tests/lookahead/test_vol_state_causality.py`

**Interfaces:**
**Verhouding tot het bestaande `VolBucket`.** `regime/buckets.py::VolBucket` is M0's SPECIFIEKE classificatie (twee assen, conjunctie op z-score en ATR-ratio). `VolState` is het GENERIEKE contract waar elke vol-toestandstoewijzer op uitkomt — M0, EWMA-kwantielen, een HMM. M0 blijft ongewijzigd en krijgt in stap 6 een adapter naar `VolState`; er wordt geen tweede M0 gebouwd.

- Produces: `state.VolState` (IntEnum: `LOW=0, NORMAL=1, HIGH=2`), `state.StateAssignment` (frozen dataclass met `states: pd.DataFrame`, `source: str`, `ordering: str`, `data_hashes: tuple[tuple[str, str], ...]`), en `state.assign_by_variance(sigma: pd.DataFrame, *, low_q: float, high_q: float, min_periods: int) -> StateAssignment`

- [ ] **Stap 5.1 — Schrijf de falende test.**

```python
# tests/unit/test_vol_state.py
"""Het toestandscontract. Drie eigenschappen, elk met een eigen falsificatie:
de ordening is op variantie en niet op gemiddelde; de toestand is discreet; en
de toewijzing draagt haar herkomst mee."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.regime.state import StateAssignment, VolState, assign_by_variance
from tradebot.utils.failfast import DataContractError

IDX = pd.date_range("2022-01-01", periods=400, freq="D", tz="UTC")


def _sigma(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"A": np.abs(rng.normal(0.5, 0.15, 400)),
         "B": np.abs(rng.normal(0.8, 0.25, 400))},
        index=IDX,
    )


def test_states_are_ordered_by_variance_not_by_mean() -> None:
    """HIGH is de toestand met de HOOGSTE sigma. Ordenen op gemiddeld rendement
    zou een richtingsclaim zijn, en die wordt op dit paneel niet gedragen:
    gemeten t-statistieken 0,57 / 1,16 / 1,72, tekens per symbool tegenstrijdig."""
    sigma = _sigma()
    result = assign_by_variance(sigma, low_q=0.25, high_q=0.75, min_periods=100)
    for column in sigma.columns:
        states = result.states[column]
        mask = states.notna()
        high = sigma[column][mask][states[mask] == VolState.HIGH]
        low = sigma[column][mask][states[mask] == VolState.LOW]
        assert high.mean() > low.mean()
    assert result.ordering == "variance"


def test_states_are_discrete() -> None:
    result = assign_by_variance(_sigma(), low_q=0.25, high_q=0.75, min_periods=100)
    values = result.states.to_numpy()
    finite = values[np.isfinite(values)]
    assert set(np.unique(finite)) <= {0.0, 1.0, 2.0}


def test_burn_in_is_nan_never_a_default_state() -> None:
    """Een toestand tijdens de opstartfase is een INGEVULDE toestand, en die
    zou als NORMAAL worden gelezen -- een besluit dat niemand heeft genomen."""
    result = assign_by_variance(_sigma(), low_q=0.25, high_q=0.75, min_periods=100)
    head = result.states.iloc[:99]
    assert head.isna().all().all()


def test_assignment_carries_its_source() -> None:
    result = assign_by_variance(
        _sigma(), low_q=0.25, high_q=0.75, min_periods=100, source="ewma_0.94",
    )
    assert result.source == "ewma_0.94"
    assert isinstance(result, StateAssignment)


def test_quantiles_must_be_ordered() -> None:
    with pytest.raises(DataContractError):
        assign_by_variance(_sigma(), low_q=0.75, high_q=0.25, min_periods=100)
```

```python
# tests/lookahead/test_vol_state_causality.py
"""R-1 op de toestand. De kwantielen zijn EXPANDING; een toestand op bar t mag
niet veranderen door bars na t toe te voegen."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.regime.state import assign_by_variance

IDX = pd.date_range("2022-01-01", periods=500, freq="D", tz="UTC")


def test_truncation_cannot_change_an_earlier_state() -> None:
    rng = np.random.default_rng(7)
    sigma = pd.DataFrame(
        {"A": np.abs(rng.normal(0.5, 0.2, 500))}, index=IDX,
    )
    full = assign_by_variance(sigma, low_q=0.25, high_q=0.75, min_periods=100)
    for cut in (200, 300, 400):
        truncated = assign_by_variance(
            sigma.iloc[:cut], low_q=0.25, high_q=0.75, min_periods=100,
        )
        pd.testing.assert_frame_equal(
            truncated.states, full.states.iloc[:cut], check_freq=False,
        )
```

- [ ] **Stap 5.2 — Draai beide en zie ze falen.**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_vol_state.py tests/lookahead/test_vol_state_causality.py -q`
Verwacht: FAIL — `ModuleNotFoundError: tradebot.regime.state`.

- [ ] **Stap 5.3 — Implementeer.**

```python
# src/tradebot/regime/state.py
"""Het VolState-contract -- de discrete toestand die een BESLUIT conditioneert.

WAAROM DIT BESTAAT
==================
Volatiliteitsmodellen in deze repository werden op twee manieren gebruikt en
beide zijn de verkeerde: als puntvoorspeller van sigma^2 beoordeeld met QLIKE
(H1, vastgelopen op de proxy), en als schaalvermenigvuldiger op de exposure
(AD-15). Dat tweede kan per constructie niet werken -- AD-16 meet dat L7 elke
uniforme schaal er weer uitdeelt.

Een volatiliteitsmodel bepaalt de volatiliteit en kent op grond daarvan een
DISCRETE TOESTAND toe. Die toestand conditioneert een besluit, niet een schaal.
Zie REGEL V in `Prompts-fases/fase_10_herstart_dagbars.md`.

DE ORDENING IS OP VARIANTIE, EN DAT IS EEN BESLUIT
===================================================
Een k-toestands-HMM schat per toestand zowel een gemiddelde als een variantie
(`regime/markov.py`, `means: (k, d)`). De verleiding is om te ordenen op het
GEMIDDELDE -- dat levert 0 = bearish, 1 = flat, 2 = bullish.

Op dit paneel wordt die lezing niet gedragen. Gemeten met de bestaande M0
vol-buckets over 10.400 symbool-bars:

    toestand   n      ann. rendement   ann. volatiliteit   t
    LAAG       1720   +49,6 %          62,6 %              1,72
    NORMAAL    8181   + 9,5 %          79,2 %              0,57
    HOOG        499   +136,4 %         137,0 %             1,16

De volatiliteit scheidt met een factor 2,2 en dat is betrouwbaar. Het rendement
scheidt niet: geen enkele t-statistiek haalt 2, en de tekens spreken elkaar per
symbool tegen (DOTUSDT +131,6 bp/dag in HOOG, SOLUSDT -45,1 bp).

Daarom: identificatie op VARIANTIE -- de grootheid die betrouwbaar wordt
geschat -- en de semantiek van de toestand wordt GEMETEN en gerapporteerd
(stap 6), nooit aangenomen.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd

from ..utils.failfast import DataContractError, require
from ..utils.time import assert_utc_index

__all__ = ["StateAssignment", "VolState", "assign_by_variance"]


class VolState(IntEnum):
    """De drie toestanden, geordend op volatiliteitsniveau."""

    LOW = 0
    NORMAL = 1
    HIGH = 2


@dataclass(frozen=True)
class StateAssignment:
    """Een toestandstoewijzing plus de herkomst die haar auditbaar maakt."""

    states: pd.DataFrame
    source: str
    ordering: str
    params: Mapping[str, float]

    def occupancy(self) -> pd.DataFrame:
        """Aandeel bars per toestand per symbool -- de invoer van de poort in stap 9."""
        rows = {}
        for column in self.states.columns:
            series = self.states[column].dropna()
            total = max(len(series), 1)
            rows[column] = {
                state.name: float((series == int(state)).sum()) / total
                for state in VolState
            }
        return pd.DataFrame(rows).T


def assign_by_variance(
    sigma: pd.DataFrame,
    *,
    low_q: float,
    high_q: float,
    min_periods: int,
    source: str = "unspecified",
) -> StateAssignment:
    """Wijs elke bar een VolState toe op EXPANDING kwantielen van sigma.

    Expanding en niet rolling, en al helemaal niet over de volledige sample:
    dat laatste is DI-2 en het zou de toestand van bar `t` laten afhangen van
    bars die op `t` nog niet bestonden.
    """
    assert_utc_index(sigma, name="assign_by_variance")
    require(
        0.0 < low_q < high_q < 1.0,
        "De kwantielen moeten oplopen en binnen (0, 1) liggen; anders kan een "
        "bar tegelijk LOW en HIGH zijn en bepaalt de volgorde van twee if-takken "
        "de uitkomst.",
        DataContractError, low_q=low_q, high_q=high_q,
    )
    require(
        min_periods > 1,
        "Een kwantiel over minder dan twee observaties bestaat niet.",
        DataContractError, min_periods=min_periods,
    )

    states = pd.DataFrame(
        np.nan, index=sigma.index, columns=sigma.columns, dtype="float64"
    )
    for column in sigma.columns:
        series = sigma[column].astype("float64")
        lo = series.expanding(min_periods=min_periods).quantile(low_q)
        hi = series.expanding(min_periods=min_periods).quantile(high_q)
        assigned = pd.Series(np.nan, index=series.index, dtype="float64")
        valid = series.notna() & lo.notna() & hi.notna()
        assigned[valid] = float(VolState.NORMAL)
        assigned[valid & (series <= lo)] = float(VolState.LOW)
        assigned[valid & (series >= hi)] = float(VolState.HIGH)
        states[column] = assigned

    return StateAssignment(
        states=states,
        source=source,
        ordering="variance",
        params={"low_q": float(low_q), "high_q": float(high_q),
                "min_periods": float(min_periods)},
    )
```

- [ ] **Stap 5.4 — Voeg de drempels toe aan `conf/model/regime.yaml`** onder een nieuwe sleutel `state:` (`low_q: 0.25`, `high_q: 0.75`, `min_periods: 250`) en breid `schemas/config.py` uit met een `VolStateConfig` (`extra="forbid"`, `frozen=True`). Geen enkel getal blijft als literal in een handtekening staan.

- [ ] **Stap 5.5 — Draai en zie ze slagen.**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_vol_state.py tests/lookahead/test_vol_state_causality.py -q`
Verwacht: 6 passed.

- [ ] **Stap 5.6 — Commit.**

```bash
git add src/tradebot/regime/state.py tests/unit/test_vol_state.py \
        tests/lookahead/test_vol_state_causality.py conf/model/regime.yaml \
        src/tradebot/schemas/config.py
git commit -m "feat(regime): VolState-contract -- discrete toestand, geordend op variantie"
```

---

### Stap 6: De toestandsdiagnose — wat draagt de toestand?

Diagnostiek, expliciet niet-promoveerbaar. Er wordt niets uit geselecteerd, dus deze stap kost **nul trials**.

**Files:**
- Create: `src/tradebot/regime/state_diagnostics.py`
- Create: `apps/run_state_diagnostics.py`
- Create: `tests/unit/test_state_diagnostics.py`
- Create: `reports/phase10_state_diagnostics.md` (output)
- Create: `artefacts/governance/phase10_state_diagnostics.json` (output)
- Modify: `dvc.yaml`

**Interfaces:**
- Consumes: `state.StateAssignment`, `state.VolState`
- Produces: `state_diagnostics.diagnose(assignment, forward_returns) -> StateDiagnostics` met velden `separation_vol`, `separation_return`, `occupancy`, `mean_duration`, `n_transitions`

- [ ] **Stap 6.1 — Schrijf de falende test.**

```python
# tests/unit/test_state_diagnostics.py
"""De diagnose moet BEIDE assen rapporteren, en zij moet de richtingsas met
haar onzekerheid rapporteren -- anders leest iemand een verschil van 1,2 SE als
een bevinding."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.regime.state import assign_by_variance
from tradebot.regime.state_diagnostics import diagnose

IDX = pd.date_range("2022-01-01", periods=600, freq="D", tz="UTC")


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(11)
    sigma = pd.DataFrame({"A": np.abs(rng.normal(0.5, 0.2, 600))}, index=IDX)
    returns = pd.DataFrame(
        {"A": rng.normal(0.0, 1.0, 600) * sigma["A"].to_numpy() / np.sqrt(365)},
        index=IDX,
    )
    return sigma, returns


def test_reports_both_axes() -> None:
    sigma, returns = _fixture()
    result = diagnose(
        assign_by_variance(sigma, low_q=0.25, high_q=0.75, min_periods=100),
        forward_returns=returns,
    )
    assert set(result.separation_vol) == {"LOW", "NORMAL", "HIGH"}
    assert set(result.separation_return) == {"LOW", "NORMAL", "HIGH"}


def test_the_direction_axis_carries_a_t_statistic() -> None:
    """Zonder t-statistiek is een conditioneel gemiddelde geen bevinding maar
    een getal. Dit is het veld dat voorkomt dat iemand +136 %/jr leest zonder
    de bijbehorende 1,16 te zien."""
    sigma, returns = _fixture()
    result = diagnose(
        assign_by_variance(sigma, low_q=0.25, high_q=0.75, min_periods=100),
        forward_returns=returns,
    )
    for state in ("LOW", "NORMAL", "HIGH"):
        assert "t_stat" in result.separation_return[state]
        assert "n" in result.separation_return[state]


def test_synthetic_returns_with_no_state_dependence_show_no_separation() -> None:
    """Negatieve controle. Op returns die per constructie NIET van de toestand
    afhangen, moet geen enkele t-statistiek boven 2 uitkomen."""
    sigma, returns = _fixture()
    result = diagnose(
        assign_by_variance(sigma, low_q=0.25, high_q=0.75, min_periods=100),
        forward_returns=returns,
    )
    assert all(
        abs(result.separation_return[s]["t_stat"]) < 2.0
        for s in ("LOW", "NORMAL", "HIGH")
    )
```

- [ ] **Stap 6.2 — Draai en zie hem falen.** Verwacht: `ModuleNotFoundError`.

- [ ] **Stap 6.3 — Implementeer `state_diagnostics.py`.** Rapporteer per toestand: `n`, `occupancy`, `ann_vol`, `ann_return`, `t_stat`, `mean_duration_bars`. En per symbool apart, want de nulmeting toont dat de tekens per symbool tegenstrijdig zijn en een gepoolde tabel dat verbergt.

- [ ] **Stap 6.4 — Schrijf de app en de DVC-stage.** `apps/run_state_diagnostics.py` (≤ 80 LOC) leest de PIT-store, bouwt σ̂ met `volatility/ewma.py::ewma_volatility_panel`, wijst toe met `assign_by_variance` en schrijft beide outputs. Voeg de stage toe aan `dvc.yaml` met `data/pit_store`, `conf/model/regime.yaml` en `conf/model/volatility.yaml` als deps.

- [ ] **Stap 6.5 — Draai en vergelijk met de nulmeting.**

```bash
D:/venv/tradebot/Scripts/python.exe apps/run_state_diagnostics.py
```

Verwacht: de vol-separatie reproduceert de orde van grootte uit de NULMETING (62,6 % / 79,2 % / 137,0 %) en geen enkele richtings-t haalt 2. **Wijkt dat af, dan is dát de bevinding** en die gaat vóór alle volgende stappen.

- [ ] **Stap 6.6 — Schrijf `reports/phase10_state_diagnostics.md`.** Verplichte secties: de twee assen naast elkaar; de per-symbool-tabel; en een paragraaf **"Wat deze diagnose NIET vaststelt"** waarin staat dat geen enkele richtingsclaim wordt gedragen en dat de toestand op variantie is geïdentificeerd.

- [ ] **Stap 6.7 — Commit.**

```bash
git add src/tradebot/regime/state_diagnostics.py apps/run_state_diagnostics.py \
        tests/unit/test_state_diagnostics.py reports/phase10_state_diagnostics.md \
        artefacts/governance/phase10_state_diagnostics.json dvc.yaml
git commit -m "feat(regime): toestandsdiagnose -- vol-separatie meetbaar, richting niet"
```

---

### Stap 7: De afbeelding die REGEL V respecteert

**Files:**
- Create: `src/tradebot/regime/state_mapping.py`
- Create: `tests/unit/test_state_mapping.py`
- Modify: `docs/ARCHITECTURAL_DECISIONS.md` (AD-25)

**Interfaces:**
- Consumes: `state.StateAssignment`, `state.VolState`
- Produces: `state_mapping.gate_by_state(exposures: pd.DataFrame, assignment: StateAssignment, *, flat_states: frozenset[VolState]) -> pd.DataFrame`

- [ ] **Stap 7.1 — Schrijf de falende test.** De eerste is de belangrijkste van deze hele fase: hij bewijst dat de afbeelding L7 overleeft.

```python
# tests/unit/test_state_mapping.py
"""REGEL V, afdwingbaar gemaakt.

AD-16 meet dat een uniforme schaal onzichtbaar is: L7 herschaalt naar het
vol-target en deelt elke constante c er weer uit. Een toestandsafbeelding die
alleen de OMVANG raakt, is daarmee per constructie een lege operatie. Deze
tests bewijzen dat `gate_by_state` de SAMENSTELLING raakt en niet de schaal."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.regime.state import VolState, assign_by_variance
from tradebot.regime.state_mapping import gate_by_state
from tradebot.utils.failfast import DataContractError

IDX = pd.date_range("2022-01-01", periods=400, freq="D", tz="UTC")


def _fixture():
    rng = np.random.default_rng(3)
    sigma = pd.DataFrame(
        {"A": np.abs(rng.normal(0.4, 0.1, 400)),
         "B": np.abs(rng.normal(0.9, 0.3, 400))},
        index=IDX,
    )
    exposures = pd.DataFrame(
        {"A": rng.uniform(-1, 1, 400), "B": rng.uniform(-1, 1, 400)}, index=IDX,
    )
    assignment = assign_by_variance(
        sigma, low_q=0.25, high_q=0.75, min_periods=100,
    )
    return exposures, assignment


def test_the_mapping_survives_vol_targeting() -> None:
    """De kerntest. Normaliseer beide boeken op gelijke bruto exposure -- de
    operatie die L7 uitvoert -- en zij MOETEN dan nog verschillen. Een
    schaalafbeelding zou hier identiek worden."""
    exposures, assignment = _fixture()
    gated = gate_by_state(
        exposures, assignment, flat_states=frozenset({VolState.HIGH}),
    )

    def gross_normalised(frame: pd.DataFrame) -> pd.DataFrame:
        gross = frame.abs().sum(axis=1).replace(0.0, np.nan)
        return frame.div(gross, axis=0)

    base = gross_normalised(exposures)
    conditioned = gross_normalised(gated)
    common = base.dropna().index.intersection(conditioned.dropna().index)
    assert not np.allclose(
        base.loc[common].to_numpy(), conditioned.loc[common].to_numpy(),
        equal_nan=True,
    )


def test_a_gated_state_produces_exactly_zero_exposure() -> None:
    exposures, assignment = _fixture()
    gated = gate_by_state(
        exposures, assignment, flat_states=frozenset({VolState.HIGH}),
    )
    for column in exposures.columns:
        high = assignment.states[column] == float(VolState.HIGH)
        assert (gated[column][high].fillna(0.0) == 0.0).all()


def test_an_ungated_state_is_passed_through_untouched() -> None:
    """Geen dempingsfactor, geen herschaling: de view of geen view."""
    exposures, assignment = _fixture()
    gated = gate_by_state(
        exposures, assignment, flat_states=frozenset({VolState.HIGH}),
    )
    for column in exposures.columns:
        keep = assignment.states[column].isin(
            [float(VolState.LOW), float(VolState.NORMAL)]
        )
        pd.testing.assert_series_equal(
            gated[column][keep], exposures[column][keep], check_names=False,
        )


def test_gating_every_state_is_refused() -> None:
    """Een afbeelding die alles dichtzet, is geen conditioneerder maar een
    uit-knop, en zij zou een lege reeks als 'resultaat' opleveren."""
    exposures, assignment = _fixture()
    with pytest.raises(DataContractError):
        gate_by_state(
            exposures, assignment,
            flat_states=frozenset({VolState.LOW, VolState.NORMAL, VolState.HIGH}),
        )


def test_burn_in_stays_nan_and_is_not_gated_to_zero() -> None:
    """Geen toestand is geen besluit. Nul zou 'wij kiezen vlak' betekenen."""
    exposures, assignment = _fixture()
    gated = gate_by_state(
        exposures, assignment, flat_states=frozenset({VolState.HIGH}),
    )
    unknown = assignment.states.isna()
    assert gated.where(unknown).isna().all().all()
```

- [ ] **Stap 7.2 — Draai en zie hem falen.** Verwacht: `ModuleNotFoundError`.

- [ ] **Stap 7.3 — Implementeer `state_mapping.py`.** De afbeelding is een **poort**, niet een factor:

```
s_t in flat_states   ->  0        (geen view)
s_t elders           ->  a_t      (volle view, ongeschaald)
s_t onbekend         ->  NaN      (geen besluit)
```

Nul vrije parameters zodra `flat_states` is geregistreerd. Geen multiplier per toestand — dat zou drie extra parameters zijn en het experiment in een sizing-experiment veranderen (het alternatief dat AD-15 al afwees).

- [ ] **Stap 7.4 — Schrijf AD-25.** Titel: *"Een toestand conditioneert de samenstelling, nooit de schaal"*. Besluit: REGEL V. Waarom: AD-16's meting. Het afgewezen alternatief: `a × (1 − p_hoog)` — afgewezen omdat AD-16 meet dat L7 die factor er weer uitdeelt, dus de afbeelding meet niets. Bewaakt door `tests/unit/test_state_mapping.py::test_the_mapping_survives_vol_targeting`.

- [ ] **Stap 7.5 — Draai en zie hem slagen.**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_state_mapping.py -q`
Verwacht: 5 passed.

- [ ] **Stap 7.6 — Commit.**

```bash
git add src/tradebot/regime/state_mapping.py tests/unit/test_state_mapping.py \
        docs/ARCHITECTURAL_DECISIONS.md
git commit -m "feat(regime): AD-25 -- toestandspoort op samenstelling; bewijs dat zij L7 overleeft"
```

---

### Stap 8: EWMA tegen GARCH, zonder proxy

H1 vroeg welk model σ² béter voorspelt en had daarvoor een proxy nodig. De toestandsvraag heeft er geen nodig: **wijzen de twee modellen een andere toestand toe, en verandert dat het besluit?** Dat is een telling.

Diagnostiek, niet-promoveerbaar. **Nul trials** — er wordt geen kandidaat geselecteerd.

**Files:**
- Create: `src/tradebot/regime/state_agreement.py`
- Create: `apps/run_state_agreement.py`
- Create: `tests/unit/test_state_agreement.py`
- Create: `reports/phase10_state_agreement.md` (output)

**Interfaces:**
- Consumes: `state.StateAssignment`, `volatility/ewma.py::ewma_volatility_panel`, `volatility/garch.py`
- Produces: `state_agreement.agreement(a: StateAssignment, b: StateAssignment) -> AgreementReport` met `fraction_identical`, `confusion` (3×3 per symbool), `n_decision_changes`

- [ ] **Stap 8.1 — Schrijf de falende test.**

```python
# tests/unit/test_state_agreement.py
"""De proxyvrije vergelijking. Een toestandstoewijzer wordt vergeleken met een
andere toestandstoewijzer, en dat is een telling in plaats van een toets."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.regime.state import assign_by_variance
from tradebot.regime.state_agreement import agreement

IDX = pd.date_range("2022-01-01", periods=400, freq="D", tz="UTC")


def _assignment(seed: int):
    rng = np.random.default_rng(seed)
    sigma = pd.DataFrame({"A": np.abs(rng.normal(0.5, 0.2, 400))}, index=IDX)
    return assign_by_variance(sigma, low_q=0.25, high_q=0.75, min_periods=100)


def test_a_model_agrees_perfectly_with_itself() -> None:
    a = _assignment(5)
    report = agreement(a, a)
    assert report.fraction_identical == 1.0
    assert report.n_decision_changes == 0


def test_different_sigma_paths_disagree_somewhere() -> None:
    report = agreement(_assignment(5), _assignment(6))
    assert report.fraction_identical < 1.0


def test_confusion_matrix_is_three_by_three_per_symbol() -> None:
    report = agreement(_assignment(5), _assignment(6))
    assert report.confusion["A"].shape == (3, 3)
```

- [ ] **Stap 8.2 — Draai en zie hem falen.**

- [ ] **Stap 8.3 — Implementeer `state_agreement.py`.**

- [ ] **Stap 8.4 — Schrijf de app.** Bouw twee σ̂-panelen — EWMA(0,94) uit `volatility/ewma.py` en de walk-forward GARCH(1,1)-t uit `volatility/garch.py`, met de bestaande foldstructuur — wijs beide toe met `assign_by_variance` op dezelfde drempels, en rapporteer.

- [ ] **Stap 8.5 — Draai en schrijf het rapport.** `reports/phase10_state_agreement.md` moet één vraag expliciet beantwoorden:

> **Wanneer de twee modellen op ≥ 95 % van de bars dezelfde toestand toewijzen, is de keuze tussen hen economisch irrelevant en is H1 ook op de toestandsas beslecht — zonder proxy, zonder DM-toets en zonder trial.**

Blijft de overeenstemming daar onder, dan is het verschil een **kandidaat** en gaat het als hypothese naar Stage C, met de trial-kosten die daarbij horen.

- [ ] **Stap 8.6 — Commit.**

```bash
git add src/tradebot/regime/state_agreement.py apps/run_state_agreement.py \
        tests/unit/test_state_agreement.py reports/phase10_state_agreement.md
git commit -m "feat(regime): proxyvrije EWMA/GARCH-vergelijking op de toestandsas"
```

---

### Stap 9: De gerealiseerde-bezettingspoort

H2 kreeg van de adequaatheidspoort *a priori* groen licht en liep daarna vast op de gerealiseerde bezetting. Gemeten in `artefacts/governance/phase6_h2_regime_benchmark.json`:

| | a priori | gerealiseerd |
|---|---|---|
| `hmm_k3`, zeldzaamste toestand | 165,0 obs per fold — `adequate: true` | **5,0 tot 28,0** obs per fold |
| bron van de aanname | `"uniform (1/k) — optimistisch"` | de fit zelf |

De poort keurde goed op een aanname die het artefact zelf **"optimistisch"** noemt. Dat is een defect in de poort, niet in het model.

**Files:**
- Modify: `src/tradebot/validation/data_adequacy.py`
- Create: `tests/unit/test_realised_occupancy_gate.py`
- Modify: `conf/model/adequacy.yaml`

**Interfaces:**
- Produces: `data_adequacy.assert_realised_occupancy(assignment: StateAssignment, *, n_folds: int, min_obs_per_state_per_fold: int, min_occupancy_fraction: float, raise_on_failure: bool = True) -> OccupancyVerdict` met `.adequate: bool` en `.measured: Mapping[str, Any]`

- [ ] **Stap 9.1 — Schrijf de falende test.**

```python
# tests/unit/test_realised_occupancy_gate.py
"""De poort die H2 had moeten tegenhouden vóór de fit in plaats van erna.

De a-priori-poort rekent met een uniforme bezetting van 1/k en noemt dat in het
artefact zelf 'optimistisch'. Voor hmm_k3 gaf dat 165,0 observaties per fold en
`adequate: true`; gerealiseerd waren het er 5,0 tot 28,0 tegen een eis van 100.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.regime.state import VolState, assign_by_variance
from tradebot.validation.data_adequacy import assert_realised_occupancy
from tradebot.utils.failfast import DataContractError

IDX = pd.date_range("2020-01-01", periods=1200, freq="D", tz="UTC")


def _balanced():
    rng = np.random.default_rng(2)
    sigma = pd.DataFrame({"A": np.abs(rng.normal(0.5, 0.2, 1200))}, index=IDX)
    return assign_by_variance(sigma, low_q=0.30, high_q=0.70, min_periods=100)


def _starved():
    """Een sigma-reeks waarin HOOG bijna nooit voorkomt -- de H2-situatie."""
    rng = np.random.default_rng(2)
    sigma = pd.DataFrame({"A": np.abs(rng.normal(0.5, 0.02, 1200))}, index=IDX)
    return assign_by_variance(sigma, low_q=0.005, high_q=0.995, min_periods=100)


def test_a_balanced_assignment_passes() -> None:
    verdict = assert_realised_occupancy(
        _balanced(), n_folds=4,
        min_obs_per_state_per_fold=50, min_occupancy_fraction=0.10,
    )
    assert verdict.adequate


def test_a_starved_state_is_refused_on_the_MINIMUM_not_the_median() -> None:
    """AD-18: het oordeel gaat over het minimum. Een mediaan van 165 met een
    minimum van 5 is precies de meting die H2 groen liet lijken."""
    with pytest.raises(DataContractError):
        assert_realised_occupancy(
            _starved(), n_folds=4,
            min_obs_per_state_per_fold=100, min_occupancy_fraction=0.10,
        )


def test_the_verdict_reports_the_measured_minimum() -> None:
    verdict = assert_realised_occupancy(
        _balanced(), n_folds=4,
        min_obs_per_state_per_fold=50, min_occupancy_fraction=0.10,
        raise_on_failure=False,
    )
    assert "rarest_state_obs_in_smallest_fold" in verdict.measured
    assert verdict.measured["occupancy_source"] == "realised"
```

- [ ] **Stap 9.2 — Draai en zie hem falen.**

- [ ] **Stap 9.3 — Implementeer.** Het oordeel gaat over het **minimum** over folds, niet de mediaan (AD-18), en `occupancy_source` is `"realised"` en niet `"uniform (1/k) — optimistisch"`.

- [ ] **Stap 9.4 — Voeg `vol_state` toe aan `conf/model/adequacy.yaml`** met `min_obs_per_state_per_fold: 100` en `min_state_occupancy_fraction: 0.10` — dezelfde waarden als `hmm`, met een commentaarblok dat uitlegt waarom de a-priori-variant niet volstond.

- [ ] **Stap 9.5 — Draai de poort op de echte toewijzing.**

```bash
D:/venv/tradebot/Scripts/python.exe apps/run_state_diagnostics.py --check-adequacy
```

> **Verwachting, en zij is ongemakkelijk.** De nulmeting toont HOOG op **4,8 %** van de bars — 499 over zes symbolen, 60 tot 117 per symbool. Over 12 folds is dat ordegrootte **5 tot 10 observaties per fold**, ruim onder de eis van 100. **De meest waarschijnlijke uitkomst van deze stap is dat de driedelige toestand op dit paneel niet adequaat is.**
>
> Als dat gebeurt, is dat geen mislukking maar het goedkoopste antwoord van de hele fase, en de reactie ligt vast: **verlaag het aantal toestanden naar twee** (`hmm_k2` haalde a priori 247,5) en herhaal stap 6. Wat **niet** mag: `min_obs_per_state_per_fold` verlagen tot de poort groen wordt. Die drempel is een conventie uit de literatuur en staat vast vóór de meting.

- [ ] **Stap 9.6 — Commit.**

```bash
git add src/tradebot/validation/data_adequacy.py \
        tests/unit/test_realised_occupancy_gate.py conf/model/adequacy.yaml
git commit -m "fix(adequacy): poort op GEREALISEERDE bezetting; de a-priori-variant was optimistisch"
```

---

# STAGE C — DE ECONOMISCHE TOETSEN

*Elke stap hier kost trials. Boek ze vóór de fit.*

---

### Stap 10: Gepaarde standaardfouten in de ladder

De ladder rapporteert per laag een Sharpe-**niveau**, en elk niveau draagt op deze steekproef een standaardfout van **0,458**. Het verschil L1 → L2 is echter een **gepaarde** grootheid op dezelfde bars, met een veel kleinere standaardfout — die nergens wordt gerapporteerd. Zolang dat zo is, valt uit de belangrijkste tabel van de repository niet af te lezen welke laagovergangen significant zijn.

**Files:**
- Modify: `src/tradebot/backtest/phase5_baseline.py`
- Create: `tests/unit/test_layer_paired_differences.py`

**Interfaces:**
- Produces: `phase5_baseline.paired_layer_difference(a: pd.Series, b: pd.Series, *, bars_per_year: int) -> PairedDifference` met `delta_sharpe`, `se`, `t_stat`, `n_obs`

- [ ] **Stap 10.1 — Schrijf de falende test.**

```python
# tests/unit/test_layer_paired_differences.py
"""Een niveauverschil van 0,26 op een steekproef met SE 0,458 zegt niets. Het
GEPAARDE verschil op dezelfde bars zegt mogelijk wel iets, en het staat er nu
niet."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.backtest.phase5_baseline import paired_layer_difference

IDX = pd.date_range("2022-01-01", periods=1000, freq="D", tz="UTC")


def test_identical_series_have_zero_difference_and_zero_t() -> None:
    rng = np.random.default_rng(1)
    series = pd.Series(rng.normal(0.0, 0.01, 1000), index=IDX)
    result = paired_layer_difference(series, series, bars_per_year=365)
    assert result.delta_sharpe == 0.0
    assert result.t_stat == 0.0


def test_the_paired_se_is_smaller_than_the_level_se() -> None:
    """De kern. Twee sterk gecorreleerde reeksen hebben een gepaarde SE die
    veel kleiner is dan 1/sqrt(T) op de niveaus -- dat is precies waarom de
    gepaarde toets het onderscheid kan maken dat de niveaus niet maken."""
    rng = np.random.default_rng(2)
    base = rng.normal(0.0005, 0.01, 1000)
    a = pd.Series(base, index=IDX)
    b = pd.Series(base + rng.normal(0.0, 0.0005, 1000), index=IDX)
    result = paired_layer_difference(a, b, bars_per_year=365)
    level_se = 1.0 / np.sqrt(1000 / 365)
    assert result.se < level_se


def test_a_genuine_degradation_is_detected() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(0.0008, 0.01, 1000)
    a = pd.Series(base, index=IDX)
    b = pd.Series(base - 0.0006, index=IDX)
    result = paired_layer_difference(a, b, bars_per_year=365)
    assert result.delta_sharpe > 0.0
    assert result.t_stat > 2.0
```

- [ ] **Stap 10.2 — Draai en zie hem falen.**

- [ ] **Stap 10.3 — Implementeer.** Gebruik de bestaande `validation/sharpe_difference.py` als rekenkern indien de handtekening past; bouw geen tweede implementatie (dezelfde regel als `validation/dsr.py` voor de DSR).

- [ ] **Stap 10.4 — Breid het artefact uit.** `artefacts/baseline/phase5_revaluation.json` krijgt per track een `layer_transitions`-blok met `L0->L1`, `L1->L2`, `L2->L3`, elk met `delta_sharpe`, `se`, `t_stat`, `n_obs`.

- [ ] **Stap 10.5 — Draai en vergelijk.**

```bash
D:/venv/tradebot/Scripts/python.exe apps/run_phase5_baseline.py
```

Verwacht: alle bestaande niveaus **bit-identiek** aan de NULMETING; het `layer_transitions`-blok is nieuw. Verschuift een niveau, dan is dat een regressie.

- [ ] **Stap 10.6 — Commit.**

```bash
git add src/tradebot/backtest/phase5_baseline.py \
        tests/unit/test_layer_paired_differences.py \
        artefacts/baseline/phase5_revaluation.json
git commit -m "feat(backtest): gepaarde verschillen per laagovergang, met standaardfout"
```

---

### Stap 11: H-10.1 — beslisfrequentie

**Hypothese.** De netto OOS-Sharpe van de baseline stijgt in `k` voor `k ∈ {1, 2, 5, 10}` — het verlies uit de ladder is turnover, geen signaalverval.

**Trials: 4.** Budget na deze stap: 21 van 25.

**Waarom deze eerst.** Van alle hypothesen in deze fase is dit de enige die het grootste gemeten verlies rechtstreeks aanvalt (kosten + latency = 0,456 tegen een brutosignaal van 0,404), én de enige die statistisch bijna gratis is. Doorgerekend over hetzelfde venster van 4,42 jaar:

| Frequentie | `n_obs` | DSR-eis bij M = 25 |
|---|---:|---:|
| dagelijks | 1615 | **1,74** |
| tweedaags | 805 | **1,74** |
| wekelijks | 230 | **1,76** |
| maandelijks | 53 | 1,87 |

Tot en met wekelijks is de bewijslast praktisch onveranderd terwijl de turnover met ongeveer een factor vijf daalt. `k = 10` zit daar net onder en is daarom de bovengrens van het grid.

**Files:**
- Create: `conf/research/preregistration_h10_1_rebalance.yaml`
- Create: `apps/run_rebalance_sweep.py`
- Create: `tests/unit/test_rebalance_sweep.py`
- Create: `reports/phase10_h1_rebalance.md` (output)

- [ ] **Stap 11.1 — Schrijf en bevries de pre-registratie.** Verplichte velden, gelijk aan `preregistration_h2_hmm_vs_m0.yaml`: `title`, `hypothesis`, `null_hypothesis`, `universe`, `period_start/end`, `granularity: 1d`, `primary_metric: net_oos_sharpe`, `planned_trials: 4`, `parameters: {k: [1, 2, 5, 10]}`, `stop_criteria`. Bevries met `apps/freeze_preregistration.py` **vóór de eerste run**.

- [ ] **Stap 11.2 — Roep het budget aan.** In `apps/run_rebalance_sweep.py`, vóór de eerste fit:

```python
assert_within_budget(4, reset_path=Path("artefacts/governance/ledger_reset.json"))
```

- [ ] **Stap 11.3 — Schrijf de test op de sweep.**

```python
# tests/unit/test_rebalance_sweep.py
"""De sweep mag precies de vier geregistreerde waarden draaien. Een vijfde is
een ongeregistreerde trial, en dat is de manoeuvre die stap 3 duur maakt."""
from __future__ import annotations

import pytest

from tradebot.utils.failfast import DataContractError

from apps.run_rebalance_sweep import REGISTERED_K, build_grid


def test_the_grid_matches_the_preregistration() -> None:
    assert build_grid() == REGISTERED_K == (1, 2, 5, 10)


def test_an_unregistered_k_is_refused() -> None:
    with pytest.raises(DataContractError):
        build_grid(override=(1, 2, 3, 5, 10))
```

- [ ] **Stap 11.4 — Draai en zie hem falen. Implementeer. Draai en zie hem slagen.** De sweep hergebruikt `backtest/baseline_runner.py::build_weight_tracks` en `backtest/phase5_baseline.py::run_all_layers` ongewijzigd; alleen `rebalance_every_bars` varieert.

- [ ] **Stap 11.5 — Draai de campagne en rapporteer.** `reports/phase10_h1_rebalance.md` toont per `k` de volledige ladder en de gepaarde overgangen uit stap 10. Boek de vier trials in de ledger.

- [ ] **Stap 11.6 — Commit.**

```bash
git add conf/research/preregistration_h10_1_rebalance.yaml apps/run_rebalance_sweep.py \
        tests/unit/test_rebalance_sweep.py reports/phase10_h1_rebalance.md \
        artefacts/governance/
git commit -m "research(H-10.1): beslisfrequentie-sweep, 4 pre-geregistreerde trials"
```

---

### Stap 12: H-10.2 — het ongebalanceerde paneel

**Hypothese.** Toetreding-op-bestaan verandert het oordeel over de baseline niet, maar levert 5,44 jaar in plaats van 4,78.

**Trials: 1.** Budget na deze stap: 20 van 25.

Gemeten historie per symbool: BTCUSDT 6,41 j · LINKUSDT 5,83 · ETHUSDT 5,44 · DOTUSDT 5,43 · AVAXUSDT 4,93 · SOLUSDT 4,85. Een cross-sectie van drie namen bestaat vanaf **2021-03-15** in plaats van 2021-10-15.

**Wees eerlijk over de opbrengst:** de t=2-drempel gaat van 0,915 naar 0,858. Dat is **6 %**, en het is gratis omdat `registry/lifecycle.py` het contract al draagt. Het is geen hefboom en het mag niet als zodanig worden gerapporteerd.

- [ ] **Stap 12.1** Bevries `conf/research/preregistration_h10_2_unbalanced.yaml`, `planned_trials: 1`.
- [ ] **Stap 12.2** Schrijf een lookahead-test die bewijst dat een symbool nooit een exposure krijgt vóór zijn eerste gecertificeerde bar. Draai; zie hem falen.
- [ ] **Stap 12.3** Bedraad `registry/lifecycle.py::SymbolLifecycle` in `baseline_runner.py::build_weight_tracks`. Draai; zie hem slagen.
- [ ] **Stap 12.4** Draai de baseline op beide panelen. Rapporteer beide ladders naast elkaar met de gepaarde verschillen.
- [ ] **Stap 12.5** Commit.

---

### Stap 13: H-10.3 — toestandsconditionering

**Hypothese.** `gate_by_state` met `flat_states = {HIGH}` op de baseline-exposure verhoogt de netto OOS-Sharpe.

**Trials: 1.** Budget na deze stap: 19 van 25.

**Voorwaarde.** Deze stap draait **alleen** wanneer stap 9 groen is. Is de driedelige toestand niet adequaat, dan draait hij op de tweedelige variant; is ook die niet adequaat, dan luidt het oordeel `UNPROVEN — insufficient data` en telt de trial **niet** mee in `M`, precies zoals `conf/model/adequacy.yaml` voorschrijft.

**Waarom `flat_states = {HIGH}` en niet iets anders.** Eén bit vrijheid, vooraf vastgelegd, met twee gronden die géén van beide een richtingsclaim zijn:
* HOOG is een risicostaat, en vlak gaan is een risicobesluit — geen voorspelling.
* F9 falsificeert shorts op dit universum onvoorwaardelijk. `HIGH → short` is daarmee verboden zonder gewijzigde premisse, en `HIGH → flat` is het enige alternatief dat overblijft.

**En de eerlijke verwachting.** De nulmeting meet in HOOG het **hoogste** gemiddelde rendement (+136,4 %/jr, t = 1,16). Vlak gaan in die toestand haalt dus vermoedelijk rendement weg. Dat is geen reden om de hypothese om te draaien — dat zou een fit op de uitkomst zijn — maar het hoort vóór de run in de pre-registratie te staan als verwachte richting van de fout.

- [ ] **Stap 13.1** Bevries `conf/research/preregistration_h10_3_state_gate.yaml`, `planned_trials: 1`, met de verwachting hierboven letterlijk in het veld `notes`.
- [ ] **Stap 13.2** Schrijf de integratietest: de geconditioneerde keten draait end-to-end door `EventDrivenEngine` en produceert een volledige ladder. Draai; zie hem falen.
- [ ] **Stap 13.3** Bedraad `gate_by_state` tussen L4 en L7 in `baseline_runner.py`. **Niet in de alpha-unit** — dat zou de AST-isolatie van `alpha/base.py` schenden. Draai; zie hem slagen.
- [ ] **Stap 13.4** Draai de campagne, boek de trial, rapporteer in `reports/phase10_h3_state_gate.md`.
- [ ] **Stap 13.5** Commit.

---

# STAGE D — OPPERVLAKTE

*Engineering, geen onderzoek. Deze stage is uitstelbaar naar een volgende fase zonder dat A, B of C erop wachten. Doe hem niet halverwege.*

> **Let op de korrel.** Stage A t/m C is uitgewerkt tot teststappen met code. Stage D is dat bewust NIET: wat hier weg mag, hangt af van uitkomsten uit A t/m C — haalt de driedelige toestand de bezettingspoort, en levert stap 14 een reden om keten A te bewaren? Stage D is daarom gespecificeerd tot BESLUITKORREL en krijgt vóór uitvoering een eigen uitwerkingsronde tot stapkorrel. Wie hem nu op deze korrel uitvoert, raadt.

---

### Stap 14: Keten A economisch scoren — als opruimbesluit, niet als promotie

De Phase 0-inventarisatie stelde vast dat de gefitte modellen in de keten zitten die nooit is gescoord: `train_cpcv` produceert `artefacts/models/`, en die map bevat alleen een README. De vier-lagen-ladder is nog nooit toegepast op iets met een fit.

**Waarom dit een opruimstap is en geen onderzoeksstap.** Onder R4 erft deze meting de **2.400** Optuna-trials die de hyperparameters hebben geselecteerd. Dat tilt `M` van 25 naar 2.425 en de DSR-eis van 1,74 naar **2,46** — waarmee de meting als promotieroute onbetaalbaar is. Er zijn twee eerlijke uitwegen en één oneerlijke:

* **Eerlijk (a)** — draai haar als **diagnostiek**, expliciet niet-promoveerbaar, uitsluitend om te beslissen of keten A wordt opgeruimd. Een diagnostiek die geen kandidaat oplevert, draagt geen selectiedruk en hoeft niet in `M`.
* **Eerlijk (b)** — hertrain met een vooraf geregistreerde, kleine vaste hyperparameterset. Dan tellen alleen die trials en blijven de 2.400 bij de parameters die zij hebben geselecteerd.
* **Oneerlijk** — de bestaande parameters gebruiken en de 2.400 niet tellen.

**Besluit: route (a).** De vraag die keten A openhoudt is een opruimvraag, geen promotievraag. Deze stap kost daarom **nul trials** en mag geen enkele promotieclaim dragen.

- [ ] **Stap 14.1 — Reproduceer keten A tot artefact.** Draai `dvc repro train_cpcv` voor één paar en stel vast of de keten überhaupt draait. **Draait hij niet, dan is dát het antwoord** en vervallen 14.2 t/m 14.4: een keten die niet draait, hoeft niet gescoord te worden om te mogen verdwijnen. Noteer de foutmelding in het rapport.

- [ ] **Stap 14.2 — Schrijf de adapter van `oos_probs` naar `a_t`.** De kansen uit `artefacts/oos_probs/` worden een exposure in `[-1, +1]` via één afbeelding zonder vrije parameter: `a_t = 2·p_t − 1`. Geen drempel, geen schaling — dezelfde discipline als AD-15, en om dezelfde reden: een drempel zou een trial zijn die de uitkomst kan bepalen zonder dat het model iets heeft gedaan.

- [ ] **Stap 14.3 — Draai de volledige ladder** via `backtest/phase5_baseline.py::run_all_layers`, met de gepaarde verschillen uit stap 10.

- [ ] **Stap 14.4 — Schrijf `reports/phase10_chain_a_score.md`** met een verplichte kop **“Dit is geen promotiebewijs”**: waarom de meting diagnostisch is (de 2.400 geërfde trials), en dat een promotieclaim op dit cijfer een mandaatschending is.

- [ ] **Stap 14.5 — Neem het opruimbesluit.** Verslaat keten A de baseline niet, dan gaan `train/ensemble.py` (1.117 LOC, 8,3 %), `labeling/meta.py` (1.181, 18,0 %), `features/regime.py` (1.056, 11,9 %), `tune/objective.py` (955, 10,4 %) en `features/ta.py` (775, 9,1 %) — samen **5.084 LOC** — naar stap 15 met een verdict. Verslaat hij hem wél, dan blijft de keten staan en is dat een bevinding die de volgende fase erft.

- [ ] **Stap 14.6 — Commit.**

```bash
git add reports/phase10_chain_a_score.md
git commit -m "diag(keten A): economische score als opruimbesluit, expliciet niet-promoveerbaar"
```

---

### Stap 15: Verdict per intraday-afhankelijke module

**1.756 LOC** verliezen onder AD-23 hun informatiebron. Behandel ze met de Phase 9-methode: per module een verdict, en verwijderen uitsluitend met een register-entry met herstelcommando.

| Module | LOC | Dekking | Waarom geraakt |
|---|---:|---:|---|
| `bars/_kernels.py` | 330 | 3,9 % | tick-tape |
| `bars/runs.py` | 254 | — | tick-tape |
| `bars/imbalance.py` | 175 | — | tick-tape |
| `bars/dollar.py` | 173 | 45,5 % | tick-tape; klasse D "ambitie" |
| `bars/__init__.py` | 5 | — | pakketmarkering |
| `alpha/csm_volume_clock.py` | 296 | 17,6 % | volume-clock bars |
| `volatility/har_rv.py` | 270 | 81,7 % | 5m realized variance |
| `data/orderbook.py` | 150 | 92,5 % | orderboek; klasse D "ambitie" |
| `alpha/microstructure.py` | 103 | 91,9 % | order flow (OFI) |

- [ ] **Stap 15.1 — Let op de enige echte afhankelijkheid.** `features/regime.py:32` importeert `generate_imbalance_bars` en `generate_runs_bars`. Dat is 1.056 LOC met 11,9 % dekking en het is de op twee na grootste geratchete module. **Beoordeel hem expliciet**: verliest hij zijn barbron, dan verliest hij zijn grond.
- [ ] **Stap 15.2 — Corrigeer de intuïtie in het register.** `features/microstructure.py` (474 LOC) bevat uitsluitend `FundingRateMean`, `FundingRateZScore` en `OpenInterestLogChange` — 8-uurs funding en dagelijkse open interest, **geen tick-data**. Alle vijf microstructuurfeatures in de authoritative 15-feature-registry komen hiervandaan en **blijven**. Idem `volatility/realized.py`: dat is de `squared_return`-proxymachinerie uit stap 2.5.
- [ ] **Stap 15.3** Schrijf `reports/phase10_intraday_surface.md` met per module een verdict en een grondslag.
- [ ] **Stap 15.4** Voer de verwijderingen uit, één commit per pakket, met `reports/phase10_removal_register.md` en per bestand een `git show`-herstelcommando.
- [ ] **Stap 15.5** Draai de volledige suite: **exact 4 failures**, geen enkele skip die een error is geworden.
- [ ] **Stap 15.6** Commit.

---

### Stap 16: `live/` — vervangen

**Het besluit uit de brainstorm, met de meting die het draagt.** De `live/`-tak is gebouwd rond een resolutie die AD-22 afschaft: `live/feed.py` (747 LOC) aggregeert `publicTrade` tot 5s-vensters *"voor exacte pariteit met de trainingsdata"*, `architecture.md` §3 begroot 500 ms end-to-end, en `live/model_signal.py` (591 LOC, **0,0 % dekking**) laadt modellen uit een map die alleen een README bevat.

Op dagbars is één beslismoment per 24 uur geen eventloop maar een geplande run.

| | aansluiten | **vervangen** |
|---|---|---|
| LOC in beheer | 6.173 blijven | 6.173 vervallen, adapter komt terug |
| Twee featurestacks | moeten worden verenigd | vervalt van rechtswege |
| Twee impactmodellen | moeten worden verenigd | er is er nog één |
| D7-pariteit | test tussen twee implementaties | **triviaal** — één implementatie |
| Dekkingspost | `live/` 38,5 %, `oms/router.py` 27,9 % moeten omhoog | de post verdwijnt |

- [ ] **Stap 16.1** Leg het besluit vast als AD-26, met deze tabel als grond.
- [ ] **Stap 16.2 — Behoud de haltketen apart en eerst.** `live/circuit_breaker.py`, `monitoring/sharpe_monitor.py`, `monitoring/execution_drift.py`, `live/execution_controller.py:39` (`risk/limits.py`) en `live/engine.py:41` (`risk/daily_loss_governor.py`). Verhuis ze naar een pakket dat geen barresolutie kent, met hun tests mee. Draai `tests/unit/test_external_monitors_can_actually_halt.py` — die moet een echte degradatie door een echte breaker blijven rijden.
- [ ] **Stap 16.3** Schrijf de falende test voor de adapter: gegeven een `RiskDecision` uit de authoritative keten produceert `daily_runner` exact de orders die `execution/order_router.py` teruggeeft, zonder tweede limietstelsel.
- [ ] **Stap 16.4** Implementeer de adapter. Draai; zie hem slagen.
- [ ] **Stap 16.5** Verwijder `live/` en `oms/` met een register-entry per bestand.
- [ ] **Stap 16.6** Werk `docs/architecture.md` §1 en §3 bij: het drielagenmodel verliest laag 3 in zijn huidige vorm.
- [ ] **Stap 16.7** Commit per stap.

---

### Stap 17: De harde omvangslimiet

`execution/impact_calibration.py` declareert `REQUIRED_DATASETS = ("orderbook_l2", "trades")`. Onder AD-23 komen die er niet. Daarmee worden AD-2 (`IMPACT_UNCALIBRATED`) en AD-3 (`SPREAD_ASSUMED`) **permanent** in plaats van voorlopig.

Beide labels zijn alleen verdedigbaar zolang zij niet binden, en dat is gemeten het geval: bij een participatie van ~1e-7 blijft de impact onder 1 bp. **Dat is nu een gelukkige omstandigheid; het hoort een limiet te zijn.**

- [ ] **Stap 17.1** Leid de participatiegrens af waarbij de impact onder 1 bp blijft, uit `execution/impact_model.py` met `eta = 2,991922`. Schrijf de afleiding op.
- [ ] **Stap 17.2** Schrijf de falende test: een boek boven die grens moet door `RiskEngine` worden geweigerd of geclipt, met een `BindingConstraint` die de nieuwe limiet noemt.
- [ ] **Stap 17.3** Voeg `max_notional_for_uncalibrated_impact` toe aan `conf/risk/default.yaml` met de afleiding in commentaar, en aan `constraint_order`. **Een limiet die niet in `constraint_order` staat, wordt stilzwijgend overgeslagen — dat crasht.**
- [ ] **Stap 17.4** Draai; zie hem slagen. Werk `docs/RISK_CONTRACT.md` bij.
- [ ] **Stap 17.5** Commit.

---

## CRITERIA & VALIDATIE (EXIT CRITERIA)

Elk criterium moet met een commando en zijn output aantoonbaar zijn. Waar het een poort betreft, moet worden bewezen dat die poort **rood kan worden**.

| # | Criterium | Bewijs |
|---|---|---|
| 1 | AD-22 t/m AD-26 staan vast met de meting die elk draagt | `docs/ARCHITECTURAL_DECISIONS.md` |
| 2 | H1 gesloten, DI-18 gesloten, F2 en F19 gemarkeerd | `scripts/check_mandate_consistency.py --strict` → 0 |
| 3 | De mandaatpoort kan rood worden | `test_scanner_goes_red_on_an_unmarked_intraday_reopening` |
| 4 | `M_new` bevroren, oude ledger integraal leesbaar | `artefacts/governance/ledger_reset.json` + `test_the_old_ledger_stays_readable` |
| 5 | **De reset maakt de poort niet permissief** | `test_the_reset_does_not_make_the_gate_permissive` |
| 6 | Een tweede reset wordt geweigerd | `test_a_second_reset_is_refused` |
| 7 | Het trial-budget blokkeert overschrijding | `test_inheriting_optuna_trials_blows_the_budget` |
| 8 | De toestand is causaal | `tests/lookahead/test_vol_state_causality.py` |
| 9 | **De toestandsafbeelding overleeft vol-targeting** | `test_the_mapping_survives_vol_targeting` |
| 10 | De bezettingspoort oordeelt op het minimum | `test_a_starved_state_is_refused_on_the_MINIMUM_not_the_median` |
| 11 | Elke laagovergang draagt een gepaarde standaardfout | `artefacts/baseline/phase5_revaluation.json` → `layer_transitions` |
| 12 | De ladderniveaus zijn bit-identiek aan de nulmeting | diff tegen de NULMETING-tabel |
| 13 | Elke campagne is vooraf bevroren | `artefacts/governance/preregistration_*.json` vóór de eerste fit |
| 14 | Trials geboekt: **6 van 25** (4 + 1 + 1) | `hypothesis_ledger.json` |
| 15 | De suite toont exact 4 failures | `python -m pytest -q` |
| 16 | Dekking is niet gedaald | `reachability_map.py --coverage` |
| 17 | Elke verwijdering is omkeerbaar | `reports/phase10_removal_register.md` |
| 18 | `reports/phase10_exit_report.md` bestaat en noemt per hypothese het oordeel | — |

---

## REGELS & HANDELINGSINSTRUCTIES

1. **Stage A gaat vóór alles.** Een resultaat uit B of C zonder bevroren `M_new` is niet interpreteerbaar.
2. **Diagnostiek en hypothese zijn verschillende dingen, en de scheiding is niet cosmetisch.** Stap 6 en 8 zijn diagnostiek: er wordt niets uit geselecteerd, dus zij kosten nul trials. Zodra er één cel uit wordt gekozen als kandidaat, erft die kandidaat **alle** cellen die zijn bekeken. Dit is de regel die het 55-cellen-voorbeeld uit de nulmeting duur maakt.
3. **Een negatieve uitkomst is een resultaat.** Alle vier de hypothesen kunnen negatief uitvallen, en dat is de meest waarschijnlijke uitkomst. Rapporteer het als bevinding, niet als tegenslag.
4. **Verlaag nooit een drempel om een poort groen te krijgen.** Stap 9 noemt de verleiding met naam.
5. **Geen enkele nieuwe voorspeller.** Deze fase voegt geen model toe. `gate_by_state` is een afbeelding op een bestaande toestand; dat is geen voorspeller.
6. **Bij twijfel: behouden.** De Phase 9-regel blijft gelden voor stap 15 en 16.
7. **Commit per stap.** Een stap die halverwege blijft steken, laat de repository in een toestand waarin de nulmeting niet meer reproduceert.

---

## STARTINSTRUCTIE

Begin met **stap 1.1**: reproduceer de nulmeting. Draai de suite, draai `apps/run_phase5_baseline.py`, en vergelijk de laddertabel en de vier failures met de NULMETING hierboven.

Wijkt er iets af, dan is dát je eerste bevinding en die schrijf je op vóór je één regel code aanraakt.

Klopt alles, ga door naar stap 1.2.
