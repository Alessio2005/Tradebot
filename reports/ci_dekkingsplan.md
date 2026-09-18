# Dekkingsplan — de afstand tot `fail_under = 70`

> **Gemeten 2026-09-18** op de referentie-interpreter (`docs/runbook.md` §0,
> Python 3.13.0), na de CI-sanering van diezelfde dag.
> **Dit rapport lost niets op.** Het meet, en het legt drie opties voor. De
> keuze is een mandaatbesluit en niet die van de uitvoerder van de sanering.

---

## 1. Het commando en het cijfer

```bash
D:/venv/tradebot/Scripts/python.exe -m pytest -m "not slow and not regression" \
    -p no:randomly --cov=src/tradebot --cov-report=term -q
```

```
TOTAL   25023 statements   9842 missed   61%
FAIL Required test coverage of 70.0% not reached. Total coverage: 60.67%
```

| | statements |
|---|---|
| totaal | 25.023 |
| gedekt | 15.181 |
| nodig voor 70 % | 17.517 |
| **gat** | **2.336** |

## 2. Waar het verschil vandaan komt

| meting | dekking | bron |
|---|---|---|
| 2026-09-04, commit `26cfd90` | 55,82 % | `reports/phase9_coverage_baseline.md` |
| 2026-09-18, commit `371755e` | 59,10 % | nulmeting van de CI-sanering |
| 2026-09-18, na de sanering | **60,67 %** | dit rapport |

De sprong van 55,82 naar 59,10 is het werk van Phase 10 en staat los van deze
opdracht. De 1,57 procentpunt daarna is bijvangst van de sanering zelf:
`tests/unit/test_app_call_sites.py` raakt elke module die een app importeert.

Dat is meteen de maat van de klus. **Een test die 34 bestanden aanraakt, levert
anderhalf procentpunt.** De resterende tien procentpunt zijn geen bijvangst.

## 3. Het gat, per pakket

Gesorteerd op ongedekte statements — dat is waar het gat feitelijk zit, niet
waar het dekkingspercentage het laagst is.

| pakket | stmts | gemist | dekking |
|---|---|---|---|
| `data` | 2322 | 1239 | 46,6 % |
| `features` | 2483 | 1130 | 54,5 % |
| `train` | 1841 | 1112 | 39,6 % |
| `live` | 1673 | 1046 | 37,5 % |
| `reporting` | 1038 | 703 | 32,3 % |
| `labeling` | 831 | 564 | 32,1 % |
| `backtest` | 1427 | 529 | 62,9 % |
| `alpha` | 1680 | 491 | 70,8 % |
| `portfolio` | 926 | 449 | 51,5 % |
| `bars` | 508 | 445 | 12,4 % |
| `validation` | 2408 | 342 | 85,8 % |
| `execution` | 826 | 337 | 59,2 % |
| `tune` | 348 | 289 | 17,0 % |
| `monitoring` | 657 | 227 | 65,4 % |
| `risk` | 1392 | 216 | 84,5 % |

De tien grootste ongedekte bestanden:

| stmts | gemist | bestand |
|---|---|---|
| 471 | 415 | `features/regime.py` |
| 445 | 408 | `train/ensemble.py` |
| 304 | 304 | `portfolio/legacy_sizing.py` |
| 328 | 300 | `reporting/phase6_vol_competition.py` |
| 323 | 265 | `labeling/meta.py` |
| 288 | 258 | `tune/objective.py` |
| 282 | 257 | `reporting/phase6_regime_benchmark.py` |
| 349 | 254 | `backtest/evaluation.py` |
| 264 | 240 | `features/ta.py` |
| 274 | 233 | `data/crypto.py` |

Samen 2.934 ongedekte statements — **meer dan het hele gat.** Het gat is dus
niet diffuus; het zit in tien bestanden, en zeven ervan staan ook op de
LOC-lijst uit DI-3 (boven 800 regels of vlak eronder).

Twee waarnemingen die de keuze in §4 sturen:

* **`portfolio/legacy_sizing.py` is 100 % ongedekt en dat is met opzet.** Hij
  staat in `docs/CODE_REGISTER.md` als geregistreerd onbereikbaar en is in
  Phase 4 ONGEWIJZIGD verhuisd zodat de Phase 3-baseline herrekenbaar blijft
  (DI-10). Hem testen betekent hem aanraken, en dat is precies wat er niet mag.
  304 statements van het gat zijn per besluit onbereikbaar.
* **`bars/_kernels.py` staat op 12,4 %** omdat het numba-kernels zijn; de
  gecompileerde paden tellen niet mee in `coverage.py`. Dat is een meetartefact,
  geen ontbrekende test — de kernels worden wél gedraaid, onder andere door
  `tests/benchmark/`, die in deze selectie is uitgesloten (`-m "not slow"`).

Van de 2.336 statements die het gat groot maken, zijn er dus zeker 304 niet
legitiem te dekken en een deel van de 445 in `bars/` is al gedekt zonder dat de
meting het ziet.

## 4. De drie opties

### (a) Tests schrijven tot 70 %

Ruwweg 2.336 statements, in de praktijk meer omdat `legacy_sizing.py` niet
meetelt. Op de hierboven genoemde tien bestanden: `features/regime.py`,
`train/ensemble.py` en `labeling/meta.py` dragen statistische contracten, dus
dit is geen mechanische testklus — elke test moet de vraag beantwoorden waarvoor
de code bestaat.

**Kosten:** een fase op zichzelf. **Opbrengst:** de drempel wordt gehaald én
betekent iets. **Risico:** de verleiding om dekking te kopen met tests die de
code uitvoeren zonder iets te beweren; dat verhoogt het cijfer en verlaagt de
waarde.

### (b) Een dekkingsratchet op het gemeten niveau, met 70 als staand doel

Zelfde patroon als `check_hardcoded_params.py` en `check_file_size.py`: de
dekking mag niet omlaag, en elke fase duwt hem omhoog. `fail_under` blijft op 70
staan als doel; de blokkerende poort wordt de ratchet.

**Kosten:** een scriptje plus een cap-bestand. **Opbrengst:** vanaf nu gaat het
cijfer alleen omhoog, en nieuwe code moet gedekt zijn. **Risico:** dit is
feitelijk een verlaging van de eis van vandaag. Dat mag alleen een bewust
besluit zijn, niet een gevolg.

### (c) De drempel accepteren als permanent rode poort, met geregistreerde motivering

De stap blijft rood, met een DI-regel die zegt waarom en wanneer hij groen mag
worden.

**Kosten:** nul. **Opbrengst:** het cijfer blijft zichtbaar en eerlijk.
**Risico:** een poort die altijd rood staat, wordt niet meer gelezen — precies
de faalwijze die `security-scan.yml` heeft voorgedaan (DI-26).

## 5. Wat hier NIET in staat

`fail_under` verlagen. `reports/phase9_coverage_baseline.md` sluit dat expliciet
uit, en de fase-opdracht die de drempel zette deed dat ook. De drempel is de
eis; het cijfer is de werkelijkheid; het verschil is het werk.

## 6. Wat de sanering wél heeft gedaan

Tot 2026-09-18 raakte deze drempel **twee** jobs zonder dat een van beide dat in
zijn stapnaam zei: `inventory.yml` (expliciet) en `ci.yml::test-fast` (impliciet
— `pytest --cov` leest `fail_under` uit `pyproject.toml`). Dat is niet veranderd
en het hoort genoemd te worden: wie deze drempel aanraakt, raakt twee workflows.
