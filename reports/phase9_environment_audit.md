# Phase 9 — omgevingsaudit (stap 2)

> **Gemeten 2026-09-04 op commit `cfeddf0`.** Elke regel hieronder is gekoppeld
> aan het commando dat hem produceert.

Stap 2 van de fase-opdracht luidt: *"Installeer de omgeving opnieuw uit
`requirements-dev.lock` in een schone virtualenv."* Die herinstallatie is **niet
uitgevoerd**, en de reden daarvoor is de eerste bevinding van deze fase.

---

## 1. De bevinding: de nulmeting is op de verkeerde interpreter gemaakt

De fase-opdracht opent met een tabel onder de kop *"De meetinstrumenten zelf
zijn stuk"* en noemt die *"de belangrijkste bevinding van de nulmeting"*. Zij is
gemeten op de **systeeminterpreter**. `docs/RUNBOOK.md` §0 wijst sinds
2026-08-27 een andere interpreter aan als de enige waarop een meting geldig is:

```
D:/venv/tradebot/Scripts/python.exe      Python 3.13.0
```

Op die interpreter is geen van de zeven bevindingen over de testketen waar.

| Grootheid | `requirements-dev.lock` | Referentie-interpreter | Systeeminterpreter |
|---|---|---|---|
| `pytest` | 9.1.1 | **9.1.1** ✅ | 9.0.2 ❌ |
| `mypy` | 2.3.1 | **2.3.1** ✅ | 1.17.0 ❌ |
| `ruff` | 0.15.12 | **0.15.12** ✅ | — |
| `pytest-cov` | 7.1.0 | **7.1.0** ✅ | `ModuleNotFoundError` ❌ |
| `pytest-randomly` | 4.1.0 | **4.1.0** ✅ | `ModuleNotFoundError` ❌ |
| `coverage` | 7.15.4 | **7.15.4** ✅ | — |
| `pandas` (runtime lock) | 2.3.3 | **2.3.3** ✅ | 2.2.3 ❌ |
| `catboost` (runtime lock) | 1.2.10 | **1.2.10** ✅ | 1.2.8 ❌ |

Reproductie:

```bash
D:/venv/tradebot/Scripts/python.exe -m pytest --version   # 9.1.1
D:/venv/tradebot/Scripts/python.exe -m mypy --version     # 2.3.1 (compiled: yes)
D:/venv/tradebot/Scripts/python.exe -m ruff --version     # 0.15.12
D:/venv/tradebot/Scripts/python.exe -c "import pytest_cov, pytest_randomly, coverage"
```

### De lock is de omgeving, veldsgewijs gecontroleerd

Niet steekproefsgewijs maar over elke pin, met genormaliseerde pakketnamen
(`flufl.lock` en `flufl-lock` zijn hetzelfde pakket; een naïeve vergelijking
meldt daar tien valse verschillen):

```
requirements.lock      : 144 pins | missing=0 diff=0
requirements-dev.lock  :  19 pins | missing=0 diff=0
venv-only packages     :   0
```

**Exit-criterium 6 van deze fase — *"`pip freeze` en `requirements-dev.lock`
verschillen nergens"* — is daarmee vóór aanvang van de fase al waar**, en de
suite draaide zojuist onder `pytest-randomly` beschikbaar (uitgezet met
`-p no:randomly` om een deterministische afdruk te krijgen; de
willekeurige-volgordemeting is exit-criterium 6 en volgt apart).

### Waarom dit ertoe doet en niet alleen een correctie is

De opdracht leidt uit de kapotte instrumenten twee gevolgtrekkingen af die
allebei vervallen:

* *"`make coverage` en `fail_under = 70` zijn nooit uitgevoerd."* Niet
  aantoonbaar. Op de referentie-interpreter is `pytest-cov` aanwezig; of het
  ooit is gedraaid, is met de aanwezigheid van het pakket niet te bepalen. Wat
  wél meetbaar is, staat in `reports/phase9_coverage_baseline.md`.
* *"`pytest-randomly` ontbreekt, dus testvolgorde-onafhankelijkheid is nooit
  getoetst."* De premisse is onwaar; het pakket staat er en `docs/RUNBOOK.md`
  regel 70 documenteert juist een aanroep die het expliciet uitzet.

Phase 0 exit-criterium 2 (*"100 % reproduceerbare build"*) is **niet gedrift**.
Wat is gedrift, is de systeeminterpreter — en die is per RUNBOOK §0 geen
meetinstrument van dit project.

---

## 2. Wat van de nulmeting wél overeind blijft

Onafhankelijk nagemeten, en alle drie bevestigd:

| Bevinding | Status | Bewijs |
|---|---|---|
| `make` bestaat niet in deze omgeving | **bevestigd** | `make --version` → `command not found`. De `Makefile` is documentatie, geen poort. |
| `make loc-check` zou rood staan op 6 bestanden | **bevestigd** | `backtest/evaluation.py` 1054 · `features/regime.py` 1056 · `live/engine.py` 913 · `portfolio/legacy_sizing.py` 824 · `schemas/config.py` 1094 · `validation/data_adequacy.py` 804 |
| De LOC-whitelist noemt `risk/portfolio.py`, dat niet bestaat | **bevestigd** | `ls src/tradebot/risk/portfolio.py` → geen bestand. `DEFERRED_ISSUES.md` DI-10 (gesloten) zegt zelf al *"`risk/portfolio.py` bestaat niet meer"*. |
| DI-3 claimt "gemeten 2026-09-01: nog 2 bestanden >800 LOC" | **onjuist, bevestigd** | Gemeten nu: **9**. En de twee die DI-3 noemt staan er met andere getallen: `labeling/meta.py` 838 → **1181**, `backtest/evaluation.py` 833 → **1054**. |
| 27 van 40 apps boven de 80-LOC-limiet | **bevestigd** | DI-4 noteert 18 van 40; die meting is verouderd. DI-4 noemt daarbij `apps/freeze_monitoring.py` als "63 regels" — het zijn er **82**, dus ook dat voorbeeld overschrijdt inmiddels. |
| Totale omvang `src/` | **bevestigd** | 299 modules, 71.487 LOC — exact de nulmeting. |

De negen bestanden boven 800 LOC:

```
1181 src/tradebot/labeling/meta.py           (whitelist)
1117 src/tradebot/train/ensemble.py          (whitelist)
1094 src/tradebot/schemas/config.py
1056 src/tradebot/features/regime.py
1054 src/tradebot/backtest/evaluation.py
 955 src/tradebot/tune/objective.py          (whitelist)
 913 src/tradebot/live/engine.py
 824 src/tradebot/portfolio/legacy_sizing.py
 804 src/tradebot/validation/data_adequacy.py
```

---

## 3. Werkboom-hygiëne — nagemeten

| Map | Bestanden | Omvang | Getrackt |
|---|---:|---:|---|
| `.mypy_cache` | 2.773 | 106 MB | nee |
| `.hypothesis` | 555 | — | nee |
| `logs/` | 68 | 5 MB | nee |

Bevestigd. Aangetroffen bovenop de nulmeting: **vier getrackte niet-Python-
bestanden binnen het pakket zelf**, die met `pip install .` meereizen naar elke
installatie:

```
src/tradebot/.claude/settings.local.json          -> ontrackt (stap 14)
src/tradebot/.claude/scheduled_tasks.lock         -> ontrackt (stap 14)
src/tradebot/artefacts/runbook.md                 -> verplaatst (stap 14)
src/tradebot/artefacts/PROPFIRM_PARALLEL_AUDIT.md -> BEHOUDEN
```

Afgehandeld in stap 14. De twee `.claude/`-bestanden zijn lokale
gereedschapstoestand en zijn uit versiebeheer gehaald; `.claude/` staat nu in
`.gitignore`. De runbook in het pakket was een TWEEDE runbook naast
`docs/runbook.md` met nul verwijzingen — dubbele waarheid — en staat nu als
`miscellaneous/runbook_wave15_2026-05-19.md`.

`PROPFIRM_PARALLEL_AUDIT.md` blijft waar hij staat: hij wordt aangehaald door
`src/tradebot/risk/daily_loss_governor.py`, `tests/unit/test_propfirm_governor.py`
en `docs/STRATEGY_AUDIT_CRYPTO_ACCOUNTS_2026-06-14.md`. Een document dat de
risicolaag als bron noemt, verplaats je niet als bijvangst van een opruiming.

---

## 4. Gevolg voor de fase

1. Stap 2 vervalt als herstelactie. Er is niets te herstellen; er was iets
   verkeerd gemeten. Elke meting in deze fase draait op
   `D:/venv/tradebot/Scripts/python.exe` en elk rapport vermeldt dat.
2. Deliverable 8 (*"Herstelde dev-omgeving"*) wordt gelezen als **aangetoonde**
   dev-omgeving; het bewijs staat hierboven en in
   `reports/phase9_coverage_baseline.md`.
3. Exit-criterium 6 is aantoonbaar waar vóór de fase begint. Dat is een
   uitkomst, geen prestatie van deze fase, en wordt in het exit-rapport zo
   opgeschreven.
4. **De grotere les geldt de rest van de nulmeting.** Eén sectie ervan is op
   het verkeerde gereedschap gemeten; dat is reden om de andere secties niet op
   gezag over te nemen maar na te meten. De bereikbaarheidskaart (stap 6) doet
   dat, en wijkt inderdaad af.
