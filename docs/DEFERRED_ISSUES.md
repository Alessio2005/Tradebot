# DEFERRED ISSUES

> Register van problemen die zijn **aangetroffen** tijdens een fase maar die
> buiten de scope van die fase vallen. Doctrine (Phase 0, regels): *"Niets
> verdwijnt stilzwijgend."* Elk item krijgt een fase-toewijzing.
>
> Dit register is append-only. Een opgelost item wordt niet verwijderd maar
> gemarkeerd met de commit die het sluit.

**Geverifieerd tegen de codebase op 2026-08-22, Phase 0.**

---

## Openstaand

| ID | Gevonden in | Probleem | Waarom niet nu | Toegewezen aan |
|---|---|---|---|---|
| **DI-1** | `risk/hmm_regime.py::predict` | `GaussianHMM.predict` levert het **Viterbi**-pad, dat de volledige reeks gebruikt inclusief observaties na `t`. Dat is *smoothed*, niet *filtered*, en schendt de strikte regel uit audit sectie 10.2. | Phase 0 verwijdert stille degradatie maar wijzigt geen modelgedrag. De audit wijst `hmm_regime.py` expliciet aan als **REDESIGN** naar M2 Filtered HMM in Phase 6. | **Phase 6** |
| **DI-2** | `risk/hmm_regime.py::_features` | `returns.rolling(5).std().fillna(returns.std())` vult de burn-in met de std over de **volledige sample**. Lookahead-lek in de vol-feature. | Idem DI-1: modelgedrag blijft in Phase 0 ongewijzigd. Wordt bij het M2-herontwerp vervangen door een expanding std, met bewijs via de Phase 2-gates. | **Phase 6** |
| **DI-3** | `reports/phase0_baseline.md` par. 3 | 9 bestanden overschrijden de 800-LOC-limiet uit `architecture.md` R-4 (D-6, P2). | D-6 is P2 en raakt geen enkel statistisch contract. Splitsen tijdens Phase 0 zou de baseline-snapshot onvergelijkbaar maken. | **Phase 5** (samenvallend met de consolidatie van de 4 backtesters) |
| **DI-4** | `reports/phase0_baseline.md` par. 4 | 19 van de 30 apps overschrijden de 80-LOC-limiet uit `architecture.md` R-6 (D-7, P2). | Idem D-6. Nieuwe apps die vanaf Phase 1 worden toegevoegd respecteren de limiet wel. | **Phase 7** |
| **DI-5** | `docs/tca_methodology.md` | D-2/D-3/D-4: `apps/calibrate_impact.py`, `tests/integration/test_tca_roundtrip.py` en `conf/tca/default.yaml` worden gedocumenteerd maar bestaan niet. | De eta-kalibratie is expliciet Phase 5. Het document wordt daar gecorrigeerd, samen met de bouw van de ontbrekende artefacten. | **Phase 5** |
| **DI-6** | `src/tradebot/backtest/` | Vier overlappende backtest-engines (`evaluation`, `portfolio`, `bidirectional`, `per_side`). | Consolidatie tot 1 event-driven engine vereist een pariteitstest, die zelf de Phase 2-gates nodig heeft. | **Phase 5** |
| **DI-7** | `src/tradebot/live/` (5x) | `RUF006` — `asyncio.create_task(...)` zonder de referentie vast te houden. Geen stijlkwestie: een niet-vastgehouden task kan door de GC worden opgeruimd, waardoor een feed- of monitortaak stilletjes verdwijnt. | Het vasthouden van task-referenties raakt de levenscyclus van de live-loop; dat is Phase 7-territorium en vereist de chaos-tests die daar worden gebouwd. | **Phase 7** |
| **DI-8** | `scripts/` (legacy) | De onderzoeksscripts (`combined_neutral_book.py`, `market_neutral_alpha.py`, ...) staan naast de Phase 0-tooling in `scripts/`. Volgens sectie 20 horen ze in een `research/`-track zonder productiecode. | Verplaatsen breekt bestaande verwijzingen in de wave-documentatie; de CI-lintstap is voorlopig gescoped op `src/` plus de Phase 0-tooling. | **Phase 5** |
| **DI-9** | `execution/`, `backtest/`, `tca/` | Hardcoded parameters (exit criterium 6). Gemeten budget staat in `scripts/check_hardcoded_params.py`. | Deze modules worden in Phase 5 herbouwd rond het gekalibreerde eta-model; nu parametriseren is dubbel werk. | **Phase 5** |
| **DI-10** | `risk/`, `portfolio/` | Idem. Inclusief een EWMA-lambda van 0.94 die als *default-argument* in `risk/portfolio.py::_safe_corr` staat in plaats van uit `conf/model/volatility.yaml` te komen. | Phase 4 ontkoppelt risk van alpha en parametriseert de laag opnieuw. | **Phase 4** |
| **DI-11** | `volatility/` (niet-baseline), `train/`, `tune/`, `selection/`, `labeling/`, `cv/` | Idem. | Level 2+ modellen worden in Phase 6 geevalueerd en daarbij geparametriseerd. | **Phase 6** |
| **DI-12** | `volatility/ewma.py`, `features/`, `alpha/` | Idem. `ewma.py` heeft géén lambda-parameter maar een `halflife=20`, en vult de burn-in met `fillna(0.0)` — een **impliciete constante volatiliteit van nul**, wat in risk parity een oneindig gewicht oplevert. | Het harden van de EWMA-estimator (lambda uit `conf`, expliciete burn-in, NaN-propagatie i.p.v. nul) is expliciet Phase 3 stap 3. Het nu al doen is scope creep. | **Phase 3** |
| **DI-13** | `data/`, `bars/` | Idem. | Phase 1 herschrijft de ingestion-laag volledig. | **Phase 1** |
| **DI-14** | `live/`, `oms/`, `monitoring/`, `compliance/`, `registry/`, `utils/` | Idem. | Phase 7 (productie-readiness). | **Phase 7** |
| **DI-15** | `artefacts/governance/symbol_lifecycle.json` | **Survivorship bias.** De publieke Bybit V5-API retourneert uitsluitend nog-verhandelde instrumenten (gemeten: 833 van 833 met status `Trading`, nul delistings). Het universum bestaat daardoor uit zes ex-post gekozen overlevers. Een momentum-strategie op overlevers oogt gunstiger dan dezelfde strategie op een periodiek hersamengesteld universum. | Vereist een TWEEDE databron met delisting-historie; dat is een inkoopbesluit, geen technische keuze. Het contract (`SymbolLifecycle`, `validate_continuity`) is wel gebouwd en weigert bars voor de listing of na de delisting, dus de data plugt in zodra hij er is. | **Phase 3** (vermelden bij elke baseline-claim), sluiten zodra een bron beschikbaar is |

---

## Gesloten

| ID | Probleem | Gesloten door |
|---|---|---|
| **DI-13** | Hardcoded parameters in `data/`, `bars/` | Deels: de NIEUWE ingestion-laag (`data/ingestion/`, `data/validation/`, `data/pit_store.py`, `data/panel.py`) is volledig config-gedreven en heeft ratchet-budget 0. De legacy `data/`-modules blijven staan tot Phase 5 consolideert. |
