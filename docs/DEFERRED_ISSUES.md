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

---

## Gesloten

| ID | Probleem | Gesloten door |
|---|---|---|
| — | _(nog geen)_ | — |
