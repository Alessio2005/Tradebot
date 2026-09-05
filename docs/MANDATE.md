# MANDAAT

> **Fase:** 10 · **Status:** bindend · **Genomen door:** de eigenaar
> **Uitgewerkt in:** AD-22, AD-23, AD-24 (`docs/ARCHITECTURAL_DECISIONS.md`)
> **Exit-criterium:** 1
>
> **Geverifieerd tegen de codebase op 2026-09-05** (fase 10, stap 1).

Dit document bevat de drie besluiten die de eigenaar heeft genomen en die fase
10 **uitvoert**. Zij worden hier vastgelegd, niet beargumenteerd: de redenering,
het afgewezen alternatief en de bewakende test staan per besluit in de
bijbehorende AD.

**Fase 10 voert deze drie uit en toetst ze niet. Zij mag geen vierde besluit
nemen.**

---

## De drie besluiten

| | Besluit | Wat het sluit | Uitgewerkt in |
|---|---|---|---|
| **B-1** | **Barresolutie: dagbars.** De handelsklok volgt de meetklok. `live/feed.py::FeedConfig.bar_seconds` vervalt. | Het resolutieverschil tussen de twee ketens; het verschil 5 vs. 6 namen; exit-criterium D7 wordt voor het eerst *gedefinieerd* | **AD-22** |
| **B-2** | **Één meetdomein: de dagbar.** Het domein is drie gecertificeerde bronnen op één frequentie. Wat een fijnere waarneming vereist, valt buiten het domein en is geen uitgestelde vraag. | H1's heropeningsconditie, DI-18, `bars/`, F2 en F19 | **AD-23** |
| **B-3** | **De ledger wordt gereset.** `M = 2776` is niet langer de trial-teller voor dit programma. | De rekenkundige onbereikbaarheid van de promotiepoort | **AD-24** |

---

## Wat het mandaat niet doet

Drie afbakeningen, omdat elk van de drie besluiten een lezing uitnodigt die
ruimer is dan bedoeld:

* **B-2 geeft geen enkele regel uit `FALSIFICATION_REGISTER.md` vrij.** F1 t/m
  F20 blijven onverkort bindend. Er mag uitsluitend een domeinmarkering worden
  toegevoegd; geen regel wordt gewijzigd, verzwakt of verwijderd.
* **B-3 raakt geen poort.** De vijf promotiepoorten in `validation/gates.py`
  krijgen geen `force=`, geen `override=`, geen `warn_only=` en geen deelscore.
  Er verandert één invoerwaarde — `M` — en verder niets.
* **B-2 sluit geen domeinconforme module.** Range-schatters op dagelijkse OHLC,
  funding, open interest en de dagelijkse volatiliteitsmodellen vallen
  binnen het domein en blijven. Zie AD-23, "Wat er binnen het domein wél mag".

---

## Waar de meetconventies staan

Niet hier. `docs/MEASUREMENT_CONTRACT.md` is de enige plaats waar het
meetvenster, de annualisatie, de standaardfouten, de verschiltoets, de
DSR-handtekening, de rendementsconventie en de purge/embargo-regel staan.

Dit mandaat zegt *wat* er is besloten; het contract zegt *waarop* er wordt
gemeten.
