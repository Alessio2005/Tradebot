# MASTER-PROMPT: FASE 11 — BREEDTE, EN DE TIJDSCHAAL DIE HAAR NIET VERVANGT

> **Fase:** 11, volgt op `fase_10_herstart_dagbars.md` · **Prioriteit:** P0 · **Revisie:** 1
> **Auditgrondslag:** `reports/AUDIT_BRUIKBAARHEID_2026-09-19.md` (nulmeting op `f50f6ca`)
> **Bindende brondocumenten:** `docs/MANDATE.md`, `docs/MEASUREMENT_CONTRACT.md`, `docs/FALSIFICATION_REGISTER.md` (F1–F20), `docs/ARCHITECTURAL_DECISIONS.md` (AD-22 t/m AD-26), `docs/DEFERRED_ISSUES.md`, `reports/phase10_h10_1_decision_frequency.md` §2
> **Trial-budget:** `M_new = 25` (AD-24), **20 resterend** (H-10.1 boekte 4, H-10.2 boekte 1)
>
> **Voor uitvoerders:** stappen gebruiken checkbox-syntaxis (`- [ ]`). Elke stap eindigt in een commit. Stage A blokkeert alles.

---

## 0. WAAROM DEZE FASE BESTAAT, EN WAT ZIJ NIET IS

De eigenaar heeft één vraag gesteld: *hoe wordt deze setup winstgevend, en zijn
grotere tijdframes daarvoor de weg?*

Deze fase beantwoordt die vraag in de volgorde waarin zij beantwoordbaar is. Dat
is niet de volgorde waarin zij is gesteld, en §2 legt uit waarom.

**Wat deze fase NIET is.** Zij is geen zoektocht naar een strategie. Negen fasen
en 2.776 trials hebben twintig hypothesen gefalsifieerd en nul modellen
gepromoveerd. Een tiende fase die opnieuw gaat zoeken zonder de reden te
adresseren waarom de vorige negen niets vonden, is de duurste vorm van hoop die
er bestaat.

**Wat deze fase WEL is.** Zij stelt vast — met rekenkunde, niet met oordeel —
welke van de vier termen in de Sharpe-begroting groot genoeg is om het gat naar
de promotiepoort te dichten, en zij dwingt de eigenaar tot het ene besluit dat
daaruit volgt.

> **De regel die deze fase regeert:**
>
> **Geen enkele stap in deze fase voegt een voorspeller toe voordat vaststaat
> dat er een meetopstelling bestaat waarin een voorspeller aantoonbaar kan
> winnen. Stage C beantwoordt die vraag. Valt zij negatief uit, dan is de
> uitkomst van deze fase een afsluitdocument en dat is een volwaardige
> oplevering.**

---

## 1. ROL EN CONTEXT

Je acteert als **Quant Research Engineer & Platform Architect**.

Je erft een repository die methodologisch beter is dan de meeste
productieomgevingen — point-in-time gecertificeerde data met hashes per reeks,
een append-only trial-ledger, een bevroren poortsample die nog nooit is gelezen
(`holdout_lock.json` draagt `reads: []`), killgates die rood worden wanneer een
gefalsifieerd gedrag terugkeert, en een falsificatieregister dat het meest
waardevolle bestand in het project is.

Je erft ook een repository die op dit moment **niet start op een verse kloon**,
waarvan **drie tests rood staan op `main`**, en waarvan de enige poort die de
live-keten end-to-end zou beproeven **nog nooit één test heeft gedraaid**.

Beide dingen zijn tegelijk waar. Stage A repareert het tweede, omdat geen enkele
meting uit Stage C of D interpreteerbaar is op een instrument dat niet
reproduceerbaar start.

**Relevante lagen (Target Architecture §19):** L3 (features), L4 (alpha), L7
(risk), L10 (backtest), L11 (validatie), L12 (governance).

---

## 2. DE NULMETING — gemeten 2026-09-20 op `f50f6ca`, referentie-venv

Alles in dit hoofdstuk is gemeten, niet aangenomen. Reproduceer het in stap 1.1
voordat je iets anders doet. **Wijkt jouw meting af, dan is dát je eerste
bevinding** (R-10).

### 2.1 De Sharpe-begroting — waar het gat zit

De promotiepoort eist op dit meetdomein een **geannualiseerde Sharpe van 1,69**
(DSR, α = 0,05, `M = 25`, T = 1.742 bars = 4,77 jaar, gemeten skew −0,60 en
kurtosis 7,61). Het beste dat ooit in deze repository is gemeten:

| Post | Gemeten | In Sharpe-eenheden |
|---|---|---:|
| **Vereist door de poort** | DSR α = 0,05, M = 25 | **1,69** |
| Beste netto Sharpe, L3 (volledige executie) | `xs_momentum_equal_weight` | **−0,168** |
| Beste Sharpe vóór executie, L2 | `long_only_equal_weight` | **+0,136** |
| Executiekosten (L2 → L3, momentumtracks) | fees + spread + impact | **≈ 0,12** |
| Funding-drag op een 100 % long boek | 4,42 %/jr bij 72,1 % vol | **≈ 0,061** |
| **Restgat, toe te schrijven aan het signaal** | | **≈ 1,6** |

**Lees die laatste regel.** Kosten en funding samen zijn ongeveer **0,18
Sharpe**. Het gat is **1,86**. Elke interventie die uitsluitend kosten aanpakt,
kan hooguit **10 %** van het gat dichten.

### 2.2 Wat de bar-resolutie doet met het onderscheidingsvermogen — niets

Ik heb de DSR-drempel herrekend bij vier resoluties op **dezelfde 4,77 jaar**:

| Resolutie | bars | bars/jaar | skew | kurtosis | **min. ann. Sharpe** |
|---|---:|---:|---:|---:|---:|
| dagelijks | 1.742 | 365 | −0,60 | 7,61 | **1,69** |
| wekelijks | 249 | 52 | −0,25 | 3,86 | **1,70** |
| 2-wekelijks | 125 | 26 | −0,07 | 2,66 | **1,69** |
| maandelijks | 58 | 12 | +0,20 | 2,75 | **1,67** |

**Aggregatie is vermogens-neutraal.** De reden is structureel en geen
toevalligheid van deze steekproef: de DSR-teststatistiek staat op
`SR_bar · √(T−1)`, en `SR_bar = SR_ann / √(bars_per_jaar)`. Het product
`SR_ann · √(T / bars_per_jaar) = SR_ann · √(kalenderjaren)` is invariant onder
aggregatie.

> **De bindende grootheid is kalendertijd, niet het aantal bars.** 4,77 jaar
> blijft 4,77 jaar, of je die nu in 1.742 of in 58 stukken knipt.

Dit weerlegt tegelijk het voor de hand liggende bezwaar tegen de vraag van de
eigenaar (*"minder bars is minder power"*) én de hoop erachter (*"grotere
tijdframes maken het makkelijker"*). Geen van beide is waar. Aggregatie is
gratis, en zij levert niets op.

De enige echte winst is de derde kolom: **de staarten worden dunner** (kurtosis
7,61 → 2,75, skew −0,60 → +0,20 — de centrale limietstelling aan het werk). Dat
verlaagt de drempel van 1,69 naar 1,67. Dat is een verbetering in de tweede
decimaal.

### 2.3 Is er trend op langere horizonnen? — niet aantoonbaar

Lo-MacKinlay (1988) variantieratio op dagelijkse log-returns, 2021-11-15 →
2026-08-23, `n = 1742`, heteroskedasticiteit-robuuste `z`. VR > 1 =
persistentie/trend; VR < 1 = mean-reversion.

| reeks | VR(2) | VR(5) | VR(10) | VR(20) | VR(60) | max abs z |
|---|---:|---:|---:|---:|---:|---:|
| BTCUSDT | 0,978 | 0,994 | 0,995 | 1,026 | **1,166** | 0,72 |
| ETHUSDT | 0,981 | 0,987 | 0,983 | 1,016 | **1,168** | 0,73 |
| SOLUSDT | 0,944 | 0,929 | 0,950 | 0,999 | **1,185** | 0,62 |
| AVAXUSDT | 1,002 | 1,002 | 1,000 | 1,098 | **1,332** | 1,40 |
| LINKUSDT | 0,981 | 0,937 | 0,894 | 0,878 | 0,892 | 1,05 |
| DOTUSDT | 0,996 | 0,917 | 0,880 | 0,914 | 0,865 | 1,27 |
| **gelijkgewogen boek** | 0,977 | 0,955 | 0,943 | 0,986 | **1,169** | 0,72 |

**De intuïtie van de eigenaar heeft het juiste teken en geen bewijs.** Op q = 60
staat VR boven 1 voor vier van de zes namen en voor het boek (1,169). Dat is de
richting die je verwacht als er trend is. Maar **geen enkele `z` haalt 1,96** —
de grootste is 1,40. Twee namen (LINK, DOT) lopen op élke horizon de andere kant
op.

Op korte horizonnen is het boek licht mean-reverting (VR(5) = 0,955), wat
overeenstemt met F8: *"crypto is XS-reversal, niet -momentum"*.

Dit is geen falsificatie van de tijdschaalhypothese. Het is de vaststelling dat
**deze steekproef haar niet kan bevestigen of weerleggen**, en dat een fase die
erop wordt gebouwd dus vooraf weet dat zij waarschijnlijk `UNPROVEN` oplevert.

### 2.4 De bindende beperking — N_eff = 1,27

Correlatiematrix van dagelijkse log-returns, `n = 1742`:

```text
gemiddelde paarsgewijze correlatie : 0,749
eigenwaarden                       : [4,74  0,37  0,31  0,23  0,20  0,15]
variantie verklaard door PC1       : 79,1 %
N_eff (participation ratio)        : 1,58
N_eff (gelijke-correlatie)         : 1,27
```

**Dit universum is effectief 1,3 activa.** Zes namen, één beta.

Twee gevolgen, en ze zijn allebei hard:

**(a) Het eigen killgate-criterium verwerpt het eigen universum.** F20 archiveerde
`cm_carry` onder meer op *"N_eff 1,54 op vier gecorreleerde energieproducten"*.
Dit universum scoort **lager** dan het universum dat op die grond is afgekeurd.

**(b) Namen toevoegen helpt niet.** `N_eff = N / (1 + (N−1)·ρ̄)` loopt naar de
asymptoot `1/ρ̄`:

| N namen | ρ̄ = 0,75 | ρ̄ = 0,55 | ρ̄ = 0,35 | ρ̄ = 0,15 |
|---:|---:|---:|---:|---:|
| 6 | 1,26 | 1,60 | 2,18 | 3,43 |
| 50 | 1,32 | 1,79 | 2,75 | 5,99 |
| 1000 | **1,33** | 1,82 | 2,85 | 6,63 |

Bij `ρ̄ = 0,749` is de asymptoot **1,33**. Van zes naar duizend crypto-perps
brengt `N_eff` van 1,27 naar 1,33.

> **Breedte binnen crypto is wiskundig uitgeput.** Dat is precies wat F10 al
> vaststelde — *"breadth meet nul-edge preciezer"* — maar nu met het getal
> erbij dat zegt waaróm: je voegt geen onafhankelijke weddenschappen toe, je
> voegt kopieën van dezelfde weddenschap toe.

### 2.5 Wat de Fundamentele Wet eist

`IR ≈ IC · √(breedte)`. Voor `IR = 1,69`:

| IC | benodigde onafhankelijke bets |
|---:|---:|
| 0,02 | 7.140 |
| 0,03 | 3.173 |
| 0,05 | 1.142 |

Beschikbaar: `N_eff × bars = 1,27 × 1.742 =` **2.204**.

En bij aggregatie naar weekbars: `1,27 × 249 =` **316**.

> **Daarmee staat de tijdschaalhypothese in haar scherpste vorm.** Zij deelt de
> breedte door zeven. Om dat terug te verdienen moet de IC met **√7 = 2,65**
> stijgen. §2.3 laat zien dat er geen aantoonbare trendstructuur is die dat zou
> dragen.

### 2.6 De staat van het instrument

Uit `reports/AUDIT_BRUIKBAARHEID_2026-09-19.md`, alle reproduceerbaar:

| Bevinding | Bewijs |
|---|---|
| Drie tests rood op `main` | `test_regime_overlay`, `test_dust_breaks_relative_limits`, `test_docs_claim_only_what_exists` |
| Verse kloon kan geen config samenstellen | `conf/env/` niet getrackt → `MissingConfigException: Could not find 'env/dev'` |
| Paper-trade-poort draait nul tests | `pytest -m e2e` → exit 5; mét `--timeout` → exit 4 |
| De grendel uit fase 9 spoor 1 ontbreekt | `I_UNDERSTAND_THIS_TRADES_REAL_MONEY` bestaat nergens |
| Vier klokken in één repository | `bar_seconds` 5 / 3600, `interval: 1h`, domein `1d` |
| Fase 10 stap 16 en 17 niet uitgevoerd | `apps/run_daily_decision.py`, `conf/portfolio/constraints.yaml` ontbreken |
| Dekking 60,66 % tegen `fail_under = 70` | 15.181/25.027 statements |

---

## 3. DE HYPOTHESE VAN DE EIGENAAR, EERLIJK GEWOGEN

De eigenaar stelt: *grotere tijdframes = betere voorspelbaarheid in verband met
trend.* Vijf oordelen, elk met zijn grond:

| # | Oordeel | Grond |
|---|---|---|
| 1 | **Toelaatbaar binnen het mandaat** — en dat is niet triviaal | Een weekbar is een **functie van** `daily_ohlcv` op de toegestane frequentie. AD-23 sluit *fijnere* waarneming uit (DI-18), niet *grovere* aggregatie. Dit is een echte opening, geen heropening van een gesloten besluit |
| 2 | **Gratis in vermogenstermen** | §2.2 — de drempel blijft 1,69 bij elke resolutie |
| 3 | **Verbetert de verdelingsaannames** | §2.2 — kurtosis 7,61 → 2,75, skew −0,60 → +0,20 |
| 4 | **Kost een factor 7 aan breedte** | §2.5 — 2.204 → 316 bets; vereist IC × 2,65 om break-even te draaien |
| 5 | **Geen aantoonbare trendstructuur om die IC-stijging te dragen** | §2.3 — max abs z = 1,40 over 35 toetsen; F7 en F8 zijn hier eerder op gevallen |

**Conclusie.** De hypothese is toelaatbaar en goedkoop, en zij verdient een
eerlijke toets — maar zij kan het gat van §2.1 niet dichten, en zij mag niet de
dragende aanname van deze fase worden. Zij krijgt in Stage D **twee trials** en
een vooraf geregistreerde beslisregel, en het antwoord `UNPROVEN` is daar een
geldige en verwachte uitkomst.

**Wat de hypothese wél kan dragen, en waar zij dus op wordt getoetst:** niet
"betere voorspelbaarheid" maar **kostenamortisatie**. Vaste kosten van 13,0 bps
per round trip worden bij twintig keer langer aanhouden twintig keer minder vaak
betaald, en impact schaalt met `√(notional)` en dus sublineair. Dat is de enige
route waarlangs de hypothese meetbaar geld oplevert, en zij is begrensd op de
0,12 Sharpe uit §2.1.

---

## 4. HET BESLUIT DAT DEZE FASE AAN DE EIGENAAR VOORLEGT

§2.4 en §2.5 laten één sluitende redenering toe:

1. Het gat naar de poort is ≈ 1,86 Sharpe (§2.1).
2. Kosten en funding samen zijn ≈ 0,18 daarvan (§2.1).
3. Het restgat moet uit signaal komen: `IR ≈ IC · √(breedte)` (§2.5).
4. De breedte is 2.204 bets en loopt binnen crypto tegen een asymptoot van
   `N_eff = 1,33` (§2.4).
5. Aggregatie verlaagt de breedte en verhoogt de IC niet aantoonbaar (§2.2, §2.3).

> **Daarmee is er binnen `conf/governance/measurement_domain.yaml` geen
> configuratie die de poort kan halen.** Dat is geen pessimisme; het is punt 4
> plus punt 5.

Er zijn precies drie uitwegen, en de keuze ertussen is een **mandaatbesluit** en
geen implementatiekeuze. Stage C dwingt die keuze af als **AD-27**:

**C-1 — Verbreed het meetdomein naar meerdere activaklassen.** De enige ingreep
die `N_eff` materieel verhoogt, want zij voegt betas toe in plaats van namen.
Prijs: een vierde gecertificeerde bron, een domeinwijziging op AD-23, en het
heropenen van de heropeningscondities van F13–F18 en F20 die daar expliciet naar
verwijzen. **Let op de val:** S&P-large-caps zijn óók één beta; de winst zit in
klassen die onderling laag correleren (aandelen × rente × grondstoffen × FX ×
crypto), niet in méér namen binnen één klasse.

**C-2 — Verleng de kalendertijd.** §2.2 zegt dat kalendertijd bindt. Bij `M = 25`
is voor een wáre Sharpe van 0,40 ongeveer **87 jaar** nodig; bij `M = 2` nog
altijd **29,7 jaar**. Op zes crypto-perps met vijf jaar historie is dit geen
route. Op een cross-asset universum met decennia PIT-historie is het dezelfde
ingreep als C-1, langs de andere as.

**C-3 — Sluit het programma af.** `fase_9_besluit_...` §9 schrijft dit expliciet
voor en noemt het *"een volwaardige oplevering en geen nederlaag"*: leg vast wat
bewezen is, wat meegaat naar een volgend project (de PIT-datalaag, de
kostenboekhouding, het pre-registratiemechanisme, het falsificatieregister), en
wat definitief niet werkte.

**Deze fase neemt dat besluit niet. Zij maakt het onontkoombaar en documenteert
de drie opties met hun prijs.**

---

## 5. MEETCONVENTIES

Niet hier. `docs/MEASUREMENT_CONTRACT.md` is de enige plaats waar het meetvenster,
de annualisatie, de standaardfouten, de verschiltoets, de DSR-handtekening, de
rendementsconventie en de purge/embargo-regel staan.

Eén toevoeging die deze fase nodig heeft en die daar nog niet staat, wordt in
stap 10 aan het contract toegevoegd: **de definitie van `N_eff` en de verplichting
om hem naast elke breedte-claim te rapporteren.**

---

## 6. TRIAL-BUDGET

`M_new = 25` (AD-24, bevroren). Geboekt: **5** (H-10.1: 4; H-10.2: 1; H-10.3
vervallen: 0). **Resterend: 20.**

Deze fase begroot:

| Stage | Post | Trials |
|---|---|---:|
| A | instrumentreparatie | **0** — repareert, selecteert niets |
| B | fase 10 stap 16/17 afmaken | **0** — uitvoering van een genomen mandaat |
| C | breedte-diagnose en AD-27 | **0** — meet de meetopstelling, geen hypothese |
| D | H-11.1, de tijdschaalhypothese | **2** |
| E | oplevering | **0** |
| | **totaal** | **2** |

**Resterend na fase 11: 18.**

Dat Stage C nul trials kost, is geen boekhoudkundige truc en moet in stap 13.4
expliciet worden verantwoord: een `N_eff`-meting en een variantieratio zijn
**eigenschappen van de data**, niet van een strategie. Zij kunnen geen kandidaat
promoveren en er is geen keuze die zij selecteren. Een trial is de prijs van een
keuze; hier wordt er geen gemaakt.

---

# STAPSGEWIJZE UITVOERING

# STAGE A — HET INSTRUMENT

*Blokkeert alles. Elke meting uit C en D op een instrument dat niet
reproduceerbaar start, is oninterpreteerbaar — niet fout, maar zonder betekenis.*

---

### Stap 1: Reproduceer de nulmeting en zet `PROJECT_STATE` recht

**Files:**
- Modify: `docs/PROJECT_STATE.md` §7
- Create: `reports/phase11_nulmeting.md`

- [ ] **Stap 1.1 — Draai het verificatieblok uit §"VERIFICATIEBLOK" integraal.**
  Noteer van **elk** commando de exitcode, niet de laatste regel uitvoer. Een
  pipe naar `tail` verbergt de exitcode; dat is tijdens de CI-sanering van
  2026-09-18 gebeurd en het maakte zes poorten ten onrechte groen, en het is
  tijdens de audit van 2026-09-19 nóg een keer gebeurd.

  Verwacht op `f50f6ca`: negen statische poorten exit 0; de testsuite **3
  FAILED**; `pytest -m e2e ...` **exit 5**; dekking **60,66 %**.

- [ ] **Stap 1.2 — Reproduceer de vijf metingen uit §2** (Sharpe-begroting,
  resolutietabel, variantieratio, correlatie/`N_eff`, Fundamentele-Wet-tabel) en
  leg ze vast in `reports/phase11_nulmeting.md` met het commando en de
  `git_sha`. Wijkt er iets af, **stop en rapporteer** — §2 is de grondslag van
  elke latere stap en een afwijking daar verandert de fase.

- [ ] **Stap 1.3 — Corrigeer `docs/PROJECT_STATE.md` §7.** Het
  verificatieblok belooft *"0 failed, 4 xfailed, 0 xpassed"*. Dat is sinds
  `f50f6ca` onwaar. Vervang het door de gemeten stand met een verwijzing naar de
  DI-regels uit stap 3. Een statusdocument dat een groen belooft dat er niet is,
  is een poort die niet rood kan worden.

- [ ] **Stap 1.4 — Commit.** `docs(phase11): de nulmeting, en wat PROJECT_STATE beloofde dat niet waar was`

---

### Stap 2: De grendel — onvoorwaardelijk, en dit gaat vóór al het andere

Fase 9 spoor 1 noemde dit *"het enige punt in de hele doorlichting dat directe
schade kan veroorzaken"* en begrootte het op een uur. Drie van de vijf items
staan nog open. De blootstelling is vandaag begrensd doordat een verse kloon
toch niet start (stap 4) — **maar dat is een toevallige grendel, en hij ontbreekt
juist op de enige machine waar `conf/env/` wél staat.**

**Files:**
- Modify: `src/tradebot/live/engine.py`
- Create: `tests/unit/test_live_start_latch.py`
- Create: `tests/integration/test_sigusr1_halts_a_running_engine.py`

- [ ] **Stap 2.1 — `LiveEngine` crasht bij opstart zonder `I_UNDERSTAND_THIS_TRADES_REAL_MONEY`.**
  Geen waarschuwing, geen logregel — een `ConfigContractError`. De vlag geldt
  uitsluitend voor `EngineMode.LIVE`; paper- en backtestmodus draaien zonder.
  Test eerst, zie hem falen, implementeer (R-11).

- [ ] **Stap 2.2 — De fat-finger-reset op `live/engine.py:653-656` verdwijnt of
  wordt een geconfigureerde, geregistreerde uitzondering** met een drempel uit
  `conf/` en een testbewijs dat hij bindt.

- [ ] **Stap 2.3 — Eén test die het SIGUSR1-pad op een LOPENDE engine aanroept**
  — niet op `HaltStore`. De bestaande 20+ kill-switch-tests raken de
  live-implementatie niet. Op Windows is de fallback `SIGTERM`
  (`engine.py:306-310`); de test dekt het pad dat op het doelplatform draait en
  markeert het andere expliciet als ongedekt.

- [ ] **Stap 2.4 — Acceptatie.** Een reviewer kan in vijf minuten aanwijzen
  waarom een onbedoelde start geen order kan plaatsen. Lukt dat niet, dan is
  deze stap niet af.

- [ ] **Stap 2.5 — Commit.** `feat(live): de grendel -- een onbedoelde start kan geen order plaatsen`

---

### Stap 3: De drie rode tests

**Files:**
- Modify: `docs/DEFERRED_ISSUES.md`
- Modify: `tests/regression/test_dust_breaks_relative_limits.py`
- Modify: `tests/unit/test_docs_claim_only_what_exists.py`

- [ ] **Stap 3.1 — `test_dust_breaks_relative_limits`.** De `config_hash` ging
  van `1b60cb664fbf9a2a` naar `9961e1613bc907a5` omdat AD-26 het risicobudget
  wijzigde. **Dit is geen testonderhoud.** `config_hash` is het mechanisme
  waarmee `OrderRouter.build_orders()` een `RiskDecision` uit een ander
  risicoregime weigert; een gewijzigde hash is precies het signaal waarvoor hij
  bestaat. Werk de pin bij **met een commentaarblok** dat AD-26 noemt, de oude
  hash bewaart, en zegt onder welke voorwaarde hij opnieuw hoort te wijzigen.
  Voeg een DI-regel toe.

- [ ] **Stap 3.2 — `test_docs_claim_only_what_exists`.** Wordt opgelost door
  stap 4; haal `conf/env/prod.yaml` uit `KNOWN_ABSENT` in dezelfde commit als de
  tracking. Niet eerder — anders is de test tussentijds rood om een derde reden.

- [ ] **Stap 3.3 — `test_regime_overlay` (DI-30) — NIET repareren.** REGEL V
  houdt niet meer onder AD-26 en de test doet precies waarvoor hij is
  geschreven. Er zijn twee inhoudelijke uitkomsten en beide zijn een
  **eigenaarsbesluit**: (a) REGEL V geldt niet meer, en dan verandert wat H2 kan
  meten en moeten `docs/EXPANSION_RESEARCH_2026-08-10.md` en de H2-rapportage
  mee; of (b) het mandaat wordt bijgesteld zodat de vol-target weer bindt, en
  dan is AD-26 niet af. Leg beide voor in `reports/phase11_open_besluiten.md`
  met het gemeten getal (19.166 tegen 19.771, 3,1 % op 1 % tolerantie).
  **Wat NIET mag:** de tolerantie oprekken tot de test groen is.

- [ ] **Stap 3.4 — Commit.** `fix(tests): twee rode tests geregistreerd, de derde is een besluit`

---

### Stap 4: `conf/env/` in versiebeheer

**Files:**
- Modify: `.gitignore`
- Create (tracked): `conf/env/{dev,ci,staging,prod}.yaml`
- Create: `tests/unit/test_every_config_group_composes.py`

- [ ] **Stap 4.1 — Bewijs het defect eerst.** Kloon de repo naar een tijdelijke
  map en draai `compose(config_name="config")`. Verwacht:
  `MissingConfigException: Could not find 'env/dev'`. Leg de uitvoer vast.

- [ ] **Stap 4.2 — `.gitignore` regel 34.** `env/` is geschreven voor
  virtualenvs en slikt de Hydra-configgroep. Maak hem specifiek (`/env/` of
  `.venv/`), zodat `conf/env/` overleeft.

- [ ] **Stap 4.3 — Commit de vier profielen zonder credentials.** Elke sleutel
  die een geheim draagt, wordt een `${oc.env:...}`-verwijzing. `prod.yaml`
  draagt op dit moment `bar_seconds: 5`; **verwijder dat veld niet hier** — het
  hoort bij stap 7, en twee sporen in hetzelfde bestand is hoe deze repository
  aan vijf plaatsen met vier waarden kwam. Zet er een `# TODO AD-22 (stap 7)`
  bij en laat stap 7 hem weghalen.

- [ ] **Stap 4.4 — Een test die elke configgroep samenstelt.** Voor elk profiel
  in `conf/env/`: `compose` slaagt en levert een geldige `Config`. Bewijs dat de
  test rood wordt door één profiel tijdelijk te hernoemen.

- [ ] **Stap 4.5 — Commit.** `fix(conf): de Hydra-env-groep stond niet in versiebeheer, en een verse kloon startte niet`

---

### Stap 5: De paper-trade-poort laten meten

**Files:**
- Modify: `pyproject.toml` (`markers`)
- Modify: `tests/e2e/test_paper_trade_smoke.py`
- Modify: `.github/workflows/paper-trade-ci.yml`
- Modify: `requirements-dev.lock`

- [ ] **Stap 5.1 — Registreer de marker `e2e`** in
  `[tool.pytest.ini_options] markers`. `--strict-markers` staat aan, dus een
  niet-geregistreerde marker is nu al een fout — hij werd alleen nooit bereikt
  omdat `-m e2e` eerst nul tests selecteerde.

- [ ] **Stap 5.2 — Zet `@pytest.mark.e2e` op de smoke-test** (naast `slow`, niet
  in plaats van).

- [ ] **Stap 5.3 — `--timeout=3600`.** Pin `pytest-timeout` in
  `requirements-dev.lock` **of** haal de vlag weg. Kies één en zeg in de
  workflow waarom.

- [ ] **Stap 5.4 — De fixture draait op dagbars.** `_make_bars` bouwt nu
  `freq="1h"`; dat is een klok die AD-22 heeft afgeschaft.

- [ ] **Stap 5.5 — Bewijs dat de poort rood kan worden.** Eén bewust rode run —
  bijvoorbeeld een circuit-breaker die afgaat — met de uitvoer in het
  stap-rapport. Zonder dat bewijs is deze stap niet af, want dat is precies de
  faalwijze die deze stap repareert.

- [ ] **Stap 5.6 — Commit.** `fix(ci): de paper-trade-poort draaide nul tests, en dat kon niemand zien`

---

### Stap 6: Eén omgeving, één oordeel

**Files:**
- Create: `scripts/check_interpreter.py`
- Modify: de zes poortscripts, `.github/workflows/*.yml`

- [ ] **Stap 6.1 — `scripts/check_interpreter.py`** vergelijkt de draaiende
  versies van `mypy`, `ruff` en `pytest` met wat `requirements-dev.lock` pint, en
  faalt met een boodschap die beide noemt.

  De grond: tijdens de audit van 2026-09-19 gaf dezelfde mypy-scope **34 fouten**
  op de globale interpreter (mypy 1.17.0) en **Success** op de referentie-venv
  (mypy 2.3.1). Datzelfde verschil trof de volle-boommeting (1.042 tegen 879).
  DI-16 is nooit gesloten, alleen verplaatst.

- [ ] **Stap 6.2 — Roep hem aan** vanuit de zes poortscripts en als eerste stap
  van elke workflow.

- [ ] **Stap 6.3 — Commit.** `feat(ci): DI-16 sluiten -- een meting op de verkeerde interpreter is nu een fout`

---

# STAGE B — FASE 10 AFMAKEN

*Twee stappen die het vorige mandaat heeft besloten en niet heeft uitgevoerd.
Nul trials: dit is uitvoering, geen selectie.*

---

### Stap 7: De dagelijkse runner (fase 10 stap 16)

**Let op — de masterprompt van fase 10 verwijst naar een bezet nummer.** Stap
16.4 schrijft "AD-26" voor, maar AD-26 is bij de merge van 2026-09-18 vergeven
aan het risicobudget (`docs/ARCHITECTURAL_DECISIONS.md:1212`, met het
hernummeringscommentaar op regel 1205-1208). **Deze stap schrijft AD-27 niet —
dat is Stage C — maar AD-28.** Werk de fase-10-prompt bij zodat de volgende
lezer niet dezelfde botsing vindt.

- [ ] **Stap 7.1 — Inventariseer wat `live/` doet dat de runner óók moet doen.**
  Vóór elke verwijdering. Minimaal: de haltketen en haar `constraint_order` (de
  meest waardevolle logica in `live/`, mag onder geen voorwaarde verdwijnen), de
  PIT-verificatie bij inlezen, de foutpaden en het fail-fast-gedrag, de logging
  en de artefactschrijving.

- [ ] **Stap 7.2 — `apps/run_daily_decision.py` (≤ 80 LOC, R-6).** Vier
  verantwoordelijkheden: lees de PIT-store tot en met gisteren; bereken de
  exposures voor vandaag; pas de haltketen toe in de bestaande volgorde; schrijf
  het besluit als artefact. Geen loop, geen wachttijden, geen sessiebeheer.

- [ ] **Stap 7.3 — Toon gedragsequivalentie.** Draai de runner en de oude loop
  over dezelfde historische bar; exposures en haltbesluiten zijn **identiek**.
  Zonder deze test is de vervanging niet aantoonbaar gedragsneutraal.

- [ ] **Stap 7.4 — `FeedConfig.bar_seconds` vervalt** (AD-22, mandaatbesluit
  B-1), en in **dezelfde commit** worden de overige klokken opgeruimd:
  `conf/env/prod.yaml:12`, `conf/conf_config.yaml:136`, `conf/config.yaml`
  `live.interval`.

- [ ] **Stap 7.5 — Breid `scripts/check_domain_consistency.py` uit naar
  `bar_seconds`**, óók in dezelfde commit. De poort documenteert op regel 45-50
  zelf dat hij deze overtreding bewust niet ziet zolang stap 16 niet is
  uitgevoerd. Die reden vervalt hier, en tot dat moment stond de domeinpoort
  groen terwijl zij de enige overtreding die ertoe doet per constructie niet kon
  waarnemen.

- [ ] **Stap 7.6 — Schrijf AD-28** — *"Één besluitmoment per dag; `live/` wordt
  een runner"*. Afgewezen alternatief: `live/` behouden met de loop op 24 uur —
  dat houdt de sessie-, reconnect- en latentiepaden in leven zonder dat iets ze
  test.

- [ ] **Stap 7.7 — Commit.**

---

### Stap 8: De harde limieten (fase 10 stap 17)

- [ ] **Stap 8.1 — `conf/portfolio/constraints.yaml`** met
  `max_notional_for_uncalibrated_impact` en `max_weight_per_symbol`. Elke waarde
  krijgt de reden, de AD-verwijzing en de voorwaarde waaronder zij omhoog mag
  (een gekalibreerde `eta`, respectievelijk een gemeten concentratie-effect).

- [ ] **Stap 8.2 — Dwing beide af in `constraint_order`**, op een expliciete
  positie, met in het commentaar waaróm die positie: de notionallimiet vóór de
  impactberekening (anders wordt een ongeldige functie geëvalueerd), de
  gewichtslimiet ná de toestandspoort en vóór het vol-target (anders herstelt L7
  de concentratie die de limiet net wegnam).

- [ ] **Stap 8.3 — Vier tests**, zoals fase 10 stap 17.3 ze opsomt.

- [ ] **Stap 8.4 — Werk AD-2 en AD-3 bij naar `PERMANENT`** met een
  `Consequence`-sectie: welke uitspraken over kosten **niet** kunnen worden
  gedaan zolang `eta` ongekalibreerd is (`status: IMPACT_UNCALIBRATED`, een
  bovengrens uit de dagrange en geen schatting).

- [ ] **Stap 8.5 — Commit.**

---

### Stap 9: De dekkingsratchet, per pakket

- [ ] **Stap 9.1 — Bouw de ratchet** naar het patroon van
  `scripts/check_file_size.py`: dekking mag niet omlaag, `fail_under = 70`
  blijft staan als doel.

- [ ] **Stap 9.2 — Zet hem PER PAKKET, niet alleen op het totaal.** Anders koopt
  dekking in `validation/` (85,8 %) de ruimte vrij om `live/` op 37,5 % en
  `oms/router.py` op 27,9 % te laten staan — en dat zijn de pakketten waar in
  productie een order de deur uitgaat.

- [ ] **Stap 9.3 — Commit.**

---

# STAGE C — DE BREEDTE, EN HET BESLUIT DAT ERUIT VOLGT

*Nul trials. Deze stage meet de meetopstelling, niet een strategie.*

---

### Stap 10: `N_eff` wordt een gemeten, gerapporteerde grootheid

**Files:**
- Create: `src/tradebot/validation/breadth.py`
- Create: `tests/unit/test_breadth.py`
- Modify: `docs/MEASUREMENT_CONTRACT.md`

- [ ] **Stap 10.1 — Eén implementatie (R-3).** `validation/inference.py` draagt
  al `hac_variance_ratio` (de Bartlett-gewogen `eta_q` voor de Lo-SE). Dat is
  **niet** de Lo-MacKinlay VR(q)-toets. Bouw `breadth.py` met: `effective_n`
  (participation ratio én gelijke-correlatie, beide, want ze verschillen hier
  met 24 %), en `variance_ratio` met homoskedastische én
  heteroskedasticiteit-robuuste `z`. Documenteer in de docstring hoe beide zich
  verhouden tot `hac_variance_ratio`, zodat niemand later een derde bouwt.

- [ ] **Stap 10.2 — Negatieve controles.** Op i.i.d. ruis: `VR(q) → 1` en
  `N_eff → N`. Op een reeks met ingebouwde AR(1)-persistentie: `VR > 1` met de
  juiste `z`. Op perfect gecorreleerde kolommen: `N_eff → 1`. Elke test moet
  rood kunnen worden.

- [ ] **Stap 10.3 — Voeg `N_eff` toe aan `MEASUREMENT_CONTRACT.md`** met de
  verplichting hem naast elke breedte- of paneelclaim te rapporteren, naar het
  model van het `(n_obs, bars_per_year, t_years)`-drietal.

- [ ] **Stap 10.4 — Commit.**

---

### Stap 11: De breedte-diagnose op het huidige universum

**Files:**
- Create: `apps/run_breadth_diagnostics.py` (≤ 80 LOC)
- Create: `reports/phase11_breadth.md`, `artefacts/governance/phase11_breadth.json`

- [ ] **Stap 11.1 — Reproduceer §2.4 en §2.5** met de module uit stap 10, en leg
  vast: correlatiematrix, eigenwaarden, PC1-aandeel, beide `N_eff`-varianten,
  de asymptoot `1/ρ̄`, en de Fundamentele-Wet-tabel.

- [ ] **Stap 11.2 — Meet `N_eff` ook op de RESULTAATreeksen**, niet alleen op de
  returns: de vier tracks uit `phase5_revaluation.json`. Als die onderling nog
  hoger correleren dan de onderliggende namen, is de effectieve breedte van de
  ladder kleiner dan 1,27 en dan is dát het getal dat telt.

- [ ] **Stap 11.3 — Zet het naast F20.** `cm_carry` is onder meer geveld op
  `N_eff = 1,54`. Dit universum scoort lager. Schrijf die vergelijking
  uit — niet als retoriek maar als de toepassing van een bestaand, vooraf
  vastgelegd criterium op het eigen universum.

- [ ] **Stap 11.4 — Commit.**

---

### Stap 12: De verdeling van het gat — waar moet het vandaan komen

**Files:**
- Create: `reports/phase11_sharpe_begroting.md`

- [ ] **Stap 12.1 — Reproduceer §2.1** met provenance per regel: het
  poortvereiste uit `dsr.py`, L2/L3 uit `phase5_revaluation.json`, de
  funding-drag uit de gecertificeerde 8h-reeks.

- [ ] **Stap 12.2 — Los de ontaarding op L3 op, of registreer haar.** Op alle
  vier de tracks geldt `gross_sharpe == net_sharpe` op L3 terwijl `cost_fees`,
  `cost_spread`, `cost_impact` en `cost_funding` alle vier niet-nul zijn. Beide
  kunnen niet waar zijn. Zolang dit staat, is de vraag *"hoeveel van het verlies
  is kosten en hoeveel is signaal"* — de vraag waarvoor de hele ladder is
  gebouwd — op L3 niet beantwoordbaar uit het artefact.

- [ ] **Stap 12.3 — Rapporteer de halt-asymmetrie.** De long-only-tracks staan op
  **1.566 van 1.743 bars (89,8 %)** gehalteerd, de momentumtracks op nul. Een
  L3-Sharpe op een boek dat negen van de tien dagen vlak ligt, meet de
  resterende 10 %. De twee groepen zijn daarmee niet op dezelfde grootheid
  vergelijkbaar, en elke tabel die ze naast elkaar zet, zegt dat erbij.

- [ ] **Stap 12.4 — Commit.**

---

### Stap 13: AD-27 — het besluit wordt voorgelegd

**Files:**
- Modify: `docs/ARCHITECTURAL_DECISIONS.md` (AD-27)
- Create: `reports/phase11_mandaatbesluit.md`

- [ ] **Stap 13.1 — Schrijf AD-27** in de vorm die deze repository gebruikt:
  `Fase / Status / Bewaakt door`, dan `Besluit`, `Waarom`, `Het afgewezen
  alternatief`. Status is `VOORGELEGD — wacht op de eigenaar`, niet `actief`.

- [ ] **Stap 13.2 — Werk de drie opties uit** (C-1, C-2, C-3 uit §4), elk met:
  de gemeten grond, de prijs in data en tijd, welke regels uit
  `FALSIFICATION_REGISTER.md` erdoor heropenen, en wat er gebeurt als de keuze
  niet wordt gemaakt.

  Voor C-1 hoort er één waarschuwing bij, want zij is de val waarin dit project
  al eerder liep: **meer namen binnen één klasse verhoogt `N_eff` niet.** F13–F18
  zijn allemaal op S&P-large-caps gemeten — óók één beta. De winst zit in
  onderling laag gecorreleerde klassen, en de tabel uit §2.4 zegt hoeveel:
  bij `ρ̄ = 0,35` haal je `N_eff ≈ 2,8`, bij `ρ̄ = 0,15` haal je `≈ 6,6`.

- [ ] **Stap 13.3 — Reken per optie door wat zij met de poort doet.** Gebruik de
  DSR-tabel: bij welk `N_eff` en welke kalendertijd komt de vereiste Sharpe
  onder 0,8? Dat is het enige getal waarop de eigenaar kan kiezen.

- [ ] **Stap 13.4 — Verantwoord de nul trials** volgens §6.

- [ ] **Stap 13.5 — Commit.**

---

# STAGE D — H-11.1, DE TIJDSCHAALHYPOTHESE

*Twee trials. Vooraf geregistreerd. `UNPROVEN` is hier een geldige en verwachte
uitkomst, en dat staat vóór de run opgeschreven (R-10).*

---

### Stap 14: Pre-registratie

**Files:**
- Create: `conf/experiment/h11_1_bar_aggregation.yaml`
- Create: `artefacts/governance/preregistration_<hash>.json`

- [ ] **Stap 14.1 — Formuleer H-11.1** exact: *aggregatie van de dagbar naar een
  grovere bar verhoogt de netto Sharpe van de beste bestaande track, en het
  mechanisme is kostenamortisatie en niet verbeterde voorspelbaarheid.*

- [ ] **Stap 14.2 — Twee trials, niet meer.** `q ∈ {5, 20}` (week, maand).
  Een derde waarde is een derde trial en die is niet begroot. De keuze voor 5 en
  20 staat vóór de run vast en wordt niet op een uitkomst herzien (R-2).

- [ ] **Stap 14.3 — De beslisregel, vooraf.** `CONFIRMED` vereist alle vier:
  1. `delta_sharpe > 0` op de ontwikkelsample;
  2. het 95 %-CI sluit nul uit **na** deflatie met `M = 25`;
  3. hetzelfde teken op de poortsample (één lezing, R-7);
  4. de gemeten `turnover_delta` verklaart ten minste de helft van
     `delta_sharpe` — want de hypothese claimt kostenamortisatie, en een
     Sharpe-winst zonder omzetdaling is een ander effect dat niet is
     voorspeld.

  Ontbreekt er één, dan is het `UNPROVEN`. `REJECTED` blijft voorbehouden aan
  een aantoonbaar negatief effect.

- [ ] **Stap 14.4 — Leg de voorspelling vast** (R-10): op grond van §2.3 en §2.5
  verwacht deze prompt `UNPROVEN`, en de reden is dat de breedte door zeven
  respectievelijk twintig wordt gedeeld terwijl er geen aantoonbare
  trendstructuur is die de benodigde IC-stijging van √7 = 2,65 draagt.
  **Wijkt de meting af, dan is de meting het antwoord** en wordt deze
  voorspelling geciteerd als de weerlegde voorspelling die zij was.

- [ ] **Stap 14.5 — Bevries en commit.**

---

### Stap 15: De meting

**Files:**
- Create: `src/tradebot/backtest/aggregation.py`
- Create: `apps/run_h11_1_bar_aggregation.py` (≤ 80 LOC)

- [ ] **Stap 15.1 — Aggregatie is causaal.** Een `q`-bar sluit op de close van
  de laatste dagbar in het blok en is pas beschikbaar ná die close. Truncatie-
  én perturbatietest (R-1), elk met een negatieve controle.

- [ ] **Stap 15.2 — Blokgrenzen zijn vast, niet rollend.** Een rollend venster
  overlapt en maakt de observaties afhankelijk; dat is een andere toets met
  andere standaardfouten. Vaste grenzen, en de keuze van de startoffset is
  **geen** vrijheidsgraad die achteraf mag worden gekozen.

- [ ] **Stap 15.3 — Funding aggregeert door SOMMATIE, niet door de laatste
  waarde.** Dit is dezelfde val die stap 1B van fase 10 heeft gerepareerd:
  `asof_join(direction="backward")` geeft de laatst bekende rate — het juiste
  antwoord voor een feature, en ongeveer een derde van de werkelijke funding
  voor een kostenpost. Op een `q = 20`-bar is de fout een factor 60.

- [ ] **Stap 15.4 — Rapporteer per `q`:** `N_eff`, bets, netto en bruto Sharpe
  met SE, `turnover_delta`, de vier kostenposten afzonderlijk, en
  `breakeven_cost_bps`.

- [ ] **Stap 15.5 — Commit.**

---

### Stap 16: Het oordeel

- [ ] **Stap 16.1 — Pas de beslisregel uit 14.3 toe, letterlijk.** Boek de twee
  trials vóór het oordeel. Werk de ledger bij; `remaining` staat daarna op 18.

- [ ] **Stap 16.2 — Bij `UNPROVEN` of `REJECTED`: schrijf de F-regel.** Het
  falsificatieregister is het meest waardevolle bestand van deze repository;
  laat het niet verlopen. De heropeningsconditie moet ten minste één `source_id`
  uit `measurement_domain.yaml` noemen, of expliciet buiten het domein worden
  gemarkeerd.

- [ ] **Stap 16.3 — Bij `CONFIRMED`: stop en escaleer.** Dan is de voorspelling
  uit 14.4 weerlegd, en dat is een groter resultaat dan de hypothese zelf — het
  betekent dat §2.3 iets mist. Loop de VR-meting opnieuw na vóórdat er iets
  wordt gepromoveerd.

- [ ] **Stap 16.4 — Commit.**

---

# STAGE E — OPLEVERING

### Stap 17: Exit-rapport

- [ ] **Stap 17.1 — `reports/phase11_exit_report.md`** met: de stand per
  exit-criterium (gemeten, niet beweerd), de trial-rekening, en de drie opties
  uit AD-27 met hun prijs.

- [ ] **Stap 17.2 — Werk `docs/PROJECT_STATE.md` bij:** §2 (wat aantoonbaar waar
  is), §3 (wat aantoonbaar niet waar is — hier hoort §2.4 van deze prompt in),
  §5 (openstaande besluiten — AD-27) en §6 (de volgende drie stappen).

- [ ] **Stap 17.3 — Fase 10 krijgt alsnog zijn exit-rapport**, want stap 7 en 8
  sluiten haar laatste twee stappen.

- [ ] **Stap 17.4 — Commit.**

---

## EXIT-CRITERIA

| # | Criterium | Bewijs |
|---|---|---|
| E1 | Een verse kloon stelt elk configprofiel samen | `compose` op alle vier profielen in een tijdelijke kloon |
| E2 | `pytest` op de CI-selectie geeft **0 FAILED** | commandouitvoer met exitcode |
| E3 | De paper-trade-poort draait ≥ 1 test en kan aantoonbaar rood worden | één bewust rode run in het rapport |
| E4 | `LiveEngine` weigert te starten in `LIVE` zonder de vlag | test + handmatige demonstratie |
| E5 | Eén barresolutie in de hele repository | `check_domain_consistency.py --strict` mét `bar_seconds` in de regex |
| E6 | `apps/run_daily_decision.py` bestaat, ≤ 80 LOC, gedragsequivalent | equivalentietest |
| E7 | Beide limieten uit stap 8 staan in `constraint_order` en binden | vier tests |
| E8 | `N_eff` staat in `MEASUREMENT_CONTRACT.md` en wordt gerapporteerd | het contract + `phase11_breadth.json` |
| E9 | AD-27 ligt er met drie doorgerekende opties | `reports/phase11_mandaatbesluit.md` |
| E10 | H-11.1 heeft een oordeel, twee geboekte trials, en bij ≠ `CONFIRMED` een F-regel | ledger + register |
| E11 | Dekkingsratchet per pakket actief | `git diff` + één bewust rode run |
| E12 | Geen drempel, ratchet of ignore-lijst is versoepeld | `git diff` over `pyproject.toml` en de ratchet-scripts, expliciet getoond |
| E13 | Elke niet-gesloten post staat als DI-regel | het diff van `DEFERRED_ISSUES.md` |
| E14 | `PROJECT_STATE.md` §7 belooft geen groen dat er niet is | het diff |

---

## FENCES — WAT DEZE OPDRACHT NIET AANRAAKT

| Niet aanraken | Waarom |
|---|---|
| `backtest/accounting.py`, `backtest/engine.py` | De authoritative keten. Stap 12.2 **meet** de L3-ontaarding en repareert haar niet; een wijziging hier maakt elk eerder resultaat onvergelijkbaar |
| `artefacts/governance/holdout_lock.json` | Eén lezing, en alleen door stap 16 via R-7 |
| `docs/FALSIFICATION_REGISTER.md` F1–F20 | Uitsluitend appenden. Geen regel wordt gewijzigd, verzwakt of verwijderd |
| `validation/gates.py` | Geen `force=`, geen `override=`, geen `warn_only=`, geen deelscore |
| `portfolio/legacy_sizing.py` | Geregistreerd onbereikbaar, ongewijzigd verhuisd zodat de Phase 3-baseline herrekenbaar blijft (DI-10) |
| `conf/risk/default.yaml` | AD-26 is een eigenaarsbesluit. Stap 3.3 legt de consequentie vóór, wijzigt de waarden niet |
| De vier `xfail(strict=True)`-killgates | Minder dan vier betekent dat er een is uitgeschakeld, niet dat er iets is opgelost |

---

## REGELS

**R-1 — Causaliteit is een test, geen intentie.** Truncatie- **én**
perturbatietest, elk met een negatieve controle die bewijst dat de test rood kan
worden.

**R-2 — Elke parameter kost een trial, en elke tak van een beslisboom ook.**
Registreer vóór de run. Een parameter die op grond van een uitkomst wordt
gewijzigd, kost retroactief ook een trial voor zijn voorganger.

**R-3 — Eén implementatie per statistische grootheid.** Sharpe-SE, DSR,
Sharpe-verschil en vanaf stap 10 ook `N_eff` en `variance_ratio` komen uit
`src/tradebot/validation/`. Een tweede implementatie is een defect, ook als zij
hetzelfde getal geeft.

**R-4 — Geen bestand boven 800 LOC.** Splits op verantwoordelijkheid.

**R-5 — Fail fast, met een reden** die uitlegt welke aanname is geschonden en
waarom die aanname bestaat.

**R-6 — Apps blijven onder 80 LOC.** Een app is een compositie, geen logica.

**R-7 — De poortsample wordt per hypothese ten hoogste eenmaal gelezen**, vóór de
run geregistreerd in `holdout_lock.json`.

**R-8 — Een getal zonder onzekerheid is geen bevinding.** Elke Sharpe, elk
verschil en elk conditioneel gemiddelde krijgt een SE, een interval en — op
paneeldata — de geclusterde en gedefleerde lezing ernaast.

**R-9 — Dit document is Nederlands; code, docstrings, commits en artefactsleutels
blijven Engels.** Waar een stap een letterlijke commitboodschap geeft, gebruik je
die verbatim.

**R-10 — Een verwachting in dit document is geen resultaat.** Wijkt de meting af,
dan is de meting het antwoord en wordt de verwachting geciteerd als de weerlegde
voorspelling die zij was. Dit geldt nadrukkelijk voor §2.3 en stap 14.4.

**R-11 — TDD, zonder uitzondering.** Test eerst, zie hem falen met de verwachte
fout, implementeer, zie hem slagen, commit. Een test die bij de eerste run
slaagt, test niet wat je denkt.

**R-12 — Commit per stap, niet per stage.**

**R-13 — Meet op de referentie-interpreter.** `D:/venv/tradebot/Scripts/python.exe`.
Vanaf stap 6 dwingt `scripts/check_interpreter.py` dit af; daarvóór doe je het
met de hand. Noteer van elk poortcommando de **exitcode**, niet de laatste regel
uitvoer.

---

## VERIFICATIEBLOK — draai dit integraal, vóór en ná

```bash
PY="D:/venv/tradebot/Scripts/python.exe"

"$PY" -c "import tradebot; print(tradebot.__file__)"
"$PY" -m mypy --version; "$PY" -m ruff --version; "$PY" -m pytest --version

"$PY" -m ruff check src/ apps/ tests/                                     ; echo "EXIT=$?"
"$PY" -m mypy src/tradebot/schemas/ src/tradebot/utils/ apps/ --ignore-missing-imports ; echo "EXIT=$?"
"$PY" -m mypy --strict src/tradebot/schemas/config.py src/tradebot/utils/failfast.py   ; echo "EXIT=$?"

"$PY" scripts/reachability_map.py --strict        ; echo "EXIT=$?"
"$PY" scripts/check_file_size.py                  ; echo "EXIT=$?"
"$PY" scripts/check_hardcoded_params.py --strict  ; echo "EXIT=$?"
"$PY" scripts/audit_fallbacks.py --strict         ; echo "EXIT=$?"
"$PY" scripts/check_banned_methods.py --strict    ; echo "EXIT=$?"
"$PY" scripts/check_domain_consistency.py --strict; echo "EXIT=$?"

"$PY" -m pytest -m "not slow and not regression" -p no:randomly --cov=src/tradebot --cov-report=term -q ; echo "EXIT=$?"
"$PY" -m pytest -m e2e tests/e2e/test_paper_trade_smoke.py -q             ; echo "EXIT=$?"
"$PY" apps/run_gates.py --out artefacts/governance/research_gates.json    ; echo "EXIT=$?"

npx --yes markdownlint-cli2@0.13.0 $(git ls-files '*.md') --config .markdownlint.json ; echo "EXIT=$?"
```

**Geen enkel commando hierboven wordt door een pipe gevoerd.** Dat is geen
stijlvoorkeur: `EXIT=$?` na een pipe rapporteert de exitcode van `tail`, en dat
heeft in deze repository al twee keer een rode poort groen laten lijken.

---

## VERBODEN ZETTEN

* Een drempel verruimen omdat het resultaat tegenvalt. `test_regime_overlay` is
  de lopende test van die norm.
* `fail_under` verlagen. De drempel is de eis; het cijfer is de werkelijkheid;
  het verschil is het werk.
* Een derde `q`-waarde toevoegen aan H-11.1 omdat de eerste twee het niet
  haalden. Dat is de zet die pre-registratie uitsluit.
* De poortsample een tweede keer lezen.
* Infrastructuur bouwen (champion/challenger, shadow, MRM, dashboards) vóór
  AD-27 is beslist. Dat is de volgorde-fout, en hij kostte dit project al
  ~1.560 regels plus acht manifests.
* Een niet-uitgevoerde meting als negatief resultaat presenteren, of een
  onderpowerde toets als falsificatie. §2.3 is expliciet *geen* falsificatie van
  de tijdschaalhypothese.
* De 60-daagse klok starten op de fase-3-baseline. `PROJECT_STATE.md` §5.4 noemt
  dat al "een systeem waarvan bekend is dat het geld verliest". Dat is een
  geldige operationele test en een ongeldige rendementsverwachting, en de
  afstand tussen die twee lezingen is waar dit soort projecten ontsporen.
* Meer crypto-namen toevoegen om de breedte te vergroten. §2.4: de asymptoot is
  1,33.

---

## STARTINSTRUCTIE

1. Draai het verificatieblok. Leg elke exitcode vast. Dit is stap 1.1 en er gaat
   niets aan vooraf.
2. Reproduceer de vijf metingen uit §2. Wijkt er iets af, stop en rapporteer.
3. Werk Stage A af in volgorde. Stage A blokkeert B, C en D.
4. Stage B en C mogen parallel, mits de bestandsverzamelingen disjunct zijn —
   controleer dat, neem het niet aan.
5. Stage D pas na Stage C, want AD-27 bepaalt of H-11.1 nog relevant is.

Rapporteer na elke stap: welke test faalde, met welke fout, wat de implementatie
werd, welke test slaagde, en de commit-SHA. Rapporteer na elke stage de stand van
`remaining` en de resterende exit-criteria.
