# MASTER-PROMPT: FASE 10 — HERSTART OP DAGBARS, VOLATILITEIT ALS TOESTAND

> **Fase:** 10, volgt op `fase_9_opschoning_en_consolidatie.md` · **Prioriteit:** P0 · **Revisie:** 2 (quant-review + meetdomein teruggebracht tot de dagbar)
> **Brainstormgrondslag:** `brainstormen.md` (Phase 0 — repository understanding & quant architecture brainstorm)
> **Bindende brondocumenten:** `docs/PROJECT_STATE.md`, `docs/FALSIFICATION_REGISTER.md`, `docs/ARCHITECTURAL_DECISIONS.md` (AD-13 t/m AD-21), `docs/DEFERRED_ISSUES.md`, `reports/phase5_exit_report.md` §10, `reports/GARCH_VS_EWMA_COMPETITION.md` §2.3 en §9, `reports/M0_VS_HMM_BENCHMARK.md` §10
> **Voorwaarde:** deze fase voert drie mandaatbesluiten uit die de eigenaar heeft genomen. Zij mag geen vierde besluit nemen.
>
> **Voor uitvoerders:** stappen gebruiken checkbox-syntaxis (`- [ ]`). Elke stap eindigt in een commit. Werk de stappen in volgorde af; Stage A blokkeert alles.

---

## 0. WAT REVISIE 2 VERANDERT

Twee soorten wijzigingen. Lees ze voordat je iets uitvoert, want beide raken stappen die in revisie 1 al leken vast te staan.

### 0.1 Het meetdomein is teruggebracht tot de dagbar

Revisie 1 sloot de sub-daagse informatiebronnen, maar bléef ze beschrijven: als uitgesloten heropeningscondities, als markerlijst in een scanner, als hertestbare alternatieven, als proxydiscussie. Daarmee bleef de fijnere waarneming een levend onderwerp in het document dat haar afschaft.

Revisie 2 draait dat om. Het meetdomein wordt **positief** gedefinieerd — precies drie toegestane bronnen, één observatiefrequentie — en alles wat daarbuiten valt, is geen uitgestelde vraag maar een niet-bestaande vraag. De poort in stap 2 is daarom een **whitelist op het meetdomein** in plaats van een blacklist op verboden woorden. Een whitelist heeft twee eigenschappen die een blacklist mist: zij is volledig zonder de uitgesloten ruimte te hoeven opsommen, en zij kan niet worden omzeild door een nieuwe naam voor hetzelfde te bedenken.

### 0.2 Elf correcties uit de quant-review

Revisie 1 was governance-technisch streng en statistisch te losjes. De volgende defecten zijn gecorrigeerd; elk verwijst naar de stap waarin de correctie is uitgewerkt.

| # | Defect in revisie 1 | Correctie | Stap |
|---|---|---|---|
| Q1 | `assign_by_variance` bepaalt de toestand van bar `t` uit σ̂_t en uit een expanding kwantiel dat bar `t` zélf bevat. De causaliteitstest toetst alleen truncatie-invariantie en ziet dit niet. | Toestand van bar `t` gebruikt uitsluitend σ̂ en kwantielen over bars ≤ `t−1`. Nieuwe test die een gelijktijdige lek aantoont. | 5 |
| Q2 | Sharpe-standaardfouten zijn `1/√T`. Dat veronderstelt normaliteit en nul autocorrelatie; dagelijkse crypto-P&L heeft beide niet. | Lo (2002)-correctie voor scheefheid en kurtosis, en Newey–West voor autocorrelatie. Één implementatie, verplicht voor elke gerapporteerde Sharpe. | 4A |
| Q3 | De laagovergangen worden getoetst met een gepaarde t op rendementsverschillen. Een Sharpe-verschil is geen rendementsverschil. | Ledoit–Wolf (2008) robuuste Sharpe-verschiltoets met HAC-kern en gestudentiseerde blokbootstrap. | 4A, 10 |
| Q4 | De nulmetingstabel poolt 10.400 symbool-bars en rapporteert t-statistieken alsof die onafhankelijk zijn. Bij ρ̄ = 0,7442 zijn ze dat niet. | Standaardfouten geclusterd op datum, plus een N_eff-deflatie die de gepoolde t met factor **0,460** verkleint. De richtingsclaim wordt daarmee nóg zwakker. | 4A, 6 |
| Q5 | De DSR wordt aangeroepen als `deflated_sharpe(sr, M, N)`. De DSR heeft ook `V[{SR_m}]`, scheefheid en kurtosis nodig; die worden stilzwijgend op de normale benadering gezet. | Expliciete handtekening met `sr_variance`, `skew`, `kurtosis`; de benadering wordt een geregistreerde keuze in plaats van een verborgen aanname. | 4A |
| Q6 | Geen afgesloten holdout. Na een ledger-reset is `M` niet controleerbaar door de reset zelf. | De laatste 12 maanden worden bevroren als poortsample, exact één keer aanraakbaar per hypothese die de ontwikkelsample overleeft. | 4B |
| Q7 | De bezettingspoort telt bars. Toestanden zijn persistent; het effectieve aantal observaties is het aantal **episodes**. | De poort telt episodes én bars, en oordeelt op het minimum van beide over folds. | 9 |
| Q8 | REGEL V beweert dat elke vermenigvuldiging redundant is. Dat geldt alleen voor een factor die op `t` voor **alle** namen gelijk is. | REGEL V is gepreciseerd: neutraliteit geldt voor cross-sectioneel uniforme factoren. De poort blijft de gekozen afbeelding, nu met de juiste grond. | §4 |
| Q9 | `gate_by_state` zet HOOG op nul; L7 levert de rest omhoog om het vol-target te halen. De concentratie die daaruit volgt, wordt niet gemeten en niet begrensd. | Verplichte concentratiemeting en een harde maximumweging in `constraint_order`. | 7, 17 |
| Q10 | Drie definities van het meetvenster staan naast elkaar: N = 1615 (4,42 j), T = 4,78 j (1745 bars) en 1743 bars in de halttabel. De t = 2-drempel van 0,915 hoort bij 4,78 j; bij N = 1615 is zij **0,951**. | Eén venstendefinitie, afgedwongen door een contract dat elke Sharpe samen met `(n_obs, bars_per_year, t_years)` rapporteert. Reconciliatie is stap 1.6 en blokkeert de fase. | 1 |
| Q11 | Vier permanent falende tests als "gedragsvingerafdruk". Rood is daarmee de normale toestand van CI, en een echte regressie valt niet op. | De vier killgates worden `xfail(strict=True)`. De suite is groen; een killgate die gaat slagen, maakt haar rood. | 1 |

Verder toegevoegd, omdat een quant-programma er niet zonder kan en revisie 1 er niets over zei: de rendementsconventie (§3.6), een verplichte marktbeta-attributie vóór elke alfaclaim (§3.7), een breakeven-kostenanalyse in plaats van één kostenpunt (stap 11), en de vaststelling dat de familie-correctie via `M` loopt en **niet** daarnaast nog een FDR-procedure krijgt (§3.5).

---

## 1. ROL EN CONTEXT

Je acteert als **Quant Research Engineer & Platform Architect**. Je erft een repository van **290 modules / 71.025 LOC in `src/`**, **145 testbestanden / 29.151 LOC**, tien DVC-stages en negen afgeronde fasen.

Het platform is gebouwd rond één principe: *een resultaat telt pas wanneer het de poort is gepasseerd die het had kunnen tegenhouden.* Negen fasen hebben nul modellen gepromoveerd en twintig hypothesen gefalsifieerd. Dat is geen mislukking maar de opbrengst: de repository weet nu wat er níet werkt, en dat weet zij met bewijs.

**De regel die deze fase regeert:**

> **Deze fase voegt geen voorspeller toe. Zij verandert de meetbasis, repareert één architectuurfout en zet de statistische meetkern recht — en zij doet alle drie zo dat een negatieve uitkomst een geldig, goedkoop antwoord is.**

**Relevante lagen (Target Architecture §19):** L2 (volatiliteit), L3 (features), L4 (alpha), L7 (risk), L10 (backtest), L11 (validatie), L12 (governance).

---

## 2. HET MEETDOMEIN

Dit hoofdstuk gaat vóór de drie mandaatbesluiten, want het definieert waarover die besluiten gaan.

### 2.1 De definitie

> **Het meetdomein van dit programma bestaat uit precies drie gecertificeerde bronnen in de PIT-store, geobserveerd op precies één frequentie:**
>
> | Bron | Frequentie | Certificering |
> |---|---|---|
> | OHLCV per symbool | **1 bar per dag** | `data/pit_store/`, hash-gecertificeerd |
> | Funding rate | 8-uurs, geaggregeerd naar dagbar | idem, via `features/microstructure.py` |
> | Open interest | 1 observatie per dag | idem |
>
> Elke grootheid die het programma gebruikt, is een functie van deze drie. Een grootheid die een fijnere waarneming vereist dan één bar per dag, is **geen uitgestelde vraag maar een niet-bestaande vraag**: er is geen bron voor, er komt geen bron voor, en er is geen conditie waaronder zij terugkeert.

### 2.2 Waarom dit een domein is en niet een verbodslijst

Een verbodslijst is nooit volledig. Zij noemt de vormen die iemand al had bedacht, en zij nodigt uit om een variant te bedenken die er niet op staat. Een domein is per constructie volledig: wat er niet in zit, zit er niet in, ongeacht hoe het heet.

Dat heeft één praktische consequentie die stap 2 uitwerkt: de mandaatpoort **whitelist** de drie bronnen. Een heropeningsconditie in `FALSIFICATION_REGISTER.md` is geldig wanneer zij een bron uit §2.1 noemt, en ongeldig wanneer zij dat niet doet. De poort hoeft niet te weten wat er buiten het domein bestaat.

### 2.3 Wat er binnen het domein wél mag

Om de veelgemaakte verwarring voor te zijn:

* **Range-gebaseerde variantieschatters mogen.** `parkinson`, `garman_klass`, `rogers_satchell` en `squared_return` in `volatility/realized.py` zijn functies van dagelijkse OHLC. Zij blijven, ongewijzigd.
* **Funding en open interest mogen.** `features/microstructure.py` (474 LOC) bevat uitsluitend `FundingRateMean`, `FundingRateZScore` en `OpenInterestLogChange`. Alle vijf features die de authoritative 15-feature-registry als "microstructuur" labelt, komen hiervandaan en zijn dagelijkse grootheden. **De naam van dat bestand is misleidend; de inhoud is domeinconform.** Stap 15.3 hernoemt het bestand naar `features/positioning.py` zodat de naam de inhoud niet meer tegenspreekt.
* **GARCH, EGARCH, GJR en EWMA mogen.** Het zijn dagelijkse modellen op dagelijkse rendementen.

Wat er níet mag, is één ding: een waarneming binnen de dag. Daaraan hangt de rest van deze fase.

---

## 3. MEETSTANDAARDEN

Bindend voor **elke** grootheid die deze fase rapporteert. Eén implementatie per standaard; een tweede implementatie is een defect, ook wanneer zij hetzelfde uitrekent.

### 3.1 Annualisatie en venster

`bars_per_year = 365` (de markt is continu). Elke gerapporteerde Sharpe draagt onlosmakelijk het drietal `(n_obs, bars_per_year, t_years)` met `t_years = n_obs / bars_per_year`. Een Sharpe zonder dat drietal is geen getal maar een gerucht, en `metrics.py` weigert hem te serialiseren.

### 3.2 De standaardfout van een Sharpe

Niet `1/√T`. Die uitdrukking veronderstelt i.i.d. normale rendementen; dagelijkse crypto-P&L is scheef, dik-staartig en autogecorreleerd, en de drie effecten werken niet dezelfde kant op.

Verplicht is de Lo (2002)-vorm met een HAC-correctie:

\[
\mathrm{SE}(\widehat{SR}) = \sqrt{\frac{1 + \tfrac{1}{2}\widehat{SR}^2 - \gamma_3 \widehat{SR} + \tfrac{\gamma_4 - 3}{4}\widehat{SR}^2}{T}} \cdot \sqrt{\eta_q}
\]

met \(\gamma_3\) de scheefheid, \(\gamma_4\) de kurtosis, en \(\eta_q\) de Newey–West-opslag over \(q = \lfloor 4(T/100)^{2/9}\rfloor\) lags — voor `T = 1615` is dat **q = 7**. De naïeve variant mag als referentiekolom worden meegerapporteerd, nooit als de toets.

### 3.3 Het verschil tussen twee Sharpes

Een Sharpe-verschil is geen gemiddeld rendementsverschil. De toets is **Ledoit–Wolf (2008)**, "Robust performance hypothesis testing with the Sharpe ratio": de HAC-geschatte covariantie van de vier momenten, gestudentiseerd met een circulaire blokbootstrap (Politis–Romano, `n_boot = 10.000`, bloklengte automatisch gekalibreerd).

Dit geldt voor **elke** vergelijking in deze fase: laagovergangen (stap 10), EWMA tegen GARCH als dat op de toestandsas nodig blijkt (stap 8), geconditioneerd tegen ongeconditioneerd (stap 13), en de panelen in stap 12.

### 3.4 Paneelafhankelijkheid

Gemeten: ρ̄ = 0,7442 over zes namen, N_eff = 1,271. Een gepoolde toets over symbool-bars behandelt 10.400 observaties als onafhankelijk, terwijl er effectief ongeveer 1.733 dagen × 1,271 namen aan informatie in zit.

Twee verplichtingen:

1. **Cluster op datum.** Elke gepoolde standaardfout wordt geclusterd op de tijdsindex, niet op de symbool-bar.
2. **Rapporteer de deflatie.** Een gepoolde t wordt naast zijn N_eff-gedefleerde tegenhanger gezet: \(t_{\text{eff}} \approx t_{\text{pooled}} \cdot \sqrt{N_{\text{eff}}/N} = t_{\text{pooled}} \cdot 0{,}460\).

Wat dat met de nulmeting doet, staat in §5.3, en het is de belangrijkste getalscorrectie van deze revisie.

### 3.5 Multiple testing

De familie-correctie loopt **uitsluitend** via `M` in de deflated Sharpe ratio. `M` is het aantal trials in de familie; de DSR is daarmee al een familiewijze correctie over precies die familie.

Er komt dus **geen** tweede laag (geen Benjamini–Hochberg, geen Bonferroni) bovenop de DSR. Twee correcties over dezelfde familie is niet conservatiever maar onbepaald: de gecombineerde grootte is niet meer te herleiden. Wie een extra correctie wil, verhoogt `M`.

### 3.6 De rendementsconventie

Revisie 1 liet dit open, en het is niet optioneel.

* Het instrument is de **perpetual future**, niet spot.
* De P&L is `prijsrendement + funding`, waarbij funding met het teken van de positie meeloopt en op de dagbar wordt geaccumuleerd.
* Kosten: `fee + spread/2 + impact`, met de impactparameters uit `execution/impact_model.py` (`eta = 2,991922`) en de labels AD-2 (`IMPACT_UNCALIBRATED`) en AD-3 (`SPREAD_ASSUMED`), die per stap 17 **permanent** worden.
* F9 falsificeert shorts op dit universum onvoorwaardelijk. Long-only is daarmee geen keuze maar een gegeven, en de fundingcomponent heeft dus één teken.

Een backtest die funding niet expliciet boekt, is op perpetuals geen backtest. `backtest/accounting.py` draagt de dubbele boekhouding al; stap 1.7 verifieert dat de fundingregel erin staat en dat de twee boeken sluiten.

### 3.7 Geen alfaclaim zonder residuele t

Zes namen met ρ̄ = 0,7442 vormen geen cross-sectie maar een gerichte weddenschap op één factor. Daarom:

> **Geen enkele uitkomst in deze fase mag "alpha" worden genoemd voordat het rendement is geregresseerd op (a) BTC-rendement en (b) het gelijkgewogen universumrendement, en de residuele alfa met haar HAC-t is gerapporteerd.**

De bruto Sharpe is een beschrijving. De residuele alfa is de claim. Het verschil hoort in elk rapport te staan, ook — en juist — wanneer het de conclusie omdraait.

### 3.8 Purge en embargo

Elke walk-forward of CV-structuur in deze fase draagt een purge gelijk aan de maximale featurelookback en een embargo van minimaal 1 % van de sample (≥ 17 bars bij N = 1615). Dit is bestaand beleid in `validation/`; het wordt hier herhaald omdat de nieuwe toestandsmodules er ook onder vallen en dat nergens stond.

### 3.9 De DSR, expliciet

`backtest/metrics.py::deflated_sharpe` krijgt de handtekening die de statistiek vereist:

```
deflated_sharpe(
    sr_hat, *, n_obs, n_trials, sr_variance, skew, kurtosis, bars_per_year
) -> DSRResult
```

met `SR₀ = √(sr_variance) · [ (1−γ)·Φ⁻¹(1−1/M) + γ·Φ⁻¹(1−1/(M·e)) ]`, γ de Euler–Mascheroni-constante, en

\[
\mathrm{DSR} = \Phi\!\left( \frac{(\widehat{SR} - SR_0)\sqrt{T-1}}{\sqrt{1 - \gamma_3 \widehat{SR} + \frac{\gamma_4-1}{4}\widehat{SR}^2}} \right)
\]

`sr_variance` is de **empirische** variantie van de trial-Sharpes wanneer die beschikbaar zijn, en anders de gedocumenteerde benadering `1/n_obs` — met dat feit in het artefact, niet in iemands hoofd. Zie stap 4A.

---

## 4. DE CORRECTIE DIE DEZE FASE DRAAGT

Dit hoofdstuk herschrijft wat de vorige fasen met volatiliteitsmodellen deden.

### 4.1 Wat er nu gebeurt

Volatiliteitsmodellen worden in deze repository op twee manieren gebruikt, en **beide zijn de verkeerde**:

1. **Als puntvoorspeller van σ², beoordeeld op forecast-nauwkeurigheid.** Dat is de H1-campagne: 48 gefitte combinaties, beoordeeld met QLIKE tegen een variantieproxy. Die opzet vraagt een proxy, en de proxy is precies waar H1 op is vastgelopen (`mean(proxy)/mean(r²)` = 1,27–2,24).
2. **Als schaalvermenigvuldiger op de exposure.** Dat is AD-15: `a_geconditioneerd[t] = a_basis[t] × (1 − p_hoog[t])`, waarbij `p_hoog` uit M0, M1 of M2 komt.

`regime/conditioning.py::_high_state` kiest de toestand op **variantie** en gooit de gefitte `means` weg — `regime/markov.py` schat ze wel (`means: np.ndarray  # (k, d)`, regel 161), maar niets leest ze.

### 4.2 Waarom (2) niet kán werken — dit is gemeten, niet beredeneerd

AD-16 heeft het al vastgesteld en het is de scherpste meting in het hele regimespoor:

> De soevereine laag schaalt het boek met `w_t = min(max_leverage, σ_target / σ_boek)`. Vermenigvuldig elke exposure met dezelfde `c`, dan deelt `w_t` er weer door. **Gemeten:** elke exposure halveren verplaatste de gemiddelde bruto notional van 8.300 naar 8.283.

### 4.3 REGEL V, gepreciseerd

Revisie 1 formuleerde hier een regel die te ruim was. De meting van AD-16 gaat over een factor die op moment `t` voor **alle** namen dezelfde is; daaruit volgt niet dat elke vermenigvuldiging redundant is. De juiste formulering:

> **REGEL V.** Laat `a_t` de exposurevector zijn en `c_t(i)` een toestandsafhankelijke factor.
>
> * Is `c_t(i) = c_t` voor alle `i` — cross-sectioneel uniform — dan is de afbeelding **per constructie een lege operatie**: L7 herschaalt op σ̂ en deelt `c_t` er weer uit. Dit is het geval van AD-15 en het is gemeten in AD-16.
> * Varieert `c_t(i)` over `i`, dan raakt de afbeelding de **samenstelling** en is zij niet leeg. Zij kost dan wel `k − 1` vrije parameters per toestand, en elke daarvan is een trial.
> * De enige cross-sectioneel gedifferentieerde afbeelding **zonder** vrije parameter is de **poort**: `c_t(i) ∈ {0, 1}`, met de toestandsverzameling die op nul gaat vooraf geregistreerd.
>
> **Bindend gevolg:** een toestand mag uitsluitend via een poort op de exposure worden afgebeeld, en de poort moet aantoonbaar vol-targeting overleven. Een multiplier per toestand is toegestaan noch verboden — hij is simpelweg te duur, en wie hem wil, betaalt hem in trials en registreert dat vooraf.

Dit is scherper dan revisie 1 én het rechtvaardigt dezelfde keuze. De poort in stap 7 blijft, nu met een grond die klopt.

### 4.4 Wat een volatiliteitsmodel wél hoort te doen

```
vol-model  →  σ̂_t  →  toestandstoewijzing  →  s_t ∈ {0, 1, 2}  →  BESLUIT
```

Drie gevolgen die deze fase uitwerkt:

* **De evaluatie verandert.** Een toestandstoewijzer wordt niet met QLIKE beoordeeld maar op vier meetbare eigenschappen: *separatie* (verschillen de toestanden in de grootheid die er toe doet?), *bezetting* (heeft elke toestand genoeg bars **én genoeg episodes**?), *persistentie* (is de toestand stabiel genoeg om op te handelen?) en *economische waarde* (verandert het besluit, en wordt de netto uitkomst beter?). **Geen daarvan vraagt een variantieproxy.** De blokkade die H1 heeft geveld, bestaat op dit spoor niet.
* **De vergelijking tussen vol-modellen wordt goedkoop.** Niet "voorspelt GARCH σ² beter dan EWMA" — op de dagbar niet identificeerbaar (§5.4) — maar "wijst GARCH een ándere toestand toe dan EWMA, en verandert dat het besluit?" Dat is een telling, geen toets. Zie stap 8.
* **De afbeelding moet REGEL V respecteren.** Zie stap 7.

### 4.5 En wat de toestand hier wél en niet blijkt te dragen

Gemeten met `regime/buckets.py::classify_vol_buckets` op de gecertificeerde PIT-store, drempels uit `conf/model/regime.yaml`, 10.400 symbool-bars:

| Toestand | n | aandeel | ann. rendement | **ann. volatiliteit** | t (gepoold) | **t (N_eff-gedefleerd)** |
|---|---:|---:|---:|---:|---:|---:|
| LAAG | 1.720 | 16,5 % | +49,6 % | **62,6 %** | 1,72 | **0,79** |
| NORMAAL | 8.181 | 78,7 % | +9,5 % | **79,2 %** | 0,57 | **0,26** |
| HOOG | 499 | 4,8 % | +136,4 % | **137,0 %** | 1,16 | **0,53** |

Lees de tabel op twee manieren, want zij zegt twee verschillende dingen.

**De toestand scheidt volatiliteit uitstekend.** 62,6 % tegen 137,0 % is een factor 2,2, en dat is wat een volatiliteitsmodel hoort te kunnen. Hier is niets mis mee.

**De toestand scheidt richting niet — en veel minder dan revisie 1 suggereerde.** De gepoolde t-statistieken lagen al onder 2. Na clustering op datum en N_eff-deflatie (§3.4) blijft er **geen enkele t boven 0,8** over. De tekens spreken elkaar bovendien per symbool tegen: DOTUSDT doet +131,6 bp/dag in HOOG, SOLUSDT −45,1 bp. Een `0 = bearish / 2 = bullish`-lezing wordt door dit paneel op geen enkele manier gedragen.

De toestand wordt daarom geïdentificeerd op **variantie** — de grootheid die betrouwbaar wordt geschat — en haar semantiek wordt **gemeten en gerapporteerd**, nooit aangenomen.

**En één bevinding die de huidige bedrading rechtstreeks raakt:** AD-15 de-grost op `p_hoog`, dus het verlaagt de exposure het sterkst in de toestand met het **hoogste** gemeten gemiddelde rendement. Dat is niet significant — het is nu zelfs uitgesproken insignificant — maar het is ook niet het teken dat de afbeelding veronderstelt. De afbeelding codeert een aanname over richting die niemand heeft gemeten.

---

## 5. NULMETING (gemeten 2026-09-04 tegen `5ae4a00` — reproduceer vóór je iets wijzigt)

Deze getallen zijn de basis waartegen deze fase wordt afgerekend. Wijkt jouw meting af, dan is dát je eerste bevinding en die schrijf je op vóór je verdergaat.

### 5.1 De vier muren

| Grootheid | Gemeten | Bron |
|---|---:|---|
| Effectieve breedte N_eff, 6 namen | **1,271** (ρ̄ = 0,7442) | eigen meting op `data/pit_store/`, formule `alpha/cm_carry.py:291` |
| N_eff per jaar | 1,40 · 1,20 · 1,41 · 1,34 · **1,19** · **1,17** | idem, 2021–2026 |
| Ann. Sharpe nodig voor t = 2 **bij N = 1615 (4,42 j)** | **0,951** | `t = SR·√T`; zie Q10 |
| Ann. Sharpe nodig voor DSR ≥ 0,95 bij M = 2776, N = 1615 | **2,47** | `backtest/metrics.py::deflated_sharpe` |
| Beste gemeten track (L0, `long_only_equal_weight`) | **0,156** | `artefacts/baseline/phase3_baseline.json` |

> **Let op de correctie in rij 3.** Revisie 1 noteerde 0,915. Dat getal hoort bij 4,78 jaar (1745 bars); bij het venster dat de DSR-rij gebruikt (N = 1615, 4,42 j) is de drempel **0,951**. Twee vensters in één tabel is precies het soort slordigheid dat een marginaal resultaat significant laat lijken. Stap 1.6 lost dit op vóór er iets wordt gemeten.

### 5.2 De vier-lagen-ladder (netto Sharpe)

| Track | L0 vectorized | L1 + risk | L2 + latency | L3 + execution |
|---|---:|---:|---:|---:|
| `xs_momentum_equal_weight` | +0,098 | +0,215 | −0,052 | −0,103 |
| `xs_momentum_risk_parity` | −0,227 | −0,005 | −0,263 | −0,331 |
| `long_only_equal_weight` | +0,039 | +0,119 | +0,136 | −0,695 |
| `long_only_risk_parity` | +0,012 | +0,120 | +0,130 | −0,721 |

`long_only_equal_weight` halteert op 2022-05-10 bij 8,31 % drawdown: **1.559 van 1.743 bars gehalteerd**. Bron: `artefacts/baseline/phase5_revaluation.json`.

> **Wat die halt met de ladder doet, en wat revisie 1 niet zag.** Als 1.559 van 1.743 bars gehalteerd zijn, is de L1-en-hoger-Sharpe berekend over ongeveer **184 actieve bars** — 0,50 jaar. De t = 2-drempel op 0,50 jaar is een annualiseerde Sharpe van **2,83**. Een gerapporteerde +0,136 op dat venster is niet zwak bewijs; het is geen bewijs.
>
> Dit heeft een harde consequentie voor stap 10: een gepaarde vergelijking tussen L1 en L2 mag **uitsluitend** over bars lopen waarop beide ketens actief zijn, en het aantal van die bars hoort in de tabel. Een gepaard verschil over verschillende actieve verzamelingen is geen gepaard verschil.

### 5.3 De vol-toestand

Zie §4.5, inclusief de N_eff-gedefleerde kolom. Reproduceerbaar met `regime/buckets.py` en `conf/model/regime.yaml`.

### 5.4 De H1-eindstand

Uit `reports/GARCH_VS_EWMA_COMPETITION.md` §9:

| Symbool | h | Variant | p (rogers_satchell) | p (garman_klass) | p (parkinson) | **p (squared_return)** |
|---|---|---|---:|---:|---:|---:|
| AVAXUSDT | 1 | `garch(1,1)-t` | 0,0076 | 0,0047 | 0,0104 | **0,517** |
| AVAXUSDT | 1 | `gjr_garch(1,1,1)-t` | 0,0075 | 0,0070 | 0,0202 | **0,752** |
| AVAXUSDT | 1 | `egarch(1,1,1)-t` | 0,0106 | 0,0079 | 0,0200 | **0,514** |
| AVAXUSDT | 5 | `garch(1,1)-t` | 0,0121 | 0,0116 | 0,0264 | **0,368** |

Op de drie range-proxies lijkt het verschil significant; op `squared_return` — de enige proxy die per constructie zuiver is voor de voorspelde grootheid `E[r_t²|F_{t−1}] = σ_t²` — verdampt het volledig. De range-proxies zijn efficiënter maar niet zuiver; `squared_return` is zuiver maar zeer ruisig. Binnen het meetdomein van §2.1 bestaat geen schatter die beide is. **Het verschil is op de dagbar niet identificeerbaar, en dat is een eigenschap van de meetbasis, niet van de modellen.**

H1 is daarmee **gesloten, niet geblokkeerd**, en EWMA(0,94) is de productie-estimator bij besluit. Dit is expliciet geen bewering dat GARCH slechter is. De vraag "wijst GARCH een andere *toestand* toe" is een andere vraag en staat open — zie stap 8.

### 5.5 De gedragsvingerafdruk

`python -m pytest -q` → **exact 4 failures**, alle vier pre-geregistreerde killgates op `cm_carry` en `cm_tsmom`.

> **Q11.** Deze conventie normaliseert rood. Stap 1.8 zet de vier killgates om naar `@pytest.mark.xfail(strict=True, reason=...)` met een verwijzing naar hun registerregel. Daarna is de vingerafdruk: **0 failed, 4 xfailed, 0 xpassed**. Een killgate die begint te slagen, maakt de suite rood — wat de bedoeling was. Het aantal blijft vier; alleen de kleur van "normaal" verandert.

### 5.6 De DSR-drempel als functie van M (N = 1615, dagbars)

| M | 2776 | 500 | 250 | 100 | 50 | **25** | 12 | 6 | 4 | 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vereiste ann. Sharpe | 2,47 | 2,24 | 2,14 | 1,99 | 1,87 | **1,74** | 1,58 | 1,40 | 1,28 | 1,03 |

### 5.7 Wat een zoektocht uit RUIS oplevert (SE ann. Sharpe = 1/√4,42 = 0,476)

| M cellen | 2 | 4 | 6 | 12 | 25 | 55 | 100 | 2776 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E[max Sharpe], SE 0,476 | 0,25 | 0,50 | 0,62 | 0,79 | 0,95 | **1,10** | 1,20 | 1,68 |
| E[max Sharpe], SE 0,458 (rev. 1) | 0,24 | 0,48 | 0,60 | 0,76 | 0,91 | 1,06 | 1,16 | 1,62 |

**Dit is tijdens de brainstorm empirisch bevestigd.** Een diagnostisch grid van 11 formatievensters × 5 houdlagen = **55 cellen** op de authoritative unit leverde een hoogste bruto Sharpe van **+1,075** op, tegen een verwacht ruismaximum van **+1,10** op het gecorrigeerde venster. De bijbehorende IC's lagen tussen −0,024 en +0,050 bij een standaardfout van 0,011. **Er zat niets in, en toch produceerde een halve middag verkennen een getal boven de DSR-eis bij M = 2.**

En let op de tweede rij: het gecorrigeerde ruismaximum ligt **hoger** dan het gemeten maximum. Het diagnostische grid presteerde onder ruis.

---

## 6. DE DRIE MANDAATBESLUITEN

Genomen door de eigenaar. Deze fase voert ze uit en toetst ze niet.

| | Besluit | Wat het sluit |
|---|---|---|
| **B-1** | **Barresolutie: dagbars.** De handelsklok volgt de meetklok. `live/feed.py::FeedConfig.bar_seconds` vervalt. | Het resolutieverschil tussen de twee ketens; het verschil 5 vs. 6 namen; exit-criterium D7 wordt voor het eerst *gedefinieerd* |
| **B-2** | **Één meetdomein: de dagbar.** Het domein is §2.1 — drie bronnen, één frequentie. Wat een fijnere waarneming vereist, valt buiten het domein en is geen uitgestelde vraag. | H1's heropeningsconditie, DI-18, `bars/`, F2 en F19 |
| **B-3** | **De ledger wordt gereset.** `M = 2776` is niet langer de trial-teller voor dit programma. | De rekenkundige onbereikbaarheid van de promotiepoort |

---

## 7. FENCES — WAT DEZE FASE NIET AANRAAKT

1. **De vijf promotiepoorten.** `validation/gates.py` krijgt geen `force=`, geen `override=`, geen `warn_only=`, geen deelscore. Eén invoerwaarde (`M`) verandert; geen poort.
2. **`FALSIFICATION_REGISTER.md` F1 t/m F20.** Alleen een domeinmarkering mag worden toegevoegd. Geen regel wordt gewijzigd, verzwakt of verwijderd.
3. **De lookahead-suite.** 628 tests, `tests/lookahead/`. Onaangeraakt; er komt uitsluitend bij.
4. **De dubbele boekhouding en de engine.** `backtest/accounting.py`, `backtest/engine.py`. Onaangeraakt behalve de toevoeging in stap 10 en de verificatie in stap 1.7.
5. **`alpha/cm_carry.py` en `alpha/cm_tsmom.py`.** Dragen pre-geregistreerde killgates.
6. **De haltketen.** `risk/kill_switches.py`, `live/circuit_breaker.py`, `monitoring/sharpe_monitor.py`, `monitoring/execution_drift.py`. Resolutie-onafhankelijk en bewezen; blijft ook wanneer stap 16 `live/` vervangt.
7. **De PIT-store en zijn hashes.** Read-only.
8. **`volatility/realized.py` en de dagelijkse funding/OI-features.** Domeinconform per §2.3; blijven ongewijzigd.

---

## 8. CONCRETE DELIVERABLES

| # | Deliverable | Stap |
|---|---|---|
| D1 | AD-22, AD-23, AD-24 in `docs/ARCHITECTURAL_DECISIONS.md` | 1 |
| D2 | `docs/MEASUREMENT_CONTRACT.md` — één venster, één annualisatie, één SE | 1 |
| D3 | Vier killgates op `xfail(strict=True)`; suite groen | 1 |
| D4 | `scripts/check_domain_consistency.py` + CI-poort (whitelist) | 2 |
| D5 | `artefacts/governance/ledger_reset.json` + bevroren `M_new` | 3 |
| D6 | `validation/inference.py` — Lo-SE, HAC, Ledoit–Wolf, blokbootstrap, datumclustering | 4A |
| D7 | `backtest/metrics.py::deflated_sharpe` met expliciete `sr_variance/skew/kurtosis` | 4A |
| D8 | `artefacts/governance/holdout_lock.json` — bevroren poortsample | 4B |
| D9 | `registry/trial_budget.py` + poort op overschrijding | 4 |
| D10 | `regime/state.py` — het `VolState`-contract, causaal met expliciete lag | 5 |
| D11 | `reports/phase10_state_diagnostics.md` + artefact (separatie, bezetting, persistentie) | 6 |
| D12 | `regime/state_mapping.py` — de poort die REGEL V respecteert | 7 |
| D13 | `reports/phase10_state_agreement.md` — EWMA vs. GARCH, proxyvrij | 8 |
| D14 | Bezettings- én episodepoort in `validation/data_adequacy.py` | 9 |
| D15 | Gepaarde Ledoit–Wolf-toetsen per laagovergang in `backtest/phase5_baseline.py` | 10 |
| D16 | Pre-registratie + uitkomst H-10.1 (beslisfrequentie, met breakeven-kosten) | 11 |
| D17 | Pre-registratie + uitkomst H-10.2 (ongebalanceerd paneel) | 12 |
| D18 | Pre-registratie + uitkomst H-10.3 (toestandspoort, met concentratiemeting) | 13 |
| D19 | `reports/phase10_chain_a_score.md` — keten A gescoord, niet-promoveerbaar | 14 |
| D20 | `reports/phase10_domain_surface.md` — verdict per module buiten het domein | 15 |
| D21 | Besluit + uitvoering `live/` (vervangen of aansluiten) | 16 |
| D22 | Harde omvangs- **én concentratielimiet** in `conf/risk/default.yaml` | 17 |
| D23 | `reports/phase10_exit_report.md` | slot |

---

## 9. GLOBALE RANDVOORWAARDEN

Deze gelden impliciet bij **elke** stap hieronder.

* **Python** uit `docs/RUNBOOK.md` par. 0 (`D:\venv\tradebot\Scripts\python.exe`). Meten op een andere interpreter is in Phase 9 zeven keer een foutmeting gebleken.
* **`ruff` is gepind op 0.15.12** (`requirements-dev.lock`, DI-16). Niet upgraden in deze fase.
* **R-1** Causaliteit is absoluut: een feature of toestand voor bar `t` gebruikt uitsluitend data ≤ `t−1`. **Dit geldt ook voor kwantielen, drempels en toestandsgrenzen** — zie Q1.
* **R-4** 800 LOC per bestand, afgedwongen door `scripts/check_file_size.py` met een cap per bestand.
* **R-5** Determinisme: gelijke `cfg + seed` ⇒ bit-identieke output. Ook de blokbootstrap draait op een geregistreerde seed.
* **R-6** Apps zijn stateless en ≤ 80 LOC; alle logica in `src/tradebot/`.
* **Elke drempel komt uit `conf/`**, nooit als literal in een handtekening. Bewaakt door `scripts/check_hardcoded_params.py`.
* **Elke nieuwe variant is een trial.** Ook een verkenning, ook een tak van een beslisboom. Zie stap 4.
* **Elke Sharpe draagt `(n_obs, bars_per_year, t_years)`** en een SE volgens §3.2. Zonder dat drietal weigert de serializer.
* **Na elke stap:** `python -m pytest -q` toont **0 failed, 4 xfailed, 0 xpassed** (na stap 1.8; daarvoor 4 failed).

---

## STAPSGEWIJZE UITVOERING

# STAGE A — MANDAAT, MEETKERN EN LEDGER

*Blokkeert alles. Zonder A is elk resultaat uit B en C ongeldig — niet omdat het verkeerd is, maar omdat het niet interpreteerbaar is.*

---

### Stap 1: Mandaat vastleggen en de meetbasis reconciliëren

**Files:**
- Modify: `docs/ARCHITECTURAL_DECISIONS.md` (append AD-22, AD-23, AD-24)
- Modify: `docs/PROJECT_STATE.md` §5 (besluiten 1–3 verwijderen, verwijzen naar AD-22/23/24)
- Create: `docs/MEASUREMENT_CONTRACT.md`
- Modify: `tests/unit/test_cm_carry_killgate.py`, `tests/unit/test_cm_tsmom_killgate.py` (of de bestanden die de vier killgates dragen)

**Interfaces:**
- Produces: de identifiers `AD-22`, `AD-23`, `AD-24`, waarnaar stap 2, 14, 15 en 16 verwijzen.

- [ ] **Stap 1.1 — Reproduceer de nulmeting.**

```bash
D:/venv/tradebot/Scripts/python.exe -m pytest -q
D:/venv/tradebot/Scripts/python.exe apps/run_phase5_baseline.py
```

Verwacht: exact 4 failures; `artefacts/baseline/phase5_revaluation.json` reproduceert de laddertabel uit §5.2. Wijkt het af, **stop en rapporteer**.

- [ ] **Stap 1.2 — Schrijf AD-22 (barresolutie).** Volg de structuur van AD-15: `Fase / Status / Bewaakt door`, dan `Besluit`, `Waarom`, `Het afgewezen alternatief`. Het besluit: dagbars als meet- én handelsresolutie; `bar_seconds` vervalt. Het afgewezen alternatief: twee resoluties naast elkaar houden en een adapter bouwen — afgewezen omdat een pariteitstest over twee resoluties de configuratie meet in plaats van het gedrag.

- [ ] **Stap 1.3 — Schrijf AD-23 (het meetdomein).** Neem §2.1 **integraal** op: de drie bronnen, de ene frequentie, en de formulering dat wat buiten het domein valt geen uitgestelde vraag is. Neem ook §2.3 op — wat er binnen het domein wél mag — want dat is de helft die mensen vergeten en waardoor er later domeinconforme modules worden gesloopt.

  Formuleer het besluit als een **whitelist**, niet als een opsomming van wat er niet mag. De zin die erin hoort:

  > Een grootheid is toelaatbaar dan en slechts dan wanneer zij een meetbare functie is van de drie bronnen in de tabel, geobserveerd op één bar per dag. Toelaatbaarheid wordt bewezen door de bron te noemen, niet door de afwezigheid van een verbod.

- [ ] **Stap 1.4 — Schrijf AD-24 (ledger-reset).** Neem de zeven regels R1–R7 uit stap 3 integraal op. Vermeld de gemeten grond: `seed_total = 2363` is een reconstructie uit een verloren logboek en volgens `registry/trial_counter.py` een ondergrens. Vermeld óók, want R2 hangt eraan, dat de reset geen enkele regel uit F1–F20 vrijgeeft.

- [ ] **Stap 1.5 — Schrijf `docs/MEASUREMENT_CONTRACT.md`.** Dit document is nieuw en het is de belangrijkste tekst van Stage A. Het bevat, en niets anders:

  1. `bars_per_year = 365`, met de reden.
  2. De **ene** definitie van het meetvenster: `period_start`, `period_end`, `n_obs`, `t_years`, en de exacte manier waarop symbolen met kortere historie worden behandeld (verwijs naar stap 12).
  3. De SE-formule uit §3.2, met de Newey–West-laglengte als functie van `T`.
  4. De verschiltoets uit §3.3.
  5. De clusterregel en de N_eff-deflatie uit §3.4.
  6. De DSR-handtekening uit §3.9.
  7. De rendementsconventie uit §3.6.
  8. De regel uit §3.7: geen alfaclaim zonder residuele t.
  9. De purge/embargo-regel uit §3.8.

  Elke stap in deze fase verwijst naar dit document in plaats van de conventie te herhalen. Eén plek, één waarheid.

- [ ] **Stap 1.6 — Reconcilieer de drie vensters (Q10). Dit blokkeert de fase.** Er staan drie waarden in de repository: `N = 1615` (4,42 j), `T = 4,78 j` (1745 bars) en `1743` bars in de halttabel. Stel per artefact vast welk venster erin zit:

```bash
D:/venv/tradebot/Scripts/python.exe -c "
import json, pathlib
for p in sorted(pathlib.Path('artefacts').rglob('*.json')):
    try: d = json.loads(p.read_text(encoding='utf-8'))
    except Exception: continue
    hits = {k: v for k, v in (d.items() if isinstance(d, dict) else [])
            if k in ('n_obs','n_bars','period_start','period_end','t_years','bars_per_year')}
    if hits: print(p, hits)
"
```

  Kies **één** venster, schrijf de keuze en de reden in `MEASUREMENT_CONTRACT.md` §2, en herbereken elke drempel die eruit volgt. De t = 2-drempel is `2/√t_years`; bij 4,42 jaar is dat **0,951** en niet 0,915. Herstel dat getal overal.

  > Waarom dit blokkeert: het verschil tussen 0,915 en 0,951 is 4 %, en dat is genoeg om een marginaal resultaat de verkeerde kant van een poort op te duwen. Een fase die met twee vensters begint, eindigt met een conclusie die van een boekhoudkeuze afhangt.

- [ ] **Stap 1.7 — Verifieer de fundingboeking (§3.6).** Stel vast dat `backtest/accounting.py` funding met het teken van de positie boekt en dat de dubbele boekhouding sluit. Schrijf een test die een positie over een fundingmoment heen houdt en aantoont dat de P&L de fundingcomponent bevat. Doet zij dat niet, **dan is dát de belangrijkste bevinding van deze fase** en gaat zij vóór alles wat hieronder staat: elke Sharpe in §5.2 is dan op een incomplete P&L gemeten.

- [ ] **Stap 1.8 — Zet de vier killgates op `xfail(strict=True)` (Q11).** Per killgate: de marker, een `reason` die de registerregel noemt (`F…`), en géén wijziging in de assertie zelf. Draai daarna:

```bash
D:/venv/tradebot/Scripts/python.exe -m pytest -q
```

Verwacht: **0 failed, 4 xfailed, 0 xpassed**. Werk de vingerafdruk bij in `docs/PROJECT_STATE.md` en in §5.5.

  > Wat dit oplevert: de suite is groen, dus een échte regressie valt onmiddellijk op. En `strict=True` maakt een killgate die begint te *slagen* rood — precies de gebeurtenis die je wilt zien, want die betekent dat het gefalsifieerde gedrag is teruggekeerd of dat de killgate is uitgehold.

- [ ] **Stap 1.9 — Commit.**

```bash
git add docs/ARCHITECTURAL_DECISIONS.md docs/PROJECT_STATE.md docs/MEASUREMENT_CONTRACT.md tests/unit/
git commit -m "docs(mandate): AD-22 dagbars, AD-23 meetdomein, AD-24 ledger-reset; meetcontract en xfail-killgates"
```

---

### Stap 2: De domeinpoort — een whitelist, geen verbodslijst

De registers dragen nu heropeningscondities die onder AD-23 nooit kunnen intreden. Een conditie die niet kan intreden en toch als "heropenbaar" te boek staat, nodigt uit tot een hertest die het mandaat verbiedt.

Revisie 1 loste dit op met een scanner die verboden termen zocht. Dat is een blacklist, en een blacklist heeft twee gebreken: zij is nooit volledig, en zij dwingt het document dat de fijnere waarneming afschaft om haar te blijven benoemen. Revisie 2 doet het omgekeerd.

**Files:**
- Create: `scripts/check_domain_consistency.py`
- Create: `conf/governance/measurement_domain.yaml`
- Create: `tests/unit/test_domain_consistency.py`
- Modify: `docs/FALSIFICATION_REGISTER.md` (H1-sectie; F2- en F19-regels)
- Modify: `docs/DEFERRED_ISSUES.md` (DI-18 naar Gesloten)
- Modify: `.github/workflows/inventory.yml`

**Interfaces:**
- Produces: `check_domain_consistency.main(argv) -> int` (0 = schoon, 1 = inconsistent), en `load_domain(path) -> Domain`.

- [ ] **Stap 2.1 — Leg het domein vast als configuratie.**

```yaml
# conf/governance/measurement_domain.yaml
# AD-23. Dit bestand IS het meetdomein. Een grootheid is toelaatbaar dan en
# slechts dan wanneer zij een functie is van een bron hieronder, geobserveerd
# op de frequentie hieronder. Toevoegen aan deze lijst is een mandaatbesluit
# en geen configuratiewijziging.
observation:
  bars_per_day: 1
  bar_resolution: "1d"

sources:
  - id: daily_ohlcv
    description: "Gecertificeerde dagelijkse open/high/low/close/volume per symbool"
    store: "data/pit_store"
  - id: funding_rate_8h
    description: "8-uurs funding rate, geaggregeerd naar de dagbar"
    store: "data/pit_store"
  - id: open_interest_daily
    description: "Open interest, één observatie per dag"
    store: "data/pit_store"

# Een heropeningsconditie in FALSIFICATION_REGISTER.md of DEFERRED_ISSUES.md is
# geldig wanneer zij ten minste één source-id hierboven noemt, of expliciet is
# gemarkeerd als buiten het domein.
out_of_domain_marker: "buiten het meetdomein (AD-23)"
```

- [ ] **Stap 2.2 — Schrijf de falende test.**

```python
# tests/unit/test_domain_consistency.py
"""AD-23 definieert een domein. Een register dat een heropening belooft zonder
een bron uit dat domein te noemen, belooft iets waarvoor geen data bestaat --
en dat is precies de tegenstrijdigheid waaruit later een verboden hertest
ontstaat.

De poort werkt met een WHITELIST. Dat is bewust: een blacklist is nooit
volledig, en zij dwingt dit project om de uitgesloten ruimte te blijven
benoemen in het document dat haar afschaft.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_domain_consistency.py"


def test_scanner_reports_clean_repository() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--strict"],
        capture_output=True, text=True, cwd=REPO,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_scanner_goes_red_on_a_reopening_without_a_domain_source(
    tmp_path: Path,
) -> None:
    """De negatieve controle. Zonder deze test toetst de poort haar eigen vorm."""
    register = tmp_path / "FALSIFICATION_REGISTER.md"
    register.write_text(
        "| F99 | verzonnen unit | bewijs | heropening zodra er een fijnere "
        "waarneming beschikbaar is |\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--strict", "--register", str(register)],
        capture_output=True, text=True, cwd=REPO,
    )
    assert result.returncode == 1
    assert "F99" in result.stdout


def test_a_reopening_that_names_a_domain_source_is_accepted(
    tmp_path: Path,
) -> None:
    """De poort mag geen legitieme heropening blokkeren; anders wordt zij
    genegeerd, en een genegeerde poort is geen poort."""
    register = tmp_path / "FALSIFICATION_REGISTER.md"
    register.write_text(
        "| F98 | verzonnen unit | bewijs | heropening bij meer historie in "
        "daily_ohlcv |\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--strict", "--register", str(register)],
        capture_output=True, text=True, cwd=REPO,
    )
    assert result.returncode == 0, result.stdout


def test_a_non_daily_resolution_in_conf_is_refused(tmp_path: Path) -> None:
    """B-1. Eén meetklok, en de poort werkt in BEIDE richtingen: elke declaratie
    die niet exact `1d` is, is een tweede meetklok in wording. Deze controle
    gebruikt daarom een grovere resolutie -- de poort hoeft niet te weten welke
    resoluties er buiten het domein bestaan, alleen welke erin zit."""
    conf = tmp_path / "feed.yaml"
    conf.write_text("bar_resolution: 1w\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--strict", "--conf-root", str(tmp_path)],
        capture_output=True, text=True, cwd=REPO,
    )
    assert result.returncode == 1
```

- [ ] **Stap 2.3 — Draai en zie hem falen.**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_domain_consistency.py -q`
Verwacht: FAIL — het script bestaat niet.

- [ ] **Stap 2.4 — Schrijf de scanner.**

```python
# scripts/check_domain_consistency.py
"""AD-23-poort: het meetdomein, afdwingbaar gemaakt.

TWEE CONTROLES
==============
1. RESOLUTIE. Geen enkel bestand in `conf/` declareert een barresolutie die
   niet exact `1d` is. Eén meetklok (B-1).
2. HEROPENING. Elke heropeningsconditie in de registers noemt ten minste één
   source-id uit `conf/governance/measurement_domain.yaml`, of staat expliciet
   gemarkeerd als buiten het domein.

WAAROM EEN WHITELIST
====================
Een blacklist van verboden informatiebronnen is nooit volledig -- zij noemt de
vormen die iemand al had bedacht -- en zij dwingt dit project om de uitgesloten
ruimte te blijven benoemen. Een whitelist is per constructie volledig: wat geen
bron uit het domein noemt, is niet toelaatbaar, ongeacht hoe het heet.

Rood worden is de bedoeling. `tests/unit/test_domain_consistency.py` bewijst dat
deze poort het kan, in beide richtingen.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DOMAIN = REPO / "conf" / "governance" / "measurement_domain.yaml"

ROW = re.compile(r"^\|\s*(F\d+|H\d+|DI-\d+)\s*\|")
RESOLUTION = re.compile(r"bar_(?:resolution|interval|timeframe)\s*:\s*([^\s#]+)")
REOPEN_HINT = re.compile(r"heropen", re.IGNORECASE)


@dataclass(frozen=True)
class Domain:
    resolution: str
    source_ids: tuple[str, ...]
    marker: str


def load_domain(path: Path = DOMAIN) -> Domain:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Domain(
        resolution=str(payload["observation"]["bar_resolution"]),
        source_ids=tuple(str(s["id"]) for s in payload["sources"]),
        marker=str(payload["out_of_domain_marker"]).lower(),
    )


def scan_resolutions(conf_root: Path, domain: Domain) -> list[str]:
    offenders: list[str] = []
    for path in sorted(conf_root.rglob("*.y*ml")):
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            match = RESOLUTION.search(line)
            if match and match.group(1).strip().strip("\"'") != domain.resolution:
                offenders.append(f"{path}:{number}: {match.group(1)}")
    return offenders


def scan_register(register: Path, domain: Domain) -> list[str]:
    offenders: list[str] = []
    for line in register.read_text(encoding="utf-8").splitlines():
        match = ROW.match(line)
        if not match or not REOPEN_HINT.search(line):
            continue
        lowered = line.lower()
        if domain.marker in lowered:
            continue
        if any(source in lowered for source in domain.source_ids):
            continue
        offenders.append(match.group(1))
    return offenders


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--domain", type=Path, default=DOMAIN)
    parser.add_argument("--conf-root", type=Path, default=REPO / "conf")
    parser.add_argument(
        "--register", type=Path,
        default=REPO / "docs" / "FALSIFICATION_REGISTER.md",
    )
    args = parser.parse_args(argv)

    domain = load_domain(args.domain)
    problems: list[str] = []

    for offender in scan_resolutions(args.conf_root, domain):
        problems.append(
            f"{offender}: declareert een barresolutie die niet "
            f"'{domain.resolution}' is (AD-22)"
        )
    for identifier in scan_register(args.register, domain):
        problems.append(
            f"{identifier}: heropeningsconditie noemt geen bron uit het "
            f"meetdomein {domain.source_ids} en is niet gemarkeerd als "
            f"'{domain.marker}' (AD-23)"
        )

    for problem in problems:
        print(problem)
    if problems and args.strict:
        return 1
    print(f"domeinconsistentie: {len(problems)} openstaande regel(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Stap 2.5 — Markeer F2 en F19 in het register.** Voeg aan beide regels toe: `buiten het meetdomein (AD-23)`. **Wijzig geen bestaand bewijs en geen bestaande formulering** — append binnen de kolom.

- [ ] **Stap 2.6 — Sluit H1 met het bewijs uit §5.4.** Voeg aan de H1-sectie van `FALSIFICATION_REGISTER.md` de tabel uit §5.4 toe met de tekst eronder, letterlijk:

> Op de drie range-proxies lijkt het verschil significant; op `squared_return` — de enige proxy die per constructie zuiver is voor de voorspelde grootheid `E[r_t²|F_{t−1}] = σ_t²` — verdampt het volledig. De range-proxies zijn efficiënter maar niet zuiver; `squared_return` is zuiver maar zeer ruisig. Binnen het meetdomein van AD-23 bestaat geen schatter die beide is, en het verschil is op de dagbar dus niet identificeerbaar. **H1 is daarmee gesloten en niet geblokkeerd**, en EWMA(0,94) is de productie-estimator bij besluit. Dit is expliciet geen bewering dat GARCH slechter is; het is de constatering dat het verschil op de meetbasis niet meetbaar is. De vraag "wijst GARCH een andere TOESTAND toe" is een andere vraag en staat open — zie stap 8.

- [ ] **Stap 2.7 — Verplaats DI-18 naar `## Gesloten`** met AD-23 als grondslag. Doe hetzelfde met het contract van `volatility/har_rv.py` in `docs/CODE_REGISTER.md` klasse D: de grondslag `DI-18` wordt `AD-23 — buiten het meetdomein, geen afnemer meer`.

- [ ] **Stap 2.8 — Voeg de poort toe aan CI.** In `.github/workflows/inventory.yml`, naast de bestaande poorten:

```yaml
      - name: Domeinconsistentie (AD-22/AD-23)
        run: python scripts/check_domain_consistency.py --strict
```

- [ ] **Stap 2.9 — Draai en zie hem slagen.**

Run: `D:/venv/tradebot/Scripts/python.exe -m pytest tests/unit/test_domain_consistency.py -q`
Verwacht: 4 passed.

- [ ] **Stap 2.10 — Commit.**

```bash
git add scripts/check_domain_consistency.py conf/governance/measurement_domain.yaml \
        tests/unit/test_domain_consistency.py \
        docs/FALSIFICATION_REGISTER.md docs/DEFERRED_ISSUES.md docs/CODE_REGISTER.md \
        .github/workflows/inventory.yml
git commit -m "governance(AD-23): meetdomein als whitelist; sluit H1, DI-18, F2 en F19"
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

> **En de regel die revisie 1 miste, R8:** de reset is een claim over de *toekomstige* zoekruimte, en die claim is door de reset zelf niet verifieerbaar. Daarom is hij gekoppeld aan het bevroren poortsample uit stap 4B. Een kandidaat die de ontwikkelsample overleeft, wordt exact één keer op het poortsample gemeten; dat is de enige meting in dit programma waarvan `M` per constructie 1 is.

**Files:**
- Create: `src/tradebot/registry/ledger_reset.py`
- Create: `tests/unit/test_ledger_reset.py`
- Create: `apps/freeze_ledger_reset.py`
- Create: `artefacts/governance/ledger_reset.json` (output)
- Modify: `src/tradebot/registry/trial_counter.py`

**Interfaces:**
- Consumes: `trial_counter.TrialCount`, `trial_counter.M_UNCERTAINTY_NOTE`
- Produces: `ledger_reset.freeze_reset(*, m_new: int, rationale: str, git_sha: str, out: Path, ledger_path: Path | None = None) -> ResetRecord`; `ledger_reset.active_trial_count(*, reset_path: Path, ledger_path: Path) -> ActiveCount`; `ledger_reset.ResetAlreadyExists`

- [ ] **Stap 3.1 — Schrijf de falende test.** Let op de vierde: dat is de negatieve controle die bewijst dat de reset géén blanco cheque is.

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
    result = deflated_sharpe(
        per_bar_best_measured,
        n_obs=1615,
        n_trials=25,
        sr_variance=1.0 / 1615,
        skew=0.0,
        kurtosis=3.0,
        bars_per_year=365,
    )
    assert result.dsr < 0.95


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
"""De eenmalige ledger-reset -- AD-24.

WAAROM DIT EEN EIGEN MODULE IS
==============================
`trial_counter.py` telt trials. Deze module doet iets anders en gevaarlijkers:
hij verlaagt een drempel. Dat hoort niet verstopt te zitten in de teller, want
dan is de handeling niet meer zichtbaar in een diff.

DE VEILIGHEID ZIT NIET IN DE REKENKUNDE
=======================================
Bij `M_new = 25` staat de DSR-eis op 1,74 geannualiseerd. Dat is hoger dan het
VERWACHTE maximum van 2.776 pure ruistrekkingen (1,68 op het gecorrigeerde
venster) -- maar de kans dat dat maximum boven 1,74 uitkomt is in de orde van
een op de vijf. Een ongeregistreerde zoektocht van die omvang haalt dus met
enige regelmaat een resultaat dat deze poort passeert.

De reset is daarom uitsluitend geldig zolang `M_new` het WERKELIJKE aantal
geprobeerde varianten telt, en die belofte wordt niet door dit bestand gedragen
maar door vier dingen buiten dit bestand: de pre-registratie, het bevriezen
vóór de eerste fit, het falsificatieregister (R2), en het afgesloten
poortsample (R8, stap 4B).
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
                "protocol": "AD-24 R1-R8",
                "note": (
                    "De oude telling is GEARCHIVEERD, niet gewist. "
                    "FALSIFICATION_REGISTER.md F1-F20 blijft onverkort bindend; "
                    "dat is de prijs van deze reset (R2). Het bevroren "
                    "poortsample (R8) is de enige externe verificatie."
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

- [ ] **Stap 3.4 — Draai en zie hem slagen.** Verwacht: 5 passed.

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
  --rationale "AD-24: herstart op dagbars (AD-22) met een kandidaatverzameling die per AD-23 binnen het meetdomein valt en disjunct is van de ruimte waarin het merendeel van de 2776 trials is uitgevoerd. F1-F20 blijft onverkort bindend (R2); het poortsample uit stap 4B is de externe verificatie (R8)."
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


def test_a_conditional_branch_counts_as_a_trial(tmp_path: Path) -> None:
    """Nieuw in revisie 2. Stap 9 kent een terugvalpad: haalt de driedelige
    toestand de bezettingspoort niet, dan wordt de tweedelige gemeten. Dat is
    een tweede specificatie op dezelfde data en dus een tweede trial, ook al
    voelt het als hetzelfde experiment.

    Een beslisboom met B takken die op de data wordt doorlopen, kost B trials
    en niet 1. Dit is de rekenregel die revisie 1 ontbrak."""
    path = _reset(tmp_path, m_new=2)
    assert_within_budget(2, reset_path=path)          # k=3 en k=2, beide geboekt
    with pytest.raises(DataContractError):
        assert_within_budget(3, reset_path=path)


def test_remaining_counts_down(tmp_path: Path) -> None:
    path = _reset(tmp_path)
    assert remaining(reset_path=path, ledger_path=LEDGER, booked=0) == 25
    assert remaining(reset_path=path, ledger_path=LEDGER, booked=6) == 19
```

- [ ] **Stap 4.2 — Draai en zie hem falen.**

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

DE REKENREGEL VOOR BESLISBOMEN
==============================
Een campagne met een voorwaardelijk terugvalpad kost het aantal takken dat op
de data wordt doorlopen, niet 1. "Als k=3 de bezettingspoort niet haalt, meet
dan k=2" is twee specificaties op dezelfde data. Boek ze beide, vooraf.
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
        "over het budget gaat, verhoogt zijn eigen lat. Tel ook elke tak van "
        "een voorwaardelijk pad mee. Herzie het plan, of leg een nieuw budget "
        "vast met een eigen grondslag.",
        DataContractError, planned=planned, budget=budget,
    )


def remaining(*, reset_path: Path, ledger_path: Path, booked: int) -> int:
    """Wat er van het budget over is na `booked` geboekte trials."""
    budget = active_trial_count(
        reset_path=reset_path, ledger_path=ledger_path
    ).total
    return budget - int(booked)
```

- [ ] **Stap 4.4 — Draai en zie hem slagen.** Verwacht: 5 passed.

- [ ] **Stap 4.5 — Commit.**

```bash
git add src/tradebot/registry/trial_budget.py tests/unit/test_trial_budget.py
git commit -m "governance(AD-24): trial-budget met een poort op overschrijding en op beslistakken"
```

---

### Stap 4A: De inferentiekern

**Dit is de stap die revisie 1 volledig ontbrak, en zonder haar is elke uitkomst in Stage C een puntschatting zonder bandbreedte.** Zij bouwt één module waarin elke standaardfout, elke toets en elke bootstrap van deze fase woont. Eén implementatie; een tweede is een defect.

**Files:**
- Create: `src/tradebot/validation/inference.py`
- Create: `tests/unit/test_inference.py`
- Modify: `src/tradebot/backtest/metrics.py` (`deflated_sharpe` krijgt de expliciete handtekening)
- Modify: `conf/validation/inference.yaml`

**Interfaces:**
- Produces:
  - `inference.sharpe_with_se(returns, *, bars_per_year, nw_lags=None) -> SharpeEstimate` met `sharpe`, `se`, `t_stat`, `n_obs`, `t_years`, `skew`, `kurtosis`, `nw_lags`
  - `inference.sharpe_difference(a, b, *, bars_per_year, n_boot, seed) -> SharpeDifference` met `delta`, `se`, `t_stat`, `p_value`, `ci_low`, `ci_high`, `n_paired_obs` — Ledoit–Wolf met HAC en gestudentiseerde circulaire blokbootstrap
  - `inference.clustered_mean(frame, *, cluster_axis="index") -> ClusteredMean` met `mean`, `se`, `t_stat`, `n_clusters`
  - `inference.neff_deflation(correlation_matrix) -> float`
  - `metrics.deflated_sharpe(sr_hat, *, n_obs, n_trials, sr_variance, skew, kurtosis, bars_per_year) -> DSRResult`

- [ ] **Stap 4A.1 — Schrijf de falende tests. Vijf eigenschappen, elk met een falsificatie.**

```python
# tests/unit/test_inference.py
"""De inferentiekern. Elke test hier falsifieert een aanname die revisie 1
stilzwijgend maakte.

Deze tests zijn opzettelijk streng op de RICHTING van de correcties, niet op
hun exacte waarde: een correctie die de verkeerde kant op werkt, is erger dan
geen correctie, want zij ziet eruit als zorgvuldigheid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.backtest.metrics import deflated_sharpe
from tradebot.validation.inference import (
    clustered_mean,
    neff_deflation,
    sharpe_difference,
    sharpe_with_se,
)

IDX = pd.date_range("2022-01-01", periods=1615, freq="D", tz="UTC")


def test_naive_se_is_recovered_under_iid_normality() -> None:
    """De Lo-correctie MOET degenereren naar 1/sqrt(T) wanneer de aannames van
    1/sqrt(T) gelden. Doet zij dat niet, dan is zij verkeerd geïmplementeerd."""
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0, 0.01, 1615), index=IDX)
    est = sharpe_with_se(returns, bars_per_year=365, nw_lags=0)
    naive = 1.0 / np.sqrt(1615 / 365)
    assert abs(est.se - naive) / naive < 0.10


def test_fat_tails_inflate_the_standard_error() -> None:
    """Dik-staartige rendementen maken een Sharpe ONZEKERDER. Een SE die daar
    niet op reageert, onderschat de onzekerheid stelselmatig."""
    rng = np.random.default_rng(1)
    thin = pd.Series(rng.normal(0.001, 0.01, 1615), index=IDX)
    fat = pd.Series(rng.standard_t(3, 1615) * 0.01 + 0.001, index=IDX)
    assert (
        sharpe_with_se(fat, bars_per_year=365).se
        > sharpe_with_se(thin, bars_per_year=365).se
    )


def test_positive_autocorrelation_inflates_the_standard_error() -> None:
    """Newey-West, in de goede richting. Positief autogecorreleerde P&L bevat
    minder informatie per bar dan een i.i.d.-reeks van dezelfde lengte."""
    rng = np.random.default_rng(2)
    innovation = rng.normal(0.0, 0.01, 1615)
    ar = np.zeros(1615)
    for i in range(1, 1615):
        ar[i] = 0.4 * ar[i - 1] + innovation[i]
    series = pd.Series(ar + 0.001, index=IDX)
    assert (
        sharpe_with_se(series, bars_per_year=365).se
        > sharpe_with_se(series, bars_per_year=365, nw_lags=0).se
    )


def test_sharpe_difference_is_paired_and_smaller_than_the_level_se() -> None:
    """De kern van stap 10. Twee sterk gecorreleerde reeksen hebben een
    verschil-SE die veel kleiner is dan de SE van elk niveau -- dat is precies
    waarom de gepaarde toets een onderscheid kan maken dat de niveaus niet
    maken."""
    rng = np.random.default_rng(3)
    base = rng.normal(0.0005, 0.01, 1615)
    a = pd.Series(base, index=IDX)
    b = pd.Series(base + rng.normal(0.0, 0.0005, 1615), index=IDX)
    diff = sharpe_difference(a, b, bars_per_year=365, n_boot=2000, seed=7)
    level = sharpe_with_se(a, bars_per_year=365)
    assert diff.se < level.se
    assert diff.n_paired_obs == 1615


def test_sharpe_difference_uses_only_bars_where_both_are_active() -> None:
    """De halt-correctie uit §5.2. Als de ene keten 1559 bars gehalteerd is,
    is een 'gepaard' verschil over alle bars niet gepaard."""
    rng = np.random.default_rng(4)
    a = pd.Series(rng.normal(0.0005, 0.01, 1615), index=IDX)
    b = a.copy()
    b.iloc[200:] = np.nan
    diff = sharpe_difference(a, b, bars_per_year=365, n_boot=500, seed=7)
    assert diff.n_paired_obs == 200


def test_identical_series_have_zero_difference_and_zero_t() -> None:
    rng = np.random.default_rng(5)
    series = pd.Series(rng.normal(0.0, 0.01, 1615), index=IDX)
    diff = sharpe_difference(series, series, bars_per_year=365, n_boot=500, seed=7)
    assert diff.delta == 0.0
    assert diff.t_stat == 0.0


def test_clustering_on_date_shrinks_the_t_statistic_on_a_correlated_panel() -> None:
    """Q4. Zes namen met rho-bar 0,74 zijn geen zes onafhankelijke reeksen.
    Een gepoolde t die daar niet op reageert, overschat met ongeveer een
    factor twee."""
    rng = np.random.default_rng(6)
    common = rng.normal(0.001, 0.01, 1615)
    panel = pd.DataFrame(
        {f"S{i}": common + rng.normal(0.0, 0.004, 1615) for i in range(6)},
        index=IDX,
    )
    pooled = panel.to_numpy().ravel()
    naive_t = pooled.mean() / (pooled.std(ddof=1) / np.sqrt(pooled.size))
    assert abs(clustered_mean(panel).t_stat) < abs(naive_t)


def test_neff_deflation_matches_the_measured_panel() -> None:
    """De gemeten waarde: N_eff = 1,271 bij zes namen geeft factor 0,460."""
    corr = np.full((6, 6), 0.7442)
    np.fill_diagonal(corr, 1.0)
    assert abs(neff_deflation(corr) - 0.460) < 0.02


def test_dsr_requires_its_moments_explicitly() -> None:
    """Q5. De DSR heeft V[SR_m], scheefheid en kurtosis nodig. Een aanroep
    zonder die argumenten mag niet stilzwijgend de normale benadering pakken."""
    with pytest.raises(TypeError):
        deflated_sharpe(0.01, n_obs=1615, n_trials=25)  # type: ignore[call-arg]


def test_dsr_is_stricter_when_the_trial_sharpes_are_more_dispersed() -> None:
    """Hoe wilder de zoektocht spreidde, hoe hoger de lat. Dat is de hele
    gedachte achter de DSR en zij hoort in de code te staan."""
    common = dict(n_obs=1615, n_trials=25, skew=0.0, kurtosis=3.0,
                  bars_per_year=365)
    tight = deflated_sharpe(0.05, sr_variance=1.0 / 1615, **common)
    wide = deflated_sharpe(0.05, sr_variance=4.0 / 1615, **common)
    assert wide.dsr < tight.dsr
```

- [ ] **Stap 4A.2 — Draai en zie ze falen.** Verwacht: `ModuleNotFoundError` plus een `TypeError`-mismatch op `deflated_sharpe`.

- [ ] **Stap 4A.3 — Implementeer `validation/inference.py`.** Eisen, en zij zijn niet onderhandelbaar:

  * `sharpe_with_se` implementeert §3.2 exact. `nw_lags=None` betekent de automatische keuze `floor(4·(T/100)^(2/9))`; `nw_lags=0` schakelt de HAC-opslag uit en bestaat uitsluitend voor de referentiekolom en de eerste test.
  * `sharpe_difference` implementeert §3.3: HAC-covariantie van de vier momenten, plus gestudentiseerde circulaire blokbootstrap (Politis–Romano). De reeksen worden **eerst** op hun gemeenschappelijke actieve index gesneden; `n_paired_obs` rapporteert die lengte. Seed uit `conf/validation/inference.yaml`, want R-5.
  * `clustered_mean` clustert op de tijdsindex.
  * `neff_deflation` geeft `sqrt(N_eff / N)` met `N_eff = N / (1 + (N−1)·ρ̄)`.
  * Elke returnwaarde is een frozen dataclass met een `to_dict()` die het drietal uit §3.1 meeneemt.

- [ ] **Stap 4A.4 — Herschrijf `metrics.py::deflated_sharpe`** naar de handtekening uit §3.9. **Verwijder de oude positionele vorm niet stilzwijgend**: laat hem `TypeError` gooien met een boodschap die naar `MEASUREMENT_CONTRACT.md` §6 verwijst, zodat elke bestaande aanroepplek zichtbaar wordt in plaats van stil door te rekenen met de normale benadering.

- [ ] **Stap 4A.5 — Herstel elke aanroepplek.** Draai de volledige suite en los elke `TypeError` op door de momenten expliciet mee te geven. Waar de trial-Sharpes niet beschikbaar zijn, geef `sr_variance=1/n_obs` mee **met een `approximation="normal"`-vlag die in het artefact belandt.** Een benadering die in een JSON staat, is een keuze; een benadering die in een default staat, is een aanname.

- [ ] **Stap 4A.6 — Draai en zie ze slagen.** Verwacht: 10 passed, en de rest van de suite ongewijzigd (0 failed, 4 xfailed).

- [ ] **Stap 4A.7 — Commit.**

```bash
git add src/tradebot/validation/inference.py tests/unit/test_inference.py \
        src/tradebot/backtest/metrics.py conf/validation/inference.yaml
git commit -m "feat(validation): inferentiekern -- Lo-SE, Newey-West, Ledoit-Wolf, datumclustering, expliciete DSR"
```

---

### Stap 4B: Het poortsample

**Waarom dit bestaat.** De ledger-reset is een belofte over de toekomstige zoekruimte, en die belofte is door de reset zelf niet te verifiëren — dat is de zwakke plek die R8 benoemt. Een afgesloten sample is de enige constructie die dat repareert zonder op discipline te vertrouwen: op dat sample is `M = 1` per constructie, want er is precies één meting mogelijk.

**Wat het kost, eerlijk becijferd.** De laatste 12 maanden afsluiten laat 3,42 jaar over voor ontwikkeling. De t = 2-drempel op de ontwikkelsample stijgt daarmee van **0,951** naar **1,081** — dat is 14 % zwaardere bewijslast. Dat is de prijs, en zij is het waard: zonder poortsample is de enige beveiliging op `M_new = 25` het woord van de uitvoerder.

**Files:**
- Create: `src/tradebot/validation/holdout.py`
- Create: `apps/freeze_holdout.py`
- Create: `tests/unit/test_holdout.py`
- Create: `artefacts/governance/holdout_lock.json` (output)
- Modify: `docs/MEASUREMENT_CONTRACT.md` §2

**Interfaces:**
- Produces: `holdout.freeze_holdout(*, split_utc: str, out: Path, git_sha: str) -> HoldoutLock`; `holdout.development_slice(frame, *, lock_path) -> DataFrame`; `holdout.gate_slice(frame, *, lock_path, hypothesis_id: str) -> DataFrame`; `holdout.HoldoutAlreadyUsed`

- [ ] **Stap 4B.1 — Schrijf de falende test.** De derde is de reden dat deze module bestaat.

```python
# tests/unit/test_holdout.py
"""Het poortsample. Eén split, één meting per hypothese, en een weigering die
niet te omzeilen is zonder een commit die zichtbaar is in een diff."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.validation.holdout import (
    HoldoutAlreadyUsed,
    development_slice,
    freeze_holdout,
    gate_slice,
)

IDX = pd.date_range("2021-10-15", periods=1615, freq="D", tz="UTC")


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"A": rng.normal(0, 0.01, 1615)}, index=IDX)


def _lock(tmp_path: Path) -> Path:
    out = tmp_path / "holdout_lock.json"
    freeze_holdout(split_utc="2025-09-05T00:00:00+00:00", out=out,
                   git_sha="deadbee")
    return out


def test_development_and_gate_slices_are_disjoint_and_exhaustive(
    tmp_path: Path,
) -> None:
    lock, frame = _lock(tmp_path), _frame()
    dev = development_slice(frame, lock_path=lock)
    gate = gate_slice(frame, lock_path=lock, hypothesis_id="H-10.1")
    assert dev.index.max() < gate.index.min()
    assert len(dev) + len(gate) == len(frame)


def test_a_second_read_of_the_gate_slice_for_the_same_hypothesis_is_refused(
    tmp_path: Path,
) -> None:
    """De enige eigenschap die telt. Twee metingen op het poortsample maken M
    op dat sample groter dan 1, en dan is het geen poortsample meer."""
    lock, frame = _lock(tmp_path), _frame()
    gate_slice(frame, lock_path=lock, hypothesis_id="H-10.1")
    with pytest.raises(HoldoutAlreadyUsed):
        gate_slice(frame, lock_path=lock, hypothesis_id="H-10.1")


def test_a_different_hypothesis_may_read_it_once(tmp_path: Path) -> None:
    lock, frame = _lock(tmp_path), _frame()
    gate_slice(frame, lock_path=lock, hypothesis_id="H-10.1")
    gate_slice(frame, lock_path=lock, hypothesis_id="H-10.3")


def test_every_read_is_logged_with_a_timestamp(tmp_path: Path) -> None:
    lock, frame = _lock(tmp_path), _frame()
    gate_slice(frame, lock_path=lock, hypothesis_id="H-10.1")
    import json
    payload = json.loads(lock.read_text(encoding="utf-8"))
    assert payload["reads"][0]["hypothesis_id"] == "H-10.1"
    assert payload["reads"][0]["read_utc"]
```

- [ ] **Stap 4B.2 — Draai en zie hem falen.**

- [ ] **Stap 4B.3 — Implementeer.** `gate_slice` schrijft de lees-entry **vóór** hij de data teruggeeft. Een uitvoerder die de run afbreekt na de meting maar vóór het loggen, heeft dan alsnog een geregistreerde lezing — en dat is de goede kant om deze race op te lossen.

- [ ] **Stap 4B.4 — Bevries de split.**

```bash
D:/venv/tradebot/Scripts/python.exe apps/freeze_holdout.py \
  --split-utc 2025-09-05T00:00:00+00:00
```

- [ ] **Stap 4B.5 — Werk `MEASUREMENT_CONTRACT.md` §2 bij** met beide vensters: `development` (t_years = 3,42, t=2-drempel **1,081**) en `gate` (t_years = 1,00, uitsluitend voor de eenmalige meting). Elke Sharpe in elk rapport noemt vanaf nu welk van de twee vensters eronder ligt.

- [ ] **Stap 4B.6 — Commit.**

```bash
git add src/tradebot/validation/holdout.py apps/freeze_holdout.py \
        tests/unit/test_holdout.py artefacts/governance/holdout_lock.json \
        docs/MEASUREMENT_CONTRACT.md
git commit -m "feat(validation): bevroren poortsample -- de externe verificatie op AD-24 R8"
```

---

# STAGE B — DE VOLATILITEITSTOESTAND

*De correctie, uitgevoerd. Elke stap hier is diagnostisch tenzij anders vermeld: er wordt niets uit geselecteerd en niets uit gepromoveerd. Alles in Stage B draait uitsluitend op de ontwikkelsample uit stap 4B.*

---

### Stap 5: Het `VolState`-contract

Er bestaat nu geen type voor "toestand". `regime/buckets.py` levert een `VolBucket(IntEnum)`, `regime/markov.py` levert kansen, `alpha/macro_regime.py` levert een `RegimeLabel`-enum, en `conditioning.py` platwalst alles tot één float. Dat is de reden dat de toestand nergens als toestand kan worden getoetst.

**Verhouding tot het bestaande `VolBucket`.** `regime/buckets.py::VolBucket` is M0's SPECIFIEKE classificatie (twee assen, conjunctie op z-score en ATR-ratio). `VolState` is het GENERIEKE contract waar elke vol-toestandstoewijzer op uitkomt — M0, EWMA-kwantielen, een HMM. M0 blijft ongewijzigd en krijgt in stap 6 een adapter naar `VolState`; er wordt geen tweede M0 gebouwd.

> **De correctie van Q1, en zij is de belangrijkste in Stage B.** Revisie 1 wees de toestand van bar `t` toe op basis van σ̂ op bar `t` en een expanding kwantiel dat bar `t` zélf bevatte. Dat is tweemaal dezelfde fout: σ̂_t uit EWMA bevat `r_t`, en het kwantiel bevat `σ̂_t`. De causaliteitstest van revisie 1 toetste alleen truncatie-invariantie — en die is *waar* voor een gelijktijdige toewijzing, want truncatie op `t` verandert de toestand op `t` niet.
>
> Vanaf revisie 2 is de lag expliciet en getest: de toestand van bar `t` is een functie van `σ̂_{t−1}` en van kwantielen over `{σ̂_s : s ≤ t−1}`. De nieuwe test spuit een spike in σ̂ en bewijst dat die pas in de toestand van de *volgende* bar zichtbaar wordt.

**Files:**
- Create: `src/tradebot/regime/state.py`
- Create: `tests/unit/test_vol_state.py`
- Create: `tests/lookahead/test_vol_state_causality.py`

**Interfaces:**
- Produces: `state.VolState` (IntEnum: `LOW=0, NORMAL=1, HIGH=2`), `state.StateAssignment` (frozen dataclass met `states`, `source`, `ordering`, `params`, methodes `occupancy()` en `episodes()`), en
  `state.assign_by_variance(sigma, *, low_q, high_q, min_periods, source="unspecified", lag=1) -> StateAssignment`

- [ ] **Stap 5.1 — Schrijf de falende tests.**

```python
# tests/unit/test_vol_state.py
"""Het toestandscontract. Vijf eigenschappen, elk met een eigen falsificatie:
de ordening is op variantie en niet op gemiddelde; de toestand is discreet; de
opstartfase is NaN en geen ingevulde toestand; de toewijzing draagt haar
herkomst mee; en de kwantielen moeten oplopen."""
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
    zou een richtingsclaim zijn, en die wordt op dit paneel niet gedragen: na
    datumclustering en N_eff-deflatie haalt geen enkele richtings-t 0,8."""
    sigma = _sigma()
    result = assign_by_variance(sigma, low_q=0.25, high_q=0.75, min_periods=100)
    known = sigma.shift(1)
    for column in sigma.columns:
        states = result.states[column]
        mask = states.notna()
        high = known[column][mask][states[mask] == VolState.HIGH]
        low = known[column][mask][states[mask] == VolState.LOW]
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
    head = result.states.iloc[:100]
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


def test_occupancy_and_episodes_are_both_reported() -> None:
    """Q7. Bars en episodes zijn verschillende grootheden, en de tweede is de
    effectieve steekproefomvang van een persistente toestand."""
    result = assign_by_variance(_sigma(), low_q=0.25, high_q=0.75, min_periods=100)
    occ, epi = result.occupancy(), result.episodes()
    assert set(occ.columns) == {"LOW", "NORMAL", "HIGH"}
    assert set(epi.columns) == {"LOW", "NORMAL", "HIGH"}
    assert (epi.to_numpy() <= occ.to_numpy() * len(result.states)).all()
```

```python
# tests/lookahead/test_vol_state_causality.py
"""R-1 op de toestand, en dit is de test die revisie 1 miste.

Truncatie-invariantie is NIET voldoende: een gelijktijdige toewijzing is
truncatie-invariant, want het afkappen van de reeks na bar t verandert de
toestand op bar t niet. De eigenschap die je wilt, is dat een schok in sigma op
bar t de toestand op bar t NIET raakt.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.regime.state import VolState, assign_by_variance

IDX = pd.date_range("2022-01-01", periods=500, freq="D", tz="UTC")


def test_truncation_cannot_change_an_earlier_state() -> None:
    rng = np.random.default_rng(7)
    sigma = pd.DataFrame({"A": np.abs(rng.normal(0.5, 0.2, 500))}, index=IDX)
    full = assign_by_variance(sigma, low_q=0.25, high_q=0.75, min_periods=100)
    for cut in (200, 300, 400):
        truncated = assign_by_variance(
            sigma.iloc[:cut], low_q=0.25, high_q=0.75, min_periods=100,
        )
        pd.testing.assert_frame_equal(
            truncated.states, full.states.iloc[:cut], check_freq=False,
        )


def test_a_spike_in_sigma_cannot_change_the_state_of_its_own_bar() -> None:
    """De harde causaliteitstest. Zet op bar 300 een extreme sigma en toon aan
    dat de toestand op bar 300 ONVERANDERD blijft en dat de HIGH pas op bar 301
    verschijnt. Een gelijktijdige toewijzing faalt hier onmiddellijk."""
    rng = np.random.default_rng(8)
    sigma = pd.DataFrame({"A": np.abs(rng.normal(0.5, 0.05, 500))}, index=IDX)
    spiked = sigma.copy()
    spiked.iloc[300, 0] = 50.0

    base = assign_by_variance(sigma, low_q=0.25, high_q=0.75, min_periods=100)
    after = assign_by_variance(spiked, low_q=0.25, high_q=0.75, min_periods=100)

    assert after.states.iloc[300, 0] == base.states.iloc[300, 0]
    assert after.states.iloc[301, 0] == float(VolState.HIGH)


def test_the_quantile_boundaries_use_only_past_bars() -> None:
    """Tweede helft van Q1. Niet alleen sigma moet gelagged zijn, ook de
    DREMPEL. Een reeks die na bar 400 explodeert, mag de toestanden vóór 400
    niet verschuiven -- ook niet via het kwantiel."""
    rng = np.random.default_rng(9)
    sigma = pd.DataFrame({"A": np.abs(rng.normal(0.5, 0.05, 500))}, index=IDX)
    exploded = sigma.copy()
    exploded.iloc[400:, 0] *= 20.0

    base = assign_by_variance(sigma, low_q=0.25, high_q=0.75, min_periods=100)
    after = assign_by_variance(exploded, low_q=0.25, high_q=0.75, min_periods=100)

    pd.testing.assert_frame_equal(
        after.states.iloc[:400], base.states.iloc[:400], check_freq=False,
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
(H1, gesloten op de meetbasis), en als schaalvermenigvuldiger op de exposure
(AD-15). Dat tweede kan per constructie niet werken -- AD-16 meet dat L7 elke
cross-sectioneel uniforme schaal er weer uitdeelt.

Een volatiliteitsmodel bepaalt de volatiliteit en kent op grond daarvan een
DISCRETE TOESTAND toe. Die toestand conditioneert een besluit, niet een schaal.
Zie REGEL V in `Prompts-fases/fase_10_herstart_dagbars.md` §4.3.

DE LAG IS NIET COSMETISCH
=========================
De toestand van bar t is een functie van sigma-dak op t-1 en van kwantielen
over bars <= t-1. Beide helften zijn nodig:

* sigma-dak op t bevat r_t. Een toestand die daarop is gebaseerd, kent het
  rendement dat hij zou moeten voorspellen.
* Een expanding kwantiel dat bar t bevat, laat de DREMPEL van de observatie
  afhangen die hij classificeert.

Truncatie-invariantie ziet geen van beide fouten; `tests/lookahead/
test_vol_state_causality.py` wel.

DE ORDENING IS OP VARIANTIE, EN DAT IS EEN BESLUIT
===================================================
Een k-toestands-HMM schat per toestand zowel een gemiddelde als een variantie
(`regime/markov.py`, `means: (k, d)`). De verleiding is om te ordenen op het
GEMIDDELDE -- dat levert 0 = bearish, 1 = flat, 2 = bullish.

Op dit paneel wordt die lezing niet gedragen. Gemeten met de bestaande M0
vol-buckets over 10.400 symbool-bars, met de gepoolde t naast de t die op
datum is geclusterd en met N_eff = 1,271 is gedefleerd:

    toestand   n      ann. rendement   ann. vol   t_pooled   t_eff
    LAAG       1720   + 49,6 %          62,6 %      1,72      0,79
    NORMAAL    8181   +  9,5 %          79,2 %      0,57      0,26
    HOOG        499   +136,4 %         137,0 %      1,16      0,53

De volatiliteit scheidt met een factor 2,2 en dat is betrouwbaar. Het rendement
scheidt niet: geen enkele gedefleerde t haalt 0,8, en de tekens spreken elkaar
per symbool tegen (DOTUSDT +131,6 bp/dag in HOOG, SOLUSDT -45,1 bp).

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
        """Aandeel bars per toestand per symbool."""
        rows = {}
        for column in self.states.columns:
            series = self.states[column].dropna()
            total = max(len(series), 1)
            rows[column] = {
                state.name: float((series == int(state)).sum()) / total
                for state in VolState
            }
        return pd.DataFrame(rows).T

    def episodes(self) -> pd.DataFrame:
        """Aantal aaneengesloten EPISODES per toestand per symbool.

        Dit is de grootheid die de bezettingspoort in stap 9 nodig heeft. Een
        toestand die 499 bars beslaat in twaalf episodes, levert twaalf
        onafhankelijke observaties en geen 499: de persistentie zit in de
        toestand zelf.
        """
        rows = {}
        for column in self.states.columns:
            series = self.states[column]
            changed = series.ne(series.shift(1)) & series.notna()
            rows[column] = {
                state.name: int((changed & (series == int(state))).sum())
                for state in VolState
            }
        return pd.DataFrame(rows).T

    def mean_duration(self) -> pd.DataFrame:
        """Gemiddelde episodelengte in bars -- occupancy x n_bars / episodes."""
        occ = self.occupancy()
        epi = self.episodes().replace(0, np.nan)
        n_bars = self.states.notna().sum()
        return occ.mul(n_bars, axis=0).div(epi)


def assign_by_variance(
    sigma: pd.DataFrame,
    *,
    low_q: float,
    high_q: float,
    min_periods: int,
    source: str = "unspecified",
    lag: int = 1,
) -> StateAssignment:
    """Wijs elke bar een VolState toe op EXPANDING kwantielen van gelagde sigma.

    Expanding en niet rolling, en al helemaal niet over de volledige sample:
    dat laatste is DI-2 en het zou de toestand van bar `t` laten afhangen van
    bars die op `t` nog niet bestonden.

    `lag=1` is geen instelling maar het contract. Hij staat in de handtekening
    zodat een test hem op 0 kan zetten en kan AANTONEN dat de causaliteitstest
    dan faalt -- een poort die niet rood kan worden, bewijst niets.
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
    require(
        lag >= 0,
        "Een negatieve lag is een lookahead met een vriendelijke naam.",
        DataContractError, lag=lag,
    )

    known = sigma.shift(lag).astype("float64")
    states = pd.DataFrame(
        np.nan, index=sigma.index, columns=sigma.columns, dtype="float64"
    )
    for column in sigma.columns:
        series = known[column]
        # Het kwantiel loopt over dezelfde gelagde reeks, dus de drempel op bar
        # t is een functie van {sigma_s : s <= t-lag}. Beide helften van Q1.
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
                "min_periods": float(min_periods), "lag": float(lag)},
    )
```

- [ ] **Stap 5.4 — Voeg de drempels toe aan `conf/model/regime.yaml`** onder een nieuwe sleutel `state:` (`low_q: 0.25`, `high_q: 0.75`, `min_periods: 250`, `lag: 1`) en breid `schemas/config.py` uit met een `VolStateConfig` (`extra="forbid"`, `frozen=True`).

  > **En registreer deze twee getallen als wat ze zijn.** `low_q = 0,25` en `high_q = 0,75` zijn vrije parameters. Zolang zij vóór elke meting worden bevroren en nooit op grond van een uitkomst worden aangepast, kosten zij nul trials. Wordt er ooit een tweede paar geprobeerd, dan kost dat een trial — en dan is de eerste er retroactief ook een. Zet die zin letterlijk in het commentaarblok van de config.

- [ ] **Stap 5.5 — Bewijs dat de causaliteitstest rood kan worden.** Voeg een test toe die `lag=0` doorgeeft en aantoont dat `test_a_spike_in_sigma_cannot_change_the_state_of_its_own_bar` dan faalt:

```python
def test_lag_zero_reintroduces_the_leak_and_the_test_catches_it() -> None:
    """De negatieve controle op de causaliteitstest zelf."""
    rng = np.random.default_rng(8)
    sigma = pd.DataFrame({"A": np.abs(rng.normal(0.5, 0.05, 500))}, index=IDX)
    spiked = sigma.copy()
    spiked.iloc[300, 0] = 50.0
    leaky = assign_by_variance(
        spiked, low_q=0.25, high_q=0.75, min_periods=100, lag=0,
    )
    assert leaky.states.iloc[300, 0] == float(VolState.HIGH)
```

- [ ] **Stap 5.6 — Draai en zie ze slagen.** Verwacht: 6 passed (unit) + 4 passed (lookahead).

- [ ] **Stap 5.7 — Commit.**

```bash
git add src/tradebot/regime/state.py tests/unit/test_vol_state.py \
        tests/lookahead/test_vol_state_causality.py conf/model/regime.yaml \
        src/tradebot/schemas/config.py
git commit -m "feat(regime): VolState-contract -- discreet, geordend op variantie, causaal met expliciete lag"
```

---

### Stap 6: De toestandsdiagnose — wat draagt de toestand?

Diagnostiek, expliciet niet-promoveerbaar. Er wordt niets uit geselecteerd, dus deze stap kost **nul trials** — mits stap 5.4 is nageleefd en er geen enkele parameter op grond van deze uitkomst wordt gewijzigd.

**Files:**
- Create: `src/tradebot/regime/state_diagnostics.py`
- Create: `apps/run_state_diagnostics.py`
- Create: `tests/unit/test_state_diagnostics.py`
- Create: `reports/phase10_state_diagnostics.md` (output)
- Create: `artefacts/governance/phase10_state_diagnostics.json` (output)
- Modify: `dvc.yaml`

**Interfaces:**
- Consumes: `state.StateAssignment`, `state.VolState`, `validation/inference.py`
- Produces: `state_diagnostics.diagnose(assignment, forward_returns, *, bars_per_year) -> StateDiagnostics` met `separation_vol`, `separation_return`, `occupancy`, `episodes`, `mean_duration`, `transition_matrix`, `per_symbol`

- [ ] **Stap 6.1 — Schrijf de falende test.**

```python
# tests/unit/test_state_diagnostics.py
"""De diagnose moet BEIDE assen rapporteren, de richtingsas met haar
onzekerheid, en die onzekerheid moet op datum geclusterd zijn -- anders leest
iemand een gepoolde 1,16 als een bevinding terwijl de gedefleerde waarde 0,53
is."""
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


def _diagnose():
    sigma, returns = _fixture()
    return diagnose(
        assign_by_variance(sigma, low_q=0.25, high_q=0.75, min_periods=100),
        forward_returns=returns,
        bars_per_year=365,
    )


def test_reports_both_axes() -> None:
    result = _diagnose()
    assert set(result.separation_vol) == {"LOW", "NORMAL", "HIGH"}
    assert set(result.separation_return) == {"LOW", "NORMAL", "HIGH"}


def test_the_direction_axis_carries_clustered_uncertainty() -> None:
    """Zonder t-statistiek is een conditioneel gemiddelde geen bevinding maar
    een getal. En zonder clustering is de t-statistiek zelf geen bevinding."""
    result = _diagnose()
    for state in ("LOW", "NORMAL", "HIGH"):
        cell = result.separation_return[state]
        for field in ("n_bars", "n_episodes", "t_stat_pooled",
                      "t_stat_clustered", "t_stat_neff_deflated", "se"):
            assert field in cell


def test_the_deflated_t_is_never_larger_than_the_pooled_t() -> None:
    """De deflatie moet de goede kant op werken. Een 'correctie' die de t
    vergroot, ziet eruit als zorgvuldigheid en is het tegendeel."""
    result = _diagnose()
    for state in ("LOW", "NORMAL", "HIGH"):
        cell = result.separation_return[state]
        assert abs(cell["t_stat_neff_deflated"]) <= abs(cell["t_stat_pooled"]) + 1e-9


def test_persistence_is_reported_as_a_transition_matrix() -> None:
    """Een toestand waarop je handelt, moet lang genoeg duren om erop te
    handelen. De diagonaal van de overgangsmatrix is die eigenschap."""
    result = _diagnose()
    assert result.transition_matrix.shape == (3, 3)
    assert np.allclose(result.transition_matrix.sum(axis=1), 1.0)


def test_synthetic_returns_with_no_state_dependence_show_no_separation() -> None:
    """Negatieve controle. Op returns die per constructie NIET van de toestand
    afhangen, mag geen enkele t boven 2 uitkomen."""
    result = _diagnose()
    assert all(
        abs(result.separation_return[s]["t_stat_clustered"]) < 2.0
        for s in ("LOW", "NORMAL", "HIGH")
    )


def test_per_symbol_results_are_not_hidden_behind_the_pool() -> None:
    """De nulmeting toont tegenstrijdige tekens per symbool. Een gepoolde tabel
    verbergt dat, en dat is precies de informatie die telt."""
    result = _diagnose()
    assert "A" in result.per_symbol
```

- [ ] **Stap 6.2 — Draai en zie hem falen.**

- [ ] **Stap 6.3 — Implementeer `state_diagnostics.py`.** Rapporteer per toestand:

  | Veld | Waarom het erin staat |
  |---|---|
  | `n_bars`, `occupancy` | bezetting, invoer van stap 9 |
  | `n_episodes`, `mean_duration_bars` | de effectieve steekproefomvang (Q7) |
  | `ann_vol` | de as waarop de toestand aantoonbaar scheidt |
  | `ann_return`, `se`, `t_stat_pooled`, `t_stat_clustered`, `t_stat_neff_deflated` | de as waarop hij dat niet doet, met drie steeds strengere lezingen (§3.4) |
  | `transition_matrix` | persistentie; een toestand die per bar wisselt, is niet handelbaar |
  | `per_symbol` | de tekens spreken elkaar tegen en dat mag niet in de pool verdwijnen |

  Alle standaardfouten komen uit `validation/inference.py`. Geen tweede implementatie.

- [ ] **Stap 6.4 — Schrijf de app en de DVC-stage.** `apps/run_state_diagnostics.py` (≤ 80 LOC) leest de PIT-store, snijdt op de **ontwikkelsample** (`holdout.development_slice`), bouwt σ̂ met `volatility/ewma.py::ewma_volatility_panel`, wijst toe met `assign_by_variance` en schrijft beide outputs. Voeg de stage toe aan `dvc.yaml` met `data/pit_store`, `conf/model/regime.yaml`, `conf/model/volatility.yaml` en `artefacts/governance/holdout_lock.json` als deps.

- [ ] **Stap 6.5 — Draai en vergelijk met de nulmeting.**

```bash
D:/venv/tradebot/Scripts/python.exe apps/run_state_diagnostics.py
```

Verwacht: de vol-separatie reproduceert de orde van grootte uit §4.5 (62,6 % / 79,2 % / 137,0 %) en geen enkele gedefleerde richtings-t haalt 1. **Wijkt dat af, dan is dát de bevinding** en die gaat vóór alle volgende stappen.

  > Twee redenen waarom kleine afwijkingen hier verwacht zijn en géén alarm: de meting loopt nu op de ontwikkelsample (3,42 j in plaats van 4,42 j), en de toestand is nu gelagged (stap 5). Rapporteer beide effecten apart, zodat een lezer kan zien welk deel van het verschil uit de lag komt. **Als de lag de vol-separatie merkbaar verkleint, is dat een belangrijke bevinding**: het zou betekenen dat een deel van de "separatie" in revisie 1 uit de gelijktijdigheid kwam.

- [ ] **Stap 6.6 — Schrijf `reports/phase10_state_diagnostics.md`.** Verplichte secties:
  1. De twee assen naast elkaar, met alle drie de t-lezingen.
  2. Bezetting én episodes per toestand per symbool.
  3. De overgangsmatrix en de gemiddelde episodelengte.
  4. Een paragraaf **"Wat deze diagnose NIET vaststelt"**: geen richtingsclaim, geen alfaclaim (§3.7), en identificatie op variantie.
  5. Een paragraaf **"Het effect van de lag"** met de vergelijking tegen de ongelagde toewijzing uit revisie 1.

- [ ] **Stap 6.7 — Commit.**

```bash
git add src/tradebot/regime/state_diagnostics.py apps/run_state_diagnostics.py \
        tests/unit/test_state_diagnostics.py reports/phase10_state_diagnostics.md \
        artefacts/governance/phase10_state_diagnostics.json dvc.yaml
git commit -m "feat(regime): toestandsdiagnose -- vol-separatie meetbaar, richting niet, beide met geclusterde SE"
```

---

### Stap 7: De poort die REGEL V respecteert

**Files:**
- Create: `src/tradebot/regime/state_mapping.py`
- Create: `tests/unit/test_state_mapping.py`
- Modify: `docs/ARCHITECTURAL_DECISIONS.md` (AD-25)

**Interfaces:**
- Consumes: `state.StateAssignment`, `state.VolState`
- Produces: `state_mapping.gate_by_state(exposures, assignment, *, flat_states: frozenset[VolState]) -> pd.DataFrame`; `state_mapping.concentration_report(exposures, gated) -> ConcentrationReport`

- [ ] **Stap 7.1 — Schrijf de falende tests.** De eerste is de belangrijkste van deze hele fase: hij bewijst dat de afbeelding L7 overleeft. De laatste twee zijn nieuw in revisie 2 en dekken Q9.

```python
# tests/unit/test_state_mapping.py
"""REGEL V, afdwingbaar gemaakt.

AD-16 meet dat een cross-sectioneel UNIFORME schaal onzichtbaar is: L7
herschaalt naar het vol-target en deelt elke constante c er weer uit. Een
toestandsafbeelding die alleen de OMVANG raakt, is daarmee per constructie een
lege operatie. Deze tests bewijzen dat `gate_by_state` de SAMENSTELLING raakt.

Nieuw in revisie 2: de poort zet namen op nul, waarna L7 de resterende namen
OPHOOGT om het vol-target te halen. Die concentratie is een risico dat de poort
introduceert, en zij hoort gemeten en begrensd te worden (Q9).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.regime.state import VolState, assign_by_variance
from tradebot.regime.state_mapping import concentration_report, gate_by_state
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


def _gated():
    exposures, assignment = _fixture()
    return exposures, assignment, gate_by_state(
        exposures, assignment, flat_states=frozenset({VolState.HIGH}),
    )


def test_the_mapping_survives_vol_targeting() -> None:
    """De kerntest. Normaliseer beide boeken op gelijke bruto exposure -- de
    operatie die L7 uitvoert -- en zij MOETEN dan nog verschillen. Een
    schaalafbeelding zou hier identiek worden."""
    exposures, _, gated = _gated()

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


def test_a_uniform_multiplier_would_fail_that_same_test() -> None:
    """De negatieve controle op REGEL V zelf. Zonder deze test bewijst de
    vorige test niets over de POORT -- alleen dat er iets is veranderd."""
    exposures, assignment, _ = _gated()
    scaled = exposures.mul(0.5)

    def gross_normalised(frame: pd.DataFrame) -> pd.DataFrame:
        gross = frame.abs().sum(axis=1).replace(0.0, np.nan)
        return frame.div(gross, axis=0)

    base = gross_normalised(exposures).dropna()
    uniform = gross_normalised(scaled).dropna()
    common = base.index.intersection(uniform.index)
    assert np.allclose(
        base.loc[common].to_numpy(), uniform.loc[common].to_numpy(),
    )


def test_a_gated_state_produces_exactly_zero_exposure() -> None:
    exposures, assignment, gated = _gated()
    for column in exposures.columns:
        high = assignment.states[column] == float(VolState.HIGH)
        assert (gated[column][high].fillna(0.0) == 0.0).all()


def test_an_ungated_state_is_passed_through_untouched() -> None:
    """Geen dempingsfactor, geen herschaling: de view of geen view."""
    exposures, assignment, gated = _gated()
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
    exposures, assignment, _ = _gated()
    with pytest.raises(DataContractError):
        gate_by_state(
            exposures, assignment,
            flat_states=frozenset({VolState.LOW, VolState.NORMAL, VolState.HIGH}),
        )


def test_burn_in_stays_nan_and_is_not_gated_to_zero() -> None:
    """Geen toestand is geen besluit. Nul zou 'wij kiezen vlak' betekenen."""
    exposures, assignment, gated = _gated()
    unknown = assignment.states.isna()
    assert gated.where(unknown).isna().all().all()


def test_the_gate_increases_concentration_and_that_is_measured() -> None:
    """Q9. De poort zet namen op nul; L7 hoogt de rest op. Die concentratie is
    een NIEUW risico dat de poort introduceert en het hoort in het artefact."""
    exposures, _, gated = _gated()
    report = concentration_report(exposures, gated)
    assert report.max_weight_gated >= report.max_weight_base - 1e-12
    assert report.effective_names_gated <= report.effective_names_base + 1e-12


def test_a_fully_gated_bar_is_flat_and_not_infinitely_levered() -> None:
    """De randgeval-test. Zijn ALLE namen op een bar gepoort, dan is het boek
    vlak. L7 mag daar niet op reageren met een deling door nul."""
    exposures, assignment = _fixture()
    all_high = assignment.states.copy()
    all_high.iloc[200:] = float(VolState.HIGH)
    from dataclasses import replace
    gated = gate_by_state(
        exposures, replace(assignment, states=all_high),
        flat_states=frozenset({VolState.HIGH}),
    )
    assert (gated.iloc[200:].abs().sum(axis=1) == 0.0).all()
```

- [ ] **Stap 7.2 — Draai en zie hem falen.**

- [ ] **Stap 7.3 — Implementeer `state_mapping.py`.** De afbeelding is een **poort**, niet een factor:

```
s_t in flat_states   ->  0        (geen view)
s_t elders           ->  a_t      (volle view, ongeschaald)
s_t onbekend         ->  NaN      (geen besluit)
```

Nul vrije parameters zodra `flat_states` is geregistreerd. Geen multiplier per toestand — dat zou per REGEL V `k − 1` extra parameters zijn, dus `k − 1` trials, en het zou het experiment in een sizing-experiment veranderen (het alternatief dat AD-15 al afwees).

`concentration_report` levert per bar: `max_weight`, `effective_names` (`1/Σw²` op genormaliseerde absolute gewichten) en `gross_after_vol_target`, elk voor het basisboek en het gepoorte boek, plus de kwantielen 50/95/99 over de tijd.

- [ ] **Stap 7.4 — Schrijf AD-25.** Titel: *"Een toestand conditioneert de samenstelling, nooit de schaal"*. Besluit: REGEL V in de gepreciseerde vorm uit §4.3 — inclusief het onderscheid tussen cross-sectioneel uniforme en gedifferentieerde factoren, want de ruwe versie ("elke vermenigvuldiging is redundant") is niet waar en een AD hoort geen onware bewering te bevatten. Waarom: AD-16's meting. Het afgewezen alternatief: `a × (1 − p_hoog)`. Bewaakt door `test_the_mapping_survives_vol_targeting` en `test_a_uniform_multiplier_would_fail_that_same_test`.

- [ ] **Stap 7.5 — Draai en zie hem slagen.** Verwacht: 8 passed.

- [ ] **Stap 7.6 — Commit.**

```bash
git add src/tradebot/regime/state_mapping.py tests/unit/test_state_mapping.py \
        docs/ARCHITECTURAL_DECISIONS.md
git commit -m "feat(regime): AD-25 -- toestandspoort op samenstelling, met concentratiemeting"
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
- Produces: `state_agreement.agreement(a, b) -> AgreementReport` met `fraction_identical`, `cohen_kappa`, `confusion` (3×3 per symbool), `n_decision_changes`, `fraction_bars_with_decision_change`

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
    assert report.cohen_kappa == 1.0


def test_different_sigma_paths_disagree_somewhere() -> None:
    report = agreement(_assignment(5), _assignment(6))
    assert report.fraction_identical < 1.0


def test_agreement_is_reported_against_chance_not_only_in_raw_percent() -> None:
    """Bij 79 % NORMAAL is 80 % ruwe overeenstemming bijna niets. Cohens kappa
    corrigeert voor de bezetting en is daarom de maat die telt."""
    report = agreement(_assignment(5), _assignment(6))
    assert report.cohen_kappa < report.fraction_identical


def test_confusion_matrix_is_three_by_three_per_symbol() -> None:
    report = agreement(_assignment(5), _assignment(6))
    assert report.confusion["A"].shape == (3, 3)
```

- [ ] **Stap 8.2 — Draai en zie hem falen.**

- [ ] **Stap 8.3 — Implementeer `state_agreement.py`.** Naast de ruwe overeenstemming **verplicht** Cohens κ: bij een bezetting van 79 % NORMAAL is 80 % ruwe overeenstemming vrijwel gelijk aan toeval, en een rapport dat alleen het ruwe percentage noemt, leest als sterk bewijs voor iets wat niets zegt.

- [ ] **Stap 8.4 — Schrijf de app.** Bouw twee σ̂-panelen op de **ontwikkelsample** — EWMA(0,94) uit `volatility/ewma.py` en de walk-forward GARCH(1,1)-t uit `volatility/garch.py`, met de bestaande foldstructuur en de purge/embargo uit §3.8 — wijs beide toe met `assign_by_variance` op dezelfde bevroren drempels, en rapporteer.

- [ ] **Stap 8.5 — Draai en schrijf het rapport.** `reports/phase10_state_agreement.md` beantwoordt één vraag expliciet, met een vooraf vastgelegde beslisregel:

> **Beslisregel, vooraf.** Wijzen de twee modellen op ≥ 95 % van de bars dezelfde toestand toe **en** is `fraction_bars_with_decision_change` < 5 %, dan is de keuze tussen hen economisch irrelevant en is H1 ook op de toestandsas beslecht — zonder proxy, zonder DM-toets en zonder trial. EWMA(0,94) blijft de productie-estimator.
>
> Blijft de overeenstemming daaronder, dan is het verschil een **kandidaat** en gaat het als hypothese naar Stage C, met de trial-kosten die daarbij horen. Het gaat **niet** stilzwijgend mee als "betere estimator".

- [ ] **Stap 8.6 — Commit.**

```bash
git add src/tradebot/regime/state_agreement.py apps/run_state_agreement.py \
        tests/unit/test_state_agreement.py reports/phase10_state_agreement.md
git commit -m "feat(regime): proxyvrije EWMA/GARCH-vergelijking op de toestandsas, met kappa"
```

---

### Stap 9: De bezettings- en episodepoort

H2 kreeg van de adequaatheidspoort *a priori* groen licht en liep daarna vast op de gerealiseerde bezetting. Gemeten in `artefacts/governance/phase6_h2_regime_benchmark.json`:

| | a priori | gerealiseerd |
|---|---|---|
| `hmm_k3`, zeldzaamste toestand | 165,0 obs per fold — `adequate: true` | **5,0 tot 28,0** obs per fold |
| bron van de aanname | `"uniform (1/k) — optimistisch"` | de fit zelf |

De poort keurde goed op een aanname die het artefact zelf **"optimistisch"** noemt. Dat is een defect in de poort, niet in het model.

> **En het tweede defect, dat revisie 1 liet staan (Q7).** Zelfs een poort op de gerealiseerde bezetting telt nog steeds *bars*. Toestanden zijn persistent: 499 HOOG-bars in twaalf episodes zijn twaalf observaties, niet 499. Een poort die bars telt, keurt een toestand goed die uit drie clusters bestaat. De poort telt daarom **beide** en oordeelt op het strengste.

**Files:**
- Modify: `src/tradebot/validation/data_adequacy.py`
- Create: `tests/unit/test_realised_occupancy_gate.py`
- Modify: `conf/model/adequacy.yaml`

**Interfaces:**
- Produces: `data_adequacy.assert_realised_occupancy(assignment, *, n_folds, min_obs_per_state_per_fold, min_episodes_per_state_per_fold, min_occupancy_fraction, raise_on_failure=True) -> OccupancyVerdict` met `.adequate: bool` en `.measured: Mapping[str, Any]`

- [ ] **Stap 9.1 — Schrijf de falende test.**

```python
# tests/unit/test_realised_occupancy_gate.py
"""De poort die H2 had moeten tegenhouden vóór de fit in plaats van erna.

De a-priori-poort rekent met een uniforme bezetting van 1/k en noemt dat in het
artefact zelf 'optimistisch'. Voor hmm_k3 gaf dat 165,0 observaties per fold en
`adequate: true`; gerealiseerd waren het er 5,0 tot 28,0 tegen een eis van 100.

Revisie 2 voegt de episodetelling toe: bars zijn niet de effectieve
steekproefomvang van een persistente toestand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.regime.state import assign_by_variance
from tradebot.validation.data_adequacy import assert_realised_occupancy
from tradebot.utils.failfast import DataContractError

IDX = pd.date_range("2020-01-01", periods=1200, freq="D", tz="UTC")
LIMITS = dict(min_obs_per_state_per_fold=50,
              min_episodes_per_state_per_fold=10,
              min_occupancy_fraction=0.10)


def _balanced():
    rng = np.random.default_rng(2)
    sigma = pd.DataFrame({"A": np.abs(rng.normal(0.5, 0.2, 1200))}, index=IDX)
    return assign_by_variance(sigma, low_q=0.30, high_q=0.70, min_periods=100)


def _starved():
    """Een sigma-reeks waarin HOOG bijna nooit voorkomt -- de H2-situatie."""
    rng = np.random.default_rng(2)
    sigma = pd.DataFrame({"A": np.abs(rng.normal(0.5, 0.02, 1200))}, index=IDX)
    return assign_by_variance(sigma, low_q=0.005, high_q=0.995, min_periods=100)


def _persistent_but_clustered():
    """Genoeg BARS, te weinig EPISODES: een lange trage sigma-golf levert een
    HOOG-toestand met honderden bars in een handvol blokken. Dit is het geval
    dat een bar-poort doorlaat en dat statistisch niets is."""
    wave = 0.5 + 0.4 * np.sin(np.linspace(0, 4 * np.pi, 1200))
    sigma = pd.DataFrame({"A": wave}, index=IDX)
    return assign_by_variance(sigma, low_q=0.30, high_q=0.70, min_periods=100)


def test_a_balanced_assignment_passes() -> None:
    verdict = assert_realised_occupancy(_balanced(), n_folds=4, **LIMITS)
    assert verdict.adequate


def test_a_starved_state_is_refused_on_the_MINIMUM_not_the_median() -> None:
    """AD-18: het oordeel gaat over het minimum. Een mediaan van 165 met een
    minimum van 5 is precies de meting die H2 groen liet lijken."""
    with pytest.raises(DataContractError):
        assert_realised_occupancy(
            _starved(), n_folds=4,
            min_obs_per_state_per_fold=100,
            min_episodes_per_state_per_fold=10,
            min_occupancy_fraction=0.10,
        )


def test_enough_bars_but_too_few_episodes_is_refused() -> None:
    """Q7. De test die revisie 1 niet had, en het geval dat er in de praktijk
    het vaakst is."""
    with pytest.raises(DataContractError):
        assert_realised_occupancy(
            _persistent_but_clustered(), n_folds=4,
            min_obs_per_state_per_fold=50,
            min_episodes_per_state_per_fold=25,
            min_occupancy_fraction=0.10,
        )


def test_the_verdict_reports_both_measured_minima() -> None:
    verdict = assert_realised_occupancy(
        _balanced(), n_folds=4, raise_on_failure=False, **LIMITS
    )
    assert "rarest_state_obs_in_smallest_fold" in verdict.measured
    assert "rarest_state_episodes_in_smallest_fold" in verdict.measured
    assert verdict.measured["occupancy_source"] == "realised"
```

- [ ] **Stap 9.2 — Draai en zie hem falen.**

- [ ] **Stap 9.3 — Implementeer.** Het oordeel gaat over het **minimum** over folds, niet de mediaan (AD-18), op **beide** grootheden, en `occupancy_source` is `"realised"` en niet `"uniform (1/k) — optimistisch"`.

- [ ] **Stap 9.4 — Voeg `vol_state` toe aan `conf/model/adequacy.yaml`** met `min_obs_per_state_per_fold: 100`, `min_episodes_per_state_per_fold: 20` en `min_state_occupancy_fraction: 0.10`, met een commentaarblok dat uitlegt waarom de a-priori-variant niet volstond en waarom bars alleen niet volstaan.

- [ ] **Stap 9.5 — Draai de poort op de echte toewijzing.**

```bash
D:/venv/tradebot/Scripts/python.exe apps/run_state_diagnostics.py --check-adequacy
```

> **Verwachting, en zij is ongemakkelijk.** De nulmeting toont HOOG op **4,8 %** van de bars — 499 over zes symbolen, 60 tot 117 per symbool. Over 12 folds is dat ordegrootte **5 tot 10 observaties per fold**, ruim onder de eis van 100, en het aantal episodes ligt daar nog onder. **De meest waarschijnlijke uitkomst van deze stap is dat de driedelige toestand op dit paneel niet adequaat is.**
>
> Als dat gebeurt, is dat geen mislukking maar het goedkoopste antwoord van de hele fase, en de reactie ligt vast: **verlaag het aantal toestanden naar twee** (`hmm_k2` haalde a priori 247,5) en herhaal stap 6. **Boek die tak als een tweede trial** — per stap 4.3 kost een beslisboom het aantal takken dat je doorloopt. Wat **niet** mag: `min_obs_per_state_per_fold` of `min_episodes_per_state_per_fold` verlagen tot de poort groen wordt. Die drempels zijn conventies en staan vast vóór de meting.
>
> Haalt ook de tweedelige toestand de poort niet, dan luidt het oordeel `UNPROVEN — insufficient data`, vervalt stap 13, en is dat het exit-antwoord op het hele toestandsspoor. Schrijf het op als bevinding, met de exacte getallen.

- [ ] **Stap 9.6 — Commit.**

```bash
git add src/tradebot/validation/data_adequacy.py \
        tests/unit/test_realised_occupancy_gate.py conf/model/adequacy.yaml
git commit -m "fix(adequacy): poort op GEREALISEERDE bezetting en op episodes; de a-priori-variant was optimistisch"
```

---

# STAGE C — DE DRIE HYPOTHESEN

*Zes trials, geboekt in `hypothesis_ledger.json` vóór de eerste run. `M_new = 25`. Elke hypothese leest de poortsample precies eenmaal.*

---

### Stap 10: Het gemeten verschil per laag

Voordat er een nieuwe hypothese wordt getoetst, moet vaststaan wat de bestaande lagen dóen. Dat is nu niet vastgelegd: `phase5_revaluation.json` bevat de vijf sporen los, maar niet de **verschillen** met hun onzekerheid — en juist die verschillen zijn de vraag.

Diagnostiek op de ontwikkelsample. **Nul trials.**

**Files:**
- Modify: `apps/run_phase5_revaluation.py`
- Modify: `artefacts/baseline/phase5_revaluation.json`
- Create: `tests/unit/test_layer_transitions.py`

**Interfaces:**
- Consumes: `validation/inference.py::sharpe_difference_test`, `validation/inference.py::block_bootstrap_ci`
- Produces: de sleutel `layer_transitions` in `phase5_revaluation.json`

- [ ] **Stap 10.1 — Definieer de vier overgangen.**

| Overgang | Van | Naar | De vraag |
|---|---|---|---|
| T1 | `long_only_equal_weight` | `l1_l2_only` | voegt de alfalaag iets toe aan gelijk gewicht? |
| T2 | `l1_l2_only` | `l1_l7_no_halt` | voegt de portefeuilleconstructie iets toe? |
| T3 | `l1_l7_no_halt` | `l1_l7_with_halt` | wat kost de haltketen? |
| T4 | `long_only_equal_weight` | `l1_l7_with_halt` | is het volledige systeem beter dan het triviale alternatief? |

- [ ] **Stap 10.2 — Schrijf de falende test.**

```python
# tests/unit/test_layer_transitions.py
"""Een verschil zonder interval is geen bevinding.

Revisie 1 gebruikte hier een gepaarde t-toets op RENDEMENTSverschillen. Dat is
de verkeerde toets: de sporen verschillen in VOLATILITEIT (L7 herschaalt naar
het vol-target, de haltketen zet 89 % van de bars vlak), en een t-toets op
rendementsverschillen negeert dat. De vraag is een SHARPE-verschil en die vraagt
Ledoit-Wolf (2008).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.validation.inference import block_bootstrap_ci, sharpe_difference_test

IDX = pd.date_range("2021-01-01", periods=1615, freq="D", tz="UTC")


def test_two_identical_tracks_show_no_difference() -> None:
    rng = np.random.default_rng(1)
    track = pd.Series(rng.normal(0.0002, 0.02, 1615), index=IDX)
    result = sharpe_difference_test(track, track.copy(), bars_per_year=365)
    assert abs(result.delta_sharpe) < 1e-12
    assert result.p_value > 0.99


def test_a_real_difference_is_detected_with_an_interval() -> None:
    rng = np.random.default_rng(2)
    a = pd.Series(rng.normal(0.0010, 0.02, 1615), index=IDX)
    b = pd.Series(rng.normal(0.0000, 0.02, 1615), index=IDX)
    result = sharpe_difference_test(a, b, bars_per_year=365)
    assert result.delta_sharpe > 0.0
    assert result.ci_low < result.delta_sharpe < result.ci_high


def test_the_test_accounts_for_unequal_volatility() -> None:
    """De reden waarom een gepaarde t op rendementsverschillen hier fout is:
    twee sporen met hetzelfde gemiddelde en verschillende vol hebben een
    verschillende Sharpe, en die vraag moet de toets kunnen zien."""
    rng = np.random.default_rng(3)
    a = pd.Series(rng.normal(0.0005, 0.010, 1615), index=IDX)
    b = pd.Series(a.to_numpy() * 2.0, index=IDX)  # zelfde signaal, dubbele vol
    result = sharpe_difference_test(a, b, bars_per_year=365)
    assert abs(result.delta_sharpe) < 1e-9  # Sharpe is scale-invariant
    assert result.p_value > 0.5


def test_a_halted_track_is_compared_on_common_active_bars_only() -> None:
    """De haltketen zet 1.559 van de 1.743 bars vlak. Nul-rendementen
    meerekenen levert een SE die de resterende 184 actieve bars niet
    weerspiegelt, en dus een t die te groot is."""
    rng = np.random.default_rng(4)
    a = pd.Series(rng.normal(0.0005, 0.02, 1615), index=IDX)
    b = a.copy()
    b.iloc[184:] = 0.0
    result = sharpe_difference_test(
        a, b, bars_per_year=365, align="common_active",
    )
    assert result.n_effective <= 184


def test_the_bootstrap_ci_is_reported_alongside_the_analytic_one() -> None:
    """Twee onafhankelijke routes naar hetzelfde interval. Wijken ze sterk af,
    dan is dat de bevinding en niet een detail."""
    rng = np.random.default_rng(5)
    a = pd.Series(rng.normal(0.0008, 0.02, 1615), index=IDX)
    ci = block_bootstrap_ci(a, statistic="sharpe", block_length=20,
                            n_boot=2000, bars_per_year=365, seed=11)
    assert ci.low < ci.point < ci.high
```

- [ ] **Stap 10.3 — Draai en zie hem falen.**

- [ ] **Stap 10.4 — Breid de revaluation-app uit.** Voeg per overgang een blok toe. Verplichte velden, en de laatste drie zijn nieuw in revisie 2:

```json
"layer_transitions": {
  "T1_equal_weight_to_l1_l2": {
    "delta_sharpe": null,
    "se_ledoit_wolf": null,
    "ci_95_analytic": [null, null],
    "ci_95_block_bootstrap": [null, null],
    "block_length": 20,
    "n_boot": 10000,
    "p_value": null,
    "n_bars": null,
    "n_effective": null,
    "align": "common_active",
    "sample": "development",
    "note": null
  }
}
```

- [ ] **Stap 10.5 — Draai en interpreteer.**

```bash
D:/venv/tradebot/Scripts/python.exe apps/run_phase5_revaluation.py --with-transitions
```

Verwachting op grond van de nulmeting: **T1 en T2 zullen niet significant zijn**. Het beste spoor is `long_only_equal_weight` met 0,156 en dat is zelf geen bevinding — bij `SE ≈ 0,54` op de ontwikkelsample is de t ongeveer 0,29.

**T3 is het interessante geval en het moet zorgvuldig worden gelezen.** De haltketen halteert 1.559 van 1.743 bars. `l1_l7_with_halt` haalt +0,136 op de resterende ~184 actieve bars ≈ 0,50 jaar. De t=2-hurdle op die horizon is **2,83**, dus +0,136 is geen zwak bewijs maar **geen bewijs**. Schrijf dat in `note`, met het getal.

- [ ] **Stap 10.6 — Commit.**

```bash
git add apps/run_phase5_revaluation.py artefacts/baseline/phase5_revaluation.json \
        tests/unit/test_layer_transitions.py
git commit -m "feat(baseline): laagovergangen met Ledoit-Wolf-SE, bootstrap-CI en halt-uitlijning"
```

---

### Stap 11: H-10.1 — De beslisfrequentie

**De hypothese.** *De haltketen halteert 89 % van de bars omdat hij per dag een besluit eist. Wordt het besluit met frequentie k genomen (k ∈ {1, 2, 5, 10} bars) en tussentijds vastgehouden, dan daalt het aantal haltes en de omzet zonder dat het rendement wordt vernietigd.*

**Waarom dit de eerste hypothese is.** Zij vraagt geen nieuwe voorspelling en geen nieuwe feature. Zij vraagt of het bestaande systeem op de goede tijdschaal opereert. Faalt zij, dan is dat een uitspraak over de dagbar als besluiteenheid — de meest fundamentele vraag in het meetdomein.

**Trialkosten: 4.** k ∈ {1, 2, 5, 10}. Vooraf geboekt.

**Files:**
- Create: `src/tradebot/portfolio/decision_frequency.py`
- Create: `apps/run_h10_1_decision_frequency.py`
- Create: `tests/unit/test_decision_frequency.py`
- Create: `tests/lookahead/test_decision_frequency_causality.py`
- Create: `conf/experiment/h10_1_decision_frequency.yaml`
- Create: `reports/phase10_h10_1_decision_frequency.md` (output)
- Create: `artefacts/governance/phase10_h10_1.json` (output)

**Interfaces:**
- Produces: `decision_frequency.hold_decision(exposures, *, k, anchor="first_bar") -> pd.DataFrame`; `decision_frequency.breakeven_cost_bps(delta_sharpe, turnover_delta, ...) -> float`

- [ ] **Stap 11.1 — Boek de hypothese in het grootboek.** Vier trials, `M_new` 6 → 10.

- [ ] **Stap 11.2 — Schrijf de falende tests.**

```python
# tests/unit/test_decision_frequency.py
"""Vasthouden is niet hetzelfde als resamplen.

De valkuil: `.resample('5D').first()` en dan forward-fillen. Dat kiest per
blok de eerste bar en dat is een lookahead zodra het blok wordt uitgelijnd op
iets anders dan de eerste beschikbare bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradebot.portfolio.decision_frequency import hold_decision
from tradebot.utils.failfast import DataContractError

IDX = pd.date_range("2022-01-01", periods=100, freq="D", tz="UTC")


def _exposures() -> pd.DataFrame:
    rng = np.random.default_rng(6)
    return pd.DataFrame(
        {"A": rng.uniform(-1, 1, 100), "B": rng.uniform(-1, 1, 100)}, index=IDX,
    )


def test_k_equals_one_is_the_identity() -> None:
    exposures = _exposures()
    pd.testing.assert_frame_equal(hold_decision(exposures, k=1), exposures)


def test_k_equals_five_holds_for_five_bars() -> None:
    held = hold_decision(_exposures(), k=5)
    for start in range(0, 95, 5):
        block = held.iloc[start:start + 5]
        assert (block.nunique() == 1).all()


def test_the_held_value_is_the_value_of_the_first_bar_in_the_block() -> None:
    exposures = _exposures()
    held = hold_decision(exposures, k=5)
    for start in range(0, 95, 5):
        pd.testing.assert_series_equal(
            held.iloc[start], exposures.iloc[start], check_names=False,
        )


def test_turnover_falls_roughly_as_one_over_k() -> None:
    exposures = _exposures()

    def turnover(frame: pd.DataFrame) -> float:
        return float(frame.diff().abs().sum(axis=1).mean())

    base = turnover(exposures)
    assert turnover(hold_decision(exposures, k=5)) < base / 3.0


def test_k_must_be_positive_and_integral() -> None:
    for bad in (0, -1, 2.5):
        with pytest.raises(DataContractError):
            hold_decision(_exposures(), k=bad)
```

```python
# tests/lookahead/test_decision_frequency_causality.py
"""R-1 op de vasthoudoperatie -- de test die de resample-valkuil vangt."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.portfolio.decision_frequency import hold_decision

IDX = pd.date_range("2022-01-01", periods=200, freq="D", tz="UTC")


def test_a_future_change_cannot_move_an_earlier_held_value() -> None:
    rng = np.random.default_rng(7)
    exposures = pd.DataFrame({"A": rng.uniform(-1, 1, 200)}, index=IDX)
    for k in (2, 5, 10):
        base = hold_decision(exposures, k=k)
        for cut in (50, 100, 150):
            truncated = hold_decision(exposures.iloc[:cut], k=k)
            pd.testing.assert_frame_equal(
                truncated, base.iloc[:cut], check_freq=False,
            )


def test_a_perturbation_at_bar_t_cannot_move_bars_before_t() -> None:
    rng = np.random.default_rng(8)
    exposures = pd.DataFrame({"A": rng.uniform(-1, 1, 200)}, index=IDX)
    perturbed = exposures.copy()
    perturbed.iloc[123, 0] = 99.0
    for k in (2, 5, 10):
        pd.testing.assert_frame_equal(
            hold_decision(perturbed, k=k).iloc[:123],
            hold_decision(exposures, k=k).iloc[:123],
            check_freq=False,
        )
```

- [ ] **Stap 11.3 — Draai en zie ze falen.**

- [ ] **Stap 11.4 — Implementeer `hold_decision`.** Blokken worden gelegd vanaf de **eerste bar van het paneel** en nooit op een kalendergrens die van de toekomst afhangt. De vastgehouden waarde is die van de eerste bar van het blok, en de haltketen loopt **ná** het vasthouden — een halt is een risicobesluit en dat wordt niet vastgehouden.

- [ ] **Stap 11.5 — Schrijf de app en het experiment-config.** Per k: aantal haltes, gemiddelde omzet, Sharpe met **Lo-gecorrigeerde** SE (§3.4), en het Sharpe-verschil tegen k=1 met Ledoit–Wolf. `n_trials: 4`, `M_new: 10`.

- [ ] **Stap 11.6 — Draai op de ontwikkelsample, één run per k.**

- [ ] **Stap 11.7 — Voeg de breakevenanalyse toe.** Nieuw in revisie 2 en het is de belangrijkste tabel van deze stap. De hypothese is een **kostenhypothese**, dus het antwoord hangt aan een kostenaanname (`eta = 2.991922`, ongekalibreerd, AD-2). Rapporteer daarom per k niet alleen het gemeten verschil maar ook:

  | Grootheid | Betekenis |
  |---|---|
  | `turnover_delta` | daling in gemiddelde dagelijkse omzet t.o.v. k=1 |
  | `delta_sharpe_gross` | Sharpe-verschil vóór kosten |
  | `delta_sharpe_net` | idem, met het huidige `eta` |
  | `breakeven_cost_bps` | de kosten per eenheid omzet waarbij `delta_sharpe_net` precies nul is |

  De conclusie luidt dan niet "k=5 is beter" maar **"k=5 is beter zodra de werkelijke kosten boven X bp liggen"** — een uitspraak die overleeft dat `eta` niet gekalibreerd is.

- [ ] **Stap 11.8 — Lees de poortsample precies eenmaal.** Alleen voor de k met het gunstigste resultaat op de ontwikkelsample. Registreer de lezing in `holdout_lock.json` vóór de run. Het resultaat op de poortsample is **de** uitkomst; het ontwikkelresultaat is de aanleiding.

- [ ] **Stap 11.9 — Schrijf het rapport en het verdict.** Verplichte velden: `verdict` ∈ {`CONFIRMED`, `REJECTED`, `UNPROVEN`}, `M_new`, `deflated_sharpe`, `t_hurdle_at_n`, de breakeventabel, en het poortsample-resultaat naast het ontwikkelresultaat.

  > **De beslisregel, vooraf.** `CONFIRMED` vereist alle vier: (1) `delta_sharpe > 0` op de ontwikkelsample, (2) het 95 %-CI sluit nul uit **na** deflatie met `M = 25`, (3) hetzelfde teken op de poortsample, en (4) `breakeven_cost_bps` ligt onder een plausibele kostenaanname. Ontbreekt er één, dan is het `UNPROVEN`. `REJECTED` is voorbehouden aan een aantoonbaar negatief effect.

- [ ] **Stap 11.10 — Commit.**

---

### Stap 12: H-10.2 — Het onevenwichtige paneel

**De hypothese.** *De cross-sectie is afgekapt op drie symbolen vanaf 2021-03-15 omdat de portefeuillelaag een gebalanceerd paneel eist. BTC heeft 6,41 jaar, LINK 5,83 en ETH 5,44. Draagt het systeem een onevenwichtig paneel met per-bar variërende breedte, dan groeit de sample met ordegrootte 30 % zonder nieuwe data.*

**Waarom dit vóór de derde hypothese komt.** Bij `SE = 0,476` en een t=2-hurdle van 0,951 op 4,42 jaar (en **1,081** op de 3,42-jarige ontwikkelsample) is elke Sharpe onder 1 statistisch onbeslisbaar. 30 % meer sample verlaagt de hurdle naar ongeveer **0,95** op de ontwikkelsample. Dat is geen alfa, maar het maakt de rest van de fase meetbaarder — en het is de enige stap in dit document die de hurdle verlaagt zonder een claim te doen.

**Trialkosten: 1.** Eén configuratie: `min_symbols_per_bar = 2`. Geen zoektocht over die drempel; wordt er ooit een tweede waarde geprobeerd, dan is het een tweede trial en is de eerste er retroactief ook een.

**Files:**
- Modify: `src/tradebot/portfolio/{weights.py, risk.py}`
- Modify: `src/tradebot/validation/data_adequacy.py`
- Create: `tests/unit/test_unbalanced_panel.py`
- Create: `apps/run_h10_2_unbalanced_panel.py`
- Create: `conf/experiment/h10_2_unbalanced_panel.yaml`
- Create: `reports/phase10_h10_2_unbalanced_panel.md` (output)

- [ ] **Stap 12.1 — Boek de hypothese.** Eén trial, `M_new` 10 → 11.

- [ ] **Stap 12.2 — Schrijf de falende test.**

```python
# tests/unit/test_unbalanced_panel.py
"""Het paneel mag per bar van breedte veranderen.

Vier eigenschappen: gewichten sommeren per bar over de BESCHIKBARE namen; een
NaN is afwezigheid en geen nul; de covariantieschatting krijgt alleen de
beschikbare namen; en onder de minimumbreedte is de bar vlak in plaats van
gedeeltelijk gevuld.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tradebot.portfolio.weights import normalise_weights

IDX = pd.date_range("2021-01-01", periods=300, freq="D", tz="UTC")


def _ragged() -> pd.DataFrame:
    rng = np.random.default_rng(9)
    frame = pd.DataFrame(
        {"BTC": rng.uniform(-1, 1, 300),
         "LINK": rng.uniform(-1, 1, 300),
         "ETH": rng.uniform(-1, 1, 300)},
        index=IDX,
    )
    frame.loc[IDX[:100], "ETH"] = np.nan
    frame.loc[IDX[:50], "LINK"] = np.nan
    return frame


def test_weights_sum_over_available_names_per_bar() -> None:
    weights = normalise_weights(_ragged(), min_symbols_per_bar=2)
    gross = weights.abs().sum(axis=1)
    active = weights.notna().sum(axis=1)
    assert np.allclose(gross[active >= 2], 1.0)


def test_absence_stays_nan_and_never_becomes_zero() -> None:
    ragged = _ragged()
    weights = normalise_weights(ragged, min_symbols_per_bar=2)
    assert weights[ragged.isna()].isna().all().all()


def test_a_bar_below_the_minimum_width_is_flat() -> None:
    weights = normalise_weights(_ragged(), min_symbols_per_bar=2)
    assert weights.iloc[:50].abs().sum(axis=1).eq(0.0).all()


def test_the_sample_grows_relative_to_the_balanced_panel() -> None:
    ragged = _ragged()
    balanced = ragged.dropna()
    unbalanced = normalise_weights(ragged, min_symbols_per_bar=2)
    active = unbalanced.abs().sum(axis=1) > 0
    assert int(active.sum()) > len(balanced)
```

- [ ] **Stap 12.3 t/m 12.6 — Draai, implementeer, run, rapporteer.** Rapporteer de gerealiseerde samplegroei in bars én in jaren, en de **nieuwe** t=2-hurdle bij de nieuwe N. Dat laatste getal is het eigenlijke resultaat van deze stap.

  > **Belangrijk, en dit is een valkuil.** Een langere sample maakt de hurdle lager, maar de baseline-Sharpes moeten dan **opnieuw** worden gemeten op de nieuwe sample. Een Sharpe van 0,156 gemeten op 3 symbolen × 3,42 jaar is niet vergelijkbaar met een Sharpe op 6 symbolen × 4,4 jaar. Werk `phase5_revaluation.json` bij en markeer de oude waarden als `superseded`, met de reden.

- [ ] **Stap 12.7 — Commit.**

---

### Stap 13: H-10.3 — De toestandspoort

**De hypothese.** *Een volatiliteitstoestand bepaalt of er een besluit wordt genomen. Een systeem dat in de HOOG-toestand geen exposure inneemt, behoudt het rendement van LAAG en NORMAAL tegen materieel lagere volatiliteit — en dat verschil overleeft de deflatie.*

**Waarom deze als laatste.** Zij hangt aan stap 9. Haalt de driedelige toestand de bezettings- en episodepoort niet, dan **vervalt deze stap** en is dat het antwoord.

**Trialkosten: 1.** Eén configuratie: `flat_states = {HIGH}`. Géén zoektocht over `flat_states` — elke extra combinatie is een extra trial en zou `M_new` opdrijven tot voorbij het punt waarop nog iets kan worden aangetoond.

**Files:**
- Create: `apps/run_h10_3_state_gate.py`
- Create: `conf/experiment/h10_3_state_gate.yaml`
- Create: `tests/unit/test_state_gate_experiment.py`
- Create: `reports/phase10_h10_3_state_gate.md` (output)
- Create: `artefacts/governance/phase10_h10_3.json` (output)

- [ ] **Stap 13.1 — Boek de hypothese.** Eén trial, `M_new` 11 → 12. **Boek daarnaast de tak uit stap 9.5** als die is doorlopen; de eindstand van `M_new` moet gelijk zijn aan het aantal daadwerkelijk doorlopen takken (verwacht: **25**, inclusief de marge uit stap 4.4).

- [ ] **Stap 13.2 — Leg de beslisregel vast vóór de run.** Letterlijk in het config-bestand, en dit is de belangrijkste regel van Stage C:

> `CONFIRMED` vereist **alle vijf**:
> 1. het Sharpe-verschil tegen het ongepoorte spoor is positief op de ontwikkelsample;
> 2. het 95 %-CI (Ledoit–Wolf, HAC met q = 7, plus block-bootstrap) sluit nul uit;
> 3. de **deflated Sharpe** met `M = 25` blijft positief — de drempel is `E[max SR] ≈ 0,95` bij `SE = 0,476`;
> 4. het teken houdt op de **poortsample**, gelezen precies eenmaal;
> 5. de **concentratiemeting** uit stap 7.3 blijft binnen de grens uit stap 17.
>
> Alles daarbuiten is `UNPROVEN`. Een lagere volatiliteit bij gelijk rendement is **geen** `CONFIRMED` als het Sharpe-CI nul bevat: dat is dan een sizing-effect en per AD-16 deelt L7 dat er weer uit.

- [ ] **Stap 13.3 — Schrijf de app.** Draai twee sporen op de ontwikkelsample — `l1_l7_with_halt` ongepoort, en hetzelfde spoor met `gate_by_state(flat_states={HIGH})` ingevoegd **vóór** L7 — en rapporteer het verschil met de volledige inferentiekern.

- [ ] **Stap 13.4 — Verplicht: het concentratierapport.** Q9. Voeg `concentration_report` toe aan het artefact, met per kwantiel (50/95/99) het maximale gewicht en het effectieve aantal namen, vóór en na de poort. **Stijgt het maximale gewicht boven de grens uit stap 17, dan is de hypothese `UNPROVEN` ongeacht de Sharpe** — een verbetering die uit concentratie komt, is een andere hypothese dan de hypothese die is geboekt.

- [ ] **Stap 13.5 — Draai, lees de poortsample eenmaal, rapporteer.**

  > **De eerlijke verwachting.** De HOOG-toestand draagt in de nulmeting **+136,4 %** geannualiseerd rendement — de hoogste van de drie, zij het met een gedefleerde t van 0,53. Die toestand dichtzetten verwijdert dus waarschijnlijk **rendement** samen met de volatiliteit, en het Sharpe-effect kan gemakkelijk nul of negatief zijn.
  >
  > **Dat is een geldig en waardevol antwoord.** Het zou vaststellen dat de vol-toestand op dit paneel geen handelbare informatie over de richting bevat — een uitspraak die de vier fases voorafgaand aan deze niet hebben kunnen doen, en die het toestandsspoor sluit in plaats van het open te laten. Schrijf dat op met de getallen, en sluit het spoor.

- [ ] **Stap 13.6 — Commit.**

---

# STAGE D — DE OPRUIMING

*Geen hypothesen, nul trials. Deze stage brengt de repository in overeenstemming met het mandaat en met het meetdomein uit §2.1.*

---

### Stap 14: Keten A wordt gesloten als opruiming, niet als hypothese

`docs/CHAIN_A_STATUS.md` beschrijft een dode keten: `alpha/factor_alpha.py::momentum_alpha` bestaat, maar geen actieve configuratie roept hem aan. Twee routes:

| Route | Wat het is | Trialkosten |
|---|---|---|
| (a) | keten A éénmalig **diagnostisch** draaien met de bestaande, ongewijzigde parameters — geen selectie, geen tuning, uitkomst gaat in een rapport | 0 |
| (b) | keten A **promoveren** tot kandidaat, met de 2.400 geërfde Optuna-trials (200 × 6 × 2) in de deflatie | 2.400+ |

**Route (b) is niet betaalbaar.** Bij `M = 2.400` en `SE = 0,476` ligt `E[max SR]` uit ruis rond **1,68**; de best gemeten Sharpe op dit paneel is 0,156. Er bestaat geen uitkomst van keten A die die drempel haalt.

> **En dat is precies de reden waarom route (a) *diagnostisch* moet blijven.** Zodra de uitkomst van (a) wordt gebruikt om te kiezen — ook informeel, ook alleen om te besluiten dat "keten A het niet is" en iets anders wél — is (b) alsnog betaald. Route (a) mag één ding produceren: een rapport dat vaststelt wat keten A doet, zonder gevolgtrekking naar wat er daarna gebeurt.

- [ ] **Stap 14.1 — Kies route (a) of sluit de keten definitief.** Leg de keuze en de reden vast in `docs/CHAIN_A_STATUS.md`.
- [ ] **Stap 14.2 — Bij route (a):** draai eenmalig op de **ontwikkelsample**, schrijf `reports/phase10_chain_a_diagnostic.md` met de expliciete kop *"Diagnostisch — niet promoveerbaar, geen trial geboekt"* en de volledige inferentiekern uit stap 4A.
- [ ] **Stap 14.3 — Bij sluiten:** verwijder de dode configuratiepaden, laat `momentum_alpha` staan (het is domein-conform en getest), en markeer `CHAIN_A_STATUS.md` als `CLOSED` met de reden en de trialrekening die de sluiting verklaart.
- [ ] **Stap 14.4 — Commit.**

---

### Stap 15: Het verdict per buiten-domein-module

Deze stap voert het mandaat uit §2 uit. Voor elke module die een observatie fijner dan de dagbar veronderstelt, geldt één van drie verdicts:

- **`ARCHIVED`** — verplaatst naar `archive/`, uit de testsuite, uit de coverage, met een `README.md` die de reden en de heropeningsvoorwaarde vastlegt;
- **`DEMOTED`** — blijft in `src/`, maar wordt uit elke actieve configuratie en uit elke DVC-stage verwijderd; de module is dood code met een expliciet label;
- **`RETAINED`** — blijft actief, omdat de module ondanks zijn naam volledig op de dagbar werkt.

**De volledige tabel.** Vul kolom 3 en 4 in vóór de eerste verplaatsing.

| Module / directory | Waarom hij ter discussie staat | Verdict | Heropeningsvoorwaarde |
|---|---|---|---|
| `features/microstructure.py` | naam suggereert een fijnere observatie; bevat feitelijk alleen dagelijkse funding en open interest | **`RETAINED`**, maar **verplicht hernoemen** → `features/positioning.py` | n.v.t. — de naam was de fout, niet de inhoud |
| `data/orderbook*.py`, `features/orderflow*.py` (indien aanwezig) | vereist een observatie die in §2.1 niet bestaat | `ARCHIVED` | een gecertificeerde bron in `measurement_domain.yaml` |
| `live/` (loop, executie, monitoring) | draait op een besluitcyclus die niet de dagbar is | zie stap 16 | zie stap 16 |
| `execution/impact.py` (`eta = 2.991922`) | ongekalibreerd; kalibratie vereist uitvoeringsdata die niet in het domein zit | `RETAINED` met harde limiet, zie stap 17 | AD-2 blijft van kracht |
| `volatility/realized.py` | naam suggereert intrabar-informatie; bevat alleen range-estimators op OHLC van de dagbar | **`RETAINED`, ongewijzigd** | n.v.t. — expliciet beschermd per §2.3 |
| elke andere module met een `bar_resolution` ≠ `1d` | schending van het domein | `ARCHIVED` | idem |

> **Discipline bij het invullen.** Vul de tabel op grond van **wat de module leest**, nooit op grond van zijn naam. De twee `RETAINED`-regels hierboven zijn precies de gevallen waarin de naam misleidt, en zij zijn de reden dat §2.3 bestaat. Een module archiveren die op de dagbar werkt, is even schadelijk als een module behouden die dat niet doet: het eerste vernietigt werkende code, het tweede houdt het mandaat leeg.

- [ ] **Stap 15.1 — Draai de domeinpoort uit stap 2** en gebruik haar output als kandidatenlijst.
- [ ] **Stap 15.2 — Vul de tabel in** en leg hem vast in `docs/MEASUREMENT_DOMAIN.md`.
- [ ] **Stap 15.3 — Voer de hernoeming uit** (`features/microstructure.py` → `features/positioning.py`), inclusief alle imports, configuratieverwijzingen en tests. Draai daarna de volledige suite: dit is een pure hernoeming en er mag **geen** test van gedrag veranderen.
- [ ] **Stap 15.4 — Voer de archiveringen uit.** Per module: verplaats, schrijf de `README.md`, verwijder uit `pyproject.toml` coverage-paden, en verifieer dat de suite groen blijft.
- [ ] **Stap 15.5 — Commit per verdict, niet in één keer.** Een gebundelde commit maakt een enkele foute archivering onomkeerbaar zonder de andere ook terug te draaien.

---

### Stap 16: De `live/`-laag wordt een dagelijkse runner

`live/loop.py`, `live/executor.py` en `live/monitoring.py` gaan uit van een besluitcyclus die niet de dagbar is. In het meetdomein uit §2.1 bestaat er precies één besluitmoment per dag.

**Files:**
- Create: `apps/run_daily_decision.py`
- Modify / archive: `src/tradebot/live/*`
- Modify: `docs/ARCHITECTURAL_DECISIONS.md` (AD-26)

- [ ] **Stap 16.1 — Inventariseer wat `live/` doet dat de dagelijkse runner óók moet doen.** Dit is de belangrijkste substap van stap 16, en zij gaat vóór elke verwijdering. Minimaal:
  - de **haltketen** en haar volgorde (`constraint_order`) — dit is de meest waardevolle logica in `live/` en zij mag onder geen voorwaarde verdwijnen;
  - de PIT-verificatie bij het inlezen;
  - de foutpaden en het fail-fast-gedrag;
  - de logging en de artefactschrijving.

- [ ] **Stap 16.2 — Schrijf `apps/run_daily_decision.py` (≤ 80 LOC, R-6).** Precies vier verantwoordelijkheden: lees de PIT-store tot en met gisteren; bereken de exposures voor vandaag; pas de haltketen toe in de bestaande volgorde; schrijf het besluit als artefact. Geen loop, geen wachttijden, geen sessiebeheer.

- [ ] **Stap 16.3 — Verifieer de gelijkwaardigheid.** Draai de nieuwe runner en de oude loop over dezelfde historische bar en toon aan dat de exposures en de haltbesluiten **identiek** zijn. Zonder deze test is de vervanging niet aantoonbaar gedragsneutraal.

- [ ] **Stap 16.4 — Schrijf AD-26.** Titel: *"Één besluitmoment per dag; `live/` wordt een runner"*. Waarom: §2.1 kent één bar per dag, dus een continue loop modelleert een besluitcyclus die niet bestaat. Afgewezen alternatief: `live/` behouden en de loop op 24 uur zetten — dat houdt de sessie-, reconnect- en latentiepaden in leven zonder dat iets ze test.

- [ ] **Stap 16.5 — Voer het verdict uit** en commit.

---

### Stap 17: De harde limieten voor ongekalibreerde kosten en concentratie

AD-2 stelt vast dat `eta = 2.991922` ongekalibreerd is; AD-3 stelt vast dat de kalibratie uitvoeringsdata vereist. Een parameter die niet kan worden gekalibreerd, is geen open punt maar een **permanente eigenschap**, en zij hoort een permanente begrenzing te krijgen.

Stap 7 en 13 voegen daar een tweede ongemeten risico aan toe: de toestandspoort zet namen op nul, waarna L7 de rest ophoogt (Q9).

**Files:**
- Modify: `conf/portfolio/constraints.yaml`
- Modify: `src/tradebot/portfolio/risk.py`
- Create: `tests/unit/test_notional_and_concentration_limits.py`
- Modify: `docs/ARCHITECTURAL_DECISIONS.md` (AD-2, AD-3 → permanent)

- [ ] **Stap 17.1 — Voeg twee limieten toe aan `constraints.yaml`.**
  - `max_notional_for_uncalibrated_impact` — de notional waarboven de ongekalibreerde impactfunctie niet mag worden toegepast;
  - `max_weight_per_symbol` — de harde bovengrens op één naam, die de concentratie uit stap 13.4 begrenst.

  Elke waarde krijgt een commentaarblok met de reden, de AD-verwijzing en de voorwaarde waaronder zij mag worden verhoogd (namelijk: een gekalibreerde `eta`, respectievelijk een gemeten concentratie-effect).

- [ ] **Stap 17.2 — Dwing beide af in de haltketen.** Voeg ze toe aan `constraint_order` op een expliciete positie en leg in het commentaar vast **waarom** die positie: de notionallimiet vóór de impactberekening (anders wordt een ongeldige functie geëvalueerd), de gewichtslimiet ná de toestandspoort en vóór het vol-target (anders herstelt L7 de concentratie die de limiet net wegnam).

- [ ] **Stap 17.3 — Schrijf de tests.** Vier: een notional onder de limiet passeert; een notional erboven wordt begrensd en niet stilzwijgend doorgelaten; een gepoort boek waarin één naam boven `max_weight_per_symbol` uitkomt, wordt teruggeschaald; en de volgorde in `constraint_order` is precies zoals in 17.2 vastgelegd.

- [ ] **Stap 17.4 — Werk AD-2 en AD-3 bij** naar `PERMANENT` met een expliciete `Consequence`-sectie: welke uitspraken over kosten in dit systeem **niet** kunnen worden gedaan zolang `eta` ongekalibreerd is. Verwijs naar de breakevenanalyse uit stap 11.7 als de manier waaróp er dan nog over kosten kan worden gesproken.

- [ ] **Stap 17.5 — Commit.**

---

# EXIT-CRITERIA

De fase is af wanneer **alle** onderstaande regels waar zijn. Dit is geen samenvatting maar de afvinklijst; een regel die niet waar is, blokkeert de afsluiting.

| # | Criterium | Verificatie |
|---|---|---|
| 1 | Het mandaat staat in `docs/MANDATE.md` en AD-22/23/24 zijn geschreven | stap 1 |
| 2 | `docs/MEASUREMENT_CONTRACT.md` bestaat en definieert één sample-venster, één `bars_per_year`, één `N` | stap 1 (Q10) |
| 3 | Alle vensterverwijzingen in alle artefacten zijn met dat contract verzoend; afwijkingen zijn gecorrigeerd of gelabeld | stap 1 (Q10) |
| 4 | De xfail-conversie is uitgevoerd; de suite meldt **0 failed / 4 xfailed / 0 xpassed** | stap 1 (Q11) |
| 5 | `conf/governance/measurement_domain.yaml` bestaat en somt exact drie gecertificeerde bronnen op | stap 2 |
| 6 | `scripts/check_domain_consistency.py` draait in CI en faalt op elke `bar_resolution ≠ 1d` | stap 2 |
| 7 | De negatieve controle op die poort is aanwezig en bewijst dat de poort rood kan worden | stap 2 |
| 8 | `hypothesis_ledger.json` is gereset, met R1–R8, `M_archived = 2776` en `M_new` geboekt vóór de eerste run | stap 3 |
| 9 | `validation/inference.py` bestaat en is de **enige** implementatie van Sharpe-SE, DSR en Sharpe-verschil | stap 4A (Q2, Q3, Q5) |
| 10 | `deflated_sharpe` heeft een expliciete signatuur met `variance_of_sharpes`, `skew` en `kurtosis` | stap 4A (Q5) |
| 11 | Elke gerapporteerde paneel-t is naast de gepoolde waarde óók geclusterd op datum en gedefleerd met N_eff | stap 4A, 6 (Q4) |
| 12 | De poortsample is bevroren in `holdout_lock.json` en per hypothese ten hoogste eenmaal gelezen | stap 4B (Q6) |
| 13 | `regime/state.py` bestaat; `VolState` is discreet, geordend op variantie, met NaN in de opstartfase | stap 5 |
| 14 | De toestand van bar `t` gebruikt uitsluitend data ≤ `t−1`, **inclusief de kwantieldrempels** | stap 5 (Q1) |
| 15 | De causaliteitstest vangt een spike op de eigen bar; de negatieve controle met `lag=0` faalt aantoonbaar | stap 5 (Q1) |
| 16 | De toestandsdiagnose rapporteert beide assen, met episodes en met alle drie de t-lezingen | stap 6 |
| 17 | `gate_by_state` overleeft vol-targeting; de uniforme-multiplier-controle faalt op dezelfde test | stap 7 |
| 18 | AD-25 is geschreven, met REGEL V in de gepreciseerde vorm | stap 7 (Q8) |
| 19 | De concentratie na de poort is gemeten en begrensd door `max_weight_per_symbol` | stap 7, 13, 17 (Q9) |
| 20 | De EWMA/GARCH-vergelijking is proxyvrij, met Cohens κ en een vooraf vastgelegde beslisregel | stap 8 |
| 21 | De adequaatheidspoort meet **gerealiseerde** bezetting **en** episodes, op het minimum over folds | stap 9 (Q7) |
| 22 | De laagovergangen zijn gemeten met Ledoit–Wolf, block-bootstrap en halt-uitlijning | stap 10 (Q3) |
| 23 | Elke hypothese heeft een `verdict`, een `M_new`, een `deflated_sharpe` en een `t_hurdle_at_n` | stap 11–13 |
| 24 | H-10.1 rapporteert een breakevenkostenanalyse naast het gemeten verschil | stap 11.7 |
| 25 | Elke alfaclaim is voorafgegaan door de marktbèta-attributie uit §3.7, met de residuele t | §3.7 |
| 26 | Elke buiten-domein-module heeft een verdict; de tabel in `docs/MEASUREMENT_DOMAIN.md` is volledig | stap 15 |
| 27 | `features/microstructure.py` is hernoemd naar `features/positioning.py`, gedragsneutraal | stap 15.3 |
| 28 | De dagelijkse runner bestaat, is gedragsequivalent aan de oude loop, en AD-26 is geschreven | stap 16 |
| 29 | De haltketen en `constraint_order` zijn ongewijzigd behalve de twee nieuwe limieten op hun vastgelegde positie | stap 16.1, 17.2 |
| 30 | AD-2 en AD-3 zijn `PERMANENT` met een `Consequence`-sectie | stap 17 |
| 31 | De volledige suite is groen op het fingerprint uit regel 4; alle 628 lookahead-tests slagen | continu |
| 32 | `ruff` 0.15.12 en `mypy --strict` zijn schoon; geen file > 800 LOC; geen app > 80 LOC | DI-16, R-4, R-6 |

---

# REGELS & HANDELINGSINSTRUCTIES

**R-1 — Causaliteit is een test, geen intentie.** Elke transformatie die een tijdreeks aanraakt, krijgt een truncatietest **en** een perturbatietest. Truncatie-invariantie alléén is niet voldoende: zij mist een gelijktijdige toewijzing (Q1). Elke nieuwe test in `tests/lookahead/` heeft een negatieve controle die bewijst dat de test rood kan worden.

**R-2 — Elke parameter kost een trial, en elke tak van een beslisboom ook.** Registreer vóór de run. Een parameter die op grond van een uitkomst wordt gewijzigd, kost retroactief ook een trial voor zijn voorganger. Zie stap 4.3.

**R-3 — Eén implementatie per statistische grootheid.** Sharpe-SE, DSR en Sharpe-verschil komen uitsluitend uit `validation/inference.py`. Een tweede implementatie is een defect, ook als zij hetzelfde getal geeft.

**R-4 — Geen bestand boven 800 LOC.** Splits op verantwoordelijkheid, niet op regelaantal.

**R-5 — Fail fast, met een reden.** Elke `require(...)` krijgt een boodschap die uitlegt welke aanname is geschonden en waarom die aanname bestaat.

**R-6 — Apps blijven onder 80 LOC.** Een app is een compositie, geen logica.

**R-7 — De poortsample wordt per hypothese ten hoogste eenmaal gelezen.** Elke lezing wordt vóór de run geregistreerd in `holdout_lock.json`. Een tweede lezing maakt de sample tot ontwikkeldata en dan is er geen poort meer.

**R-8 — Een getal zonder onzekerheid is geen bevinding.** Elke gerapporteerde Sharpe, elk verschil en elk conditioneel gemiddelde krijgt een SE, een interval en — op paneeldata — de geclusterde en gedefleerde lezing ernaast.

**R-9 — Alles in dit document is Nederlands; alle code, docstrings, commits en artefactsleutels blijven Engels.** Docstrings mogen Nederlands zijn waar zij een besluit uitleggen in plaats van een interface.

**R-10 — Een verwachting in dit document is geen resultaat.** Waar hierboven "verwachting" staat, staat een voorspelling die door de meting kan worden weerlegd. Wijkt de meting af, dan is de meting het antwoord en wordt de verwachting geciteerd als de weerlegde voorspelling die zij was.

**R-11 — TDD, zonder uitzondering.** Test eerst, zie hem falen met de verwachte fout, implementeer, zie hem slagen, commit. Een test die bij de eerste run slaagt, test niet wat je denkt.

**R-12 — Commit per stap, niet per stage.** Een gebundelde commit maakt één foute substap onomkeerbaar zonder de rest terug te draaien.

---

# STARTINSTRUCTIE

Begin bij **stap 1**. Werk de stappen in volgorde af; Stage A is een voorwaarde voor Stage B, en Stage B voor Stage C.

Vóór de eerste regel code, drie handelingen in deze volgorde:

1. **Lees `docs/ARCHITECTURAL_DECISIONS.md` (AD-1 t/m AD-21) en `artefacts/governance/hypothesis_ledger.json` volledig.** Alles in dit document veronderstelt beide.
2. **Draai de volledige suite en leg het fingerprint vast** — het aantal passed, failed, xfailed en xpassed vóór enige wijziging. Zonder dat startpunt is niet aantoonbaar dat stap 1's xfail-conversie gedragsneutraal was.
3. **Stel het venster vast en schrijf `docs/MEASUREMENT_CONTRACT.md`** vóórdat er één artefact wordt aangeraakt. Q10 is de fout die alle andere getallen in dit document vervuilt; hij wordt eerst gerepareerd.

Rapporteer na elke stap: welke test faalde, met welke fout, wat de implementatie werd, welke test slaagde, en de commit-SHA. Rapporteer na elke stage de stand van `M_new` en de resterende exit-criteria.

Sla geen stap over omdat de uitkomst voorspelbaar lijkt. De verwachtingen in dit document zijn expliciet gemaakt zodat zij kunnen worden weerlegd — niet zodat zij de meting kunnen vervangen.
