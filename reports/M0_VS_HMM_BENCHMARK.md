# H2 — HET M2 FILTERED HMM TEGEN M0 CAUSAL VOL-BUCKETS

> **Deliverable 18** · Phase 6 stap 11 · Phase 7/8 Stage C-3
> **git_sha:** `648ea74`
> **Pre-registratie:** `3d3af28730a6c7f9da48d13139522a05` (bevroren 2026-08-25T17:52:22.282373+00:00, M bij bevriezing = 2764)
> **Universum:** 6 gecertificeerde reeksen · 1743 bars · 2021-11-15 t/m 2026-08-23
> **Primaire track:** `xs_momentum_risk_parity` · 1200 OOS-bars · embargo 5 bars
> **Kostenlabels:** `IMPACT_UNCALIBRATED` · `SPREAD_ASSUMED` (1.0 bp half-spread) · risk `config_hash` `1b60cb664fbf9a2a`

## 0. De uitkomst, eerst

**Geen enkele conditioneerder is gepromoveerd. M0 Causal Vol-Buckets blijft de productie-baseline.**

| Oordeel | Aantal | Betekenis |
|---|---|---|
| PROMOTED | 0 | superieure netto OOS Sharpe na kosten, alle poorten open |
| FALSIFIED | 0 | aantoonbaar slechter op economische gronden — turnover of spread-afhankelijkheid, niet de Sharpe-toets |
| UNPROVEN | 0 | geen verbetering, en de toets kon het verwachte effect niet zien |
| DESCOPED | 6 | niet beoordeelbaar: de zeldzaamste toestand haalt de adequaatheidspoort niet |

Welk stop-criterium waar bond:

| Stop-criterium | Conditioneerders |
|---|---|
| `state_occupancy_below_adequacy` | 6 |

## 1. Wat vooraf vastlag

| | |
|---|---|
| Nulhypothese | de netto OOS Sharpe onder M2-conditionering is ≤ die onder M0, na kosten |
| Primaire maat | `net_oos_sharpe_after_costs` |
| Toets | Jobson-Korkie met Memmel-correctie op het gepaarde Sharpe-verschil |
| Geplande trials | 6 (M1, M2-gaussian en M2-student_t, elk met k = 2 en k = 3) |
| Verwacht effect | 0.08 geannualiseerde Sharpe-eenheden |
| Baseline-Sharpe vooraf | -0.3314 (`phase5_revaluation.json :: xs_momentum_risk_parity / L3_execution`) |

De pre-registratie noteerde het oordeel over haar eigen power VOOR de run, en dat is de belangrijkste zin erin:

| ρ tussen de armen | MDE (Sharpe-eenheden) | informatief? |
|---|---|---|
| 0.80 | 0.833 | nee |
| 0.90 | 0.589 | nee |
| 0.95 | 0.417 | nee |
| 0.99 | 0.186 | nee |

De correlatie die nodig zou zijn om 0,08 te zien is ρ = 0,99816 — en bij die correlatie verandert de overlay de returnreeks zo weinig dat een effect van 0,08 er niet KAN zijn. De opzet is intern tegenstrijdig, en dat stond er voordat er iets was gemeten.

## 2. Wat de architectuur al beperkt: dit is een tilt, geen risicoreductie

De soevereine laag schaalt het hele boek met `w_t = min(max_leverage, σ_target / σ_boek)`. Vermenigvuldig elke exposure met dezelfde `c`, dan deelt `w_t` er weer door: **een regime-overlay kan dit boek niet de-grossen.** Wat overblijft is de asymmetrie TUSSEN symbolen — elk symbool heeft zijn eigen regime — en dat maakt van H2 een cross-sectionele tilt: haal het risicobudget weg bij wat nu onrustig is en geef het aan de rest.

Gemeten op de M0-arm tegen de ongeconditioneerde arm:

| Grootheid | M0 / ongeconditioneerd |
|---|---|
| bruto notional | 0.9763 |
| turnover | 1.1404 |
| bars met een positie | 0.9942 |

Een bruto notional die vrijwel gelijk blijft terwijl de turnover stijgt, is precies wat een tilt doet: het risicobudget krimpt niet, het verhuist — en verhuizen kost fees. Elke zin in dit rapport die 'risicoreductie' zou suggereren, zou onjuist zijn.

## 3. De baseline zelf, en wat zij kost

| Arm | netto OOS Sharpe | turnover (notional) | fees | bars met positie | bars met fill |
|---|---|---|---|---|---|
| ongeconditioneerd | -0.0955 | 2.699.004 | 1484.46 | 1.200 | 1.200 |
| M0 vol-buckets | -0.2501 | 3.077.931 | 1692.87 | 1.193 | 1.195 |

**M0 zelf verbetert de baseline niet: de delta tegen géén overlay is -0.1546 Sharpe-eenheden** (gepaard: z = -0.582, p = 0.7197, ρ = 0.8840). Dat is geen detail voor de vergelijking die volgt: de uitdagers worden tegen M0 gemeten omdat de pre-registratie dat voorschrijft, maar een uitdager die M0 verslaat, hoeft daarmee de ONGECONDITIONEERDE baseline nog niet te verslaan. Beide getallen staan daarom in §5.

Regimewisselingen van M0 op de gescoorde bars, per symbool:

| Symbool | wisselingen | gemiddelde regimeduur (bars) |
|---|---|---|
| BTCUSDT | 153 | 7.8 |
| ETHUSDT | 116 | 10.3 |
| SOLUSDT | 108 | 11.0 |
| AVAXUSDT | 118 | 10.1 |
| LINKUSDT | 136 | 8.8 |
| DOTUSDT | 108 | 11.0 |

## 4. De adequaatheidspoort — en waarom hij hier bindt

Stop-criterium 1 van de pre-registratie meet de **filtered** bezetting van de zeldzaamste toestand per fold, op het trainvenster waarop de parameters zijn geschat. Zakt die onder 100 observaties, dan schat het model daar een gemiddelde, een schaal en k−1 overgangskansen op enkele tientallen punten, en is die toestand een uitschieterdetector in plaats van een regime.

De a-priori poort in `conf/model/adequacy.yaml` gebruikt de UNIFORME aanname (1/k per toestand) en noemt zichzelf daarbij optimistisch. Op 495 trainbars geeft dat 247,5 observaties bij k = 2 — ruim boven de eis. De gemeten filtered bezetting is een ander getal:

| Conditioneerder | min | mediaan | fits onder de poort | trainbars per fold |
|---|---|---|---|---|
| `m1-k2` | 5.00 | 45.00 | 66/72 | 495 |
| `m1-k3` | 5.00 | 45.00 | 66/72 | 495 |
| `hmm2-diag-gaussian` | 15.72 | 310.60 | 14/72 | 495 |
| `hmm3-diag-gaussian` | 9.53 | 161.98 | 22/72 | 495 |
| `hmm2-diag-student_t` | 27.97 | 377.28 | 2/72 | 495 |
| `hmm3-diag-student_t` | 15.44 | 143.99 | 26/72 | 495 |

**M1 zakt het diepst, en de oorzaak is de M0-bucket zelf.** Zijn toestanden ZIJN de buckets, en die bestaan pas na `zscore_min_periods` bars: vóór die grens is het regime ONGEDEFINIEERD en niet 'normaal bij gebrek aan beter'. Het trainvenster van de vroegste fold telt daardoor 187 bruikbare bars in plaats van 1.287, en het zeldzaamste regime (HOOG) heeft daar vijf waarnemingen. Een overgangskans uit vijf overgangen is geen schatting.

Het oordeel gaat over het MINIMUM en niet over de mediaan. Een walk-forward-evaluatie is alleen zinnig wanneer elke fold erin geldig is; een Sharpe over 1.200 OOS-bars waarvan een deel uit een model komt dat daar niet gefit had mogen worden, is geen schoon getal. De mediaan en het aantal fits onder de poort staan er wél bij, want zij bepalen of het oordeel 'dit model kan hier niet' luidt of 'deze folds konden niet'.

## 5. Door de authoritative engine, met volledige kosten

| Conditioneerder | netto Sharpe | Δ vs. M0 | Δ vs. geen overlay | turnover-ratio | Δ fees | bars met positie | oordeel |
|---|---|---|---|---|---|---|---|
| `m1-k2` | -0.3192 | -0.0691 | -0.2237 | 0.954 | -77.95 | 1.199 | DESCOPED |
| `m1-k3` | -0.3211 | -0.0710 | -0.2256 | 0.954 | -77.95 | 1.199 | DESCOPED |
| `hmm2-diag-gaussian` | -0.1853 | 0.0648 | -0.0898 | 1.372 | 629.64 | 1.200 | DESCOPED |
| `hmm3-diag-gaussian` | -0.2030 | 0.0471 | -0.1075 | 1.139 | 235.57 | 1.200 | DESCOPED |
| `hmm2-diag-student_t` | -0.4094 | -0.1594 | -0.3140 | 0.684 | -534.63 | 694 | DESCOPED |
| `hmm3-diag-student_t` | -0.0959 | 0.1542 | -0.0004 | 1.049 | 82.44 | 1.200 | DESCOPED |

Alle armen draaien door dezelfde `EventDrivenEngine`, met dezelfde risicolaag, dezelfde router, dezelfde venue en dezelfde kosten. De ENIGE ingang die verschilt is de exposure: de basisexposure maal de regimefactor. Een verschil in uitkomst kan dus nergens anders vandaan komen.

**`hmm2-diag-student_t` houdt op 694 van de 1.200 gescoorde bars nog een positie.** Zijn factor is zo vaak zo klein dat de resterende order onder de `min_notional` van de venue valt en niet wordt geplaatst. Zijn Sharpe staat dus op aanzienlijk minder bars dan die van de andere armen en is daarmee niet één op één vergelijkbaar — precies het onderscheid dat §0.5 van de fase-opdracht uitvraagt.

Per conditioneerder, de regimediagnostiek op de gescoorde bars:

| Conditioneerder | gem. factor | gem. absolute factorverandering per bar | wisselingen (mediaan over symbolen) | gem. regimeduur |
|---|---|---|---|---|
| `m1-k2` | 0.9442 | 0.0244 | 37 | 31.8 |
| `m1-k3` | 0.9421 | 0.0261 | 116 | 10.3 |
| `hmm2-diag-gaussian` | 0.7686 | 0.0923 | 114 | 10.5 |
| `hmm3-diag-gaussian` | 0.8653 | 0.0516 | 240 | 5.5 |
| `hmm2-diag-student_t` | 0.6811 | 0.0675 | 86 | 13.8 |
| `hmm3-diag-student_t` | 0.8089 | 0.0475 | 182 | 6.9 |

## 6. De gepaarde toets, en de controle die haar geldig maakt

| Conditioneerder | Sharpe uitdager | Sharpe M0 | Δ | z | p (eenzijdig) | ρ |
|---|---|---|---|---|---|---|
| `m1-k2` | -0.3192 | -0.2501 | -0.0691 | -0.358 | 0.6398 | 0.9387 |
| `m1-k3` | -0.3211 | -0.2501 | -0.0710 | -0.368 | 0.6435 | 0.9387 |
| `hmm2-diag-gaussian` | -0.1853 | -0.2501 | 0.0648 | 0.191 | 0.4241 | 0.8116 |
| `hmm3-diag-gaussian` | -0.2030 | -0.2501 | 0.0471 | 0.167 | 0.4336 | 0.8699 |
| `hmm2-diag-student_t` | -0.4094 | -0.2501 | -0.1594 | -0.354 | 0.6382 | 0.6660 |
| `hmm3-diag-student_t` | -0.0959 | -0.2501 | 0.1542 | 0.482 | 0.3148 | 0.8321 |

**De negatieve controles.** Exit-criterium 12 eist er een per statistische toets; voor deze toets zijn het er twee, en zij meten verschillende dingen. Beide draaien op een GEPAARDE stationaire block-bootstrap van de echte returnreeksen, zodat de autocorrelatie, de staarten en de onderlinge correlatie meegaan.

| Controle | gemeten | eis | uitkomst |
|---|---|---|---|
| size (twee gelijke Sharpes) | 0.048 | ≤ 0.10 | GESLAAGD |
| power bij het verwachte effect (0.08) | 0.106 | ≥ 0.80 | GEFAALD |
| power bij de MDE (0.417) | 0.716 | ≥ 0.80 | ijkpunt |

De toets houdt zijn niveau — hij verwerpt 0.048 van de tijd op een nul waar het verschil per constructie nul is — en hij ziet het effect waarop de power-analyse hem ijkt. Wat hij NIET ziet is het effect waar de hypothese over gaat: bij 0.08 Sharpe-eenheden verwerpt hij 0.106 van de tijd, tegen een doel van 0.80.

De power bij de MDE ligt onder het doel, en dat is geen tegenspraak: die MDE komt uit het ρ = 0,95-scenario van de pre-registratie, terwijl de GEMETEN correlatie tussen de armen 0.666 tot 0.939 is. Bij een lagere correlatie is het gepaarde verschil ruiziger en is dezelfde MDE minder goed te zien. Het ijkpunt bevestigt dus wat het hoort te bevestigen: de toets werkt, en zijn resolutie ligt bij tienden van een Sharpe-eenheid.

Dat is de GEMETEN versie van wat de pre-registratie analytisch al vaststelde, en het is de reden dat een niet-significante uitslag hier `UNPROVEN` heet en geen falsificatie draagt (no-go 8).

## 7. Spread-sensitiviteit

De spread is `SPREAD_ASSUMED` op 1.0 bp; Corwin-Schultz is op deze data verworpen (32,6–65,9 bp, 33–38 % negatief, AD-3). De pre-registratie eist daarom: bij welke aangenomen half-spread verdwijnt een gemeten verbetering? Verdwijnt zij al bij 3 bp, dan is de promotie een spread-aanname en geen modelresultaat.

Netto OOS Sharpe van de M0-arm per aangenomen half-spread:

| half-spread (bp) | 1.0 | 2.0 | 3.0 | 5.0 | 10.0 |
|---|---|---|---|---|---|
| M0 netto Sharpe | -0.2501 | -0.3655 | -0.3494 | -0.4090 | -0.3149 |

Δ Sharpe van elke uitdager tegen M0, per half-spread:

| Conditioneerder | 1.0 bp | 2.0 bp | 3.0 bp | 5.0 bp | 10.0 bp | winst verdwijnt bij |
|---|---|---|---|---|---|---|
| `m1-k2` | -0.0691 | 0.0088 | 0.0015 | 0.0817 | 0.0098 | n.v.t. — geen winst bij de basis-spread |
| `m1-k3` | -0.0710 | 0.0015 | 0.0089 | 0.0787 | -0.0038 | n.v.t. — geen winst bij de basis-spread |
| `hmm2-diag-gaussian` | 0.0648 | 0.1293 | 0.0386 | 0.1023 | 0.0309 | n.v.t. — geen winst bij de basis-spread |
| `hmm3-diag-gaussian` | 0.0471 | 0.0153 | -0.0200 | 0.0926 | -0.0083 | 3.0 bp |
| `hmm2-diag-student_t` | -0.1594 | -0.0241 | -0.0413 | 0.0397 | -0.0114 | n.v.t. — geen winst bij de basis-spread |
| `hmm3-diag-student_t` | 0.1542 | 0.2305 | 0.1753 | 0.1113 | -0.0132 | 10.0 bp |

**De curve is niet monotoon, en dat beperkt hoe scherp dit criterium te lezen is.** Een bredere spread kost per constructie meer, dus de netto Sharpe zou moeten dalen. Hij doet dat niet: de M0-arm loopt van -0.2501 naar -0.3655 naar -0.3494 naar -0.4090 naar -0.3149. De oorzaak is padafhankelijkheid in de soevereine laag — de drawdown-breaker is een eenrichtingsdeur, en een klein kostenverschil bepaalt of hij op een bepaalde bar afgaat. 'De half-spread waarbij de winst verdwijnt' is daarmee een GROVE aanwijzing en geen scherpe drempel; een waarde van 3 bp uit deze tabel draagt niet het gewicht dat een monotone curve eraan zou geven. Het criterium bindt hier nergens, en dat oordeel hangt niet van deze fijnstructuur af.

`n.v.t.` betekent hier niet-van-toepassing en niet niet-geschonden: zonder gemeten winst is er niets dat bij 3 bp kan verdwijnen. Het criterium kan dan per constructie niet binden, en het als geschonden boeken zou elke arm zonder verbetering ook nog een spread-falsificatie geven.

## 8. Convergentie, gedegenereerde fits en toestandsscheiding

| Conditioneerder | convergentieratio | toestandsvariantie-ratio (min / mediaan) |
|---|---|---|
| `m1-k2` | 1.00 | — |
| `m1-k3` | 1.00 | — |
| `hmm2-diag-gaussian` | 1.00 | 0.0369 / 0.1687 |
| `hmm3-diag-gaussian` | 1.00 | 0.0094 / 0.0782 |
| `hmm2-diag-student_t` | 0.92 | 0.0393 / 0.2607 |
| `hmm3-diag-student_t` | 0.56 | 0.0042 / 0.0644 |

De toestandsvariantie-ratio is `min(var) / max(var)` over de toestanden. Ligt hij rond 1, dan zijn twee toestanden hetzelfde regime met twee namen en is `k` te groot voor deze data. Het is een DIAGNOSE en geen poort — de poort is de bezetting uit §4, en die staat in de pre-registratie.

Een convergentieratio onder 1 telt de fits waarin de EM is gestopt voordat zij convergeerde, inclusief de Student-t-fits die op een instortende toestand zijn teruggerold (`degenerate`). Dat is een geregistreerd resultaat en geen weggevangen fout.

## 9. De ledger en `M`

| | |
|---|---|
| `M` vóór deze run | 2776 |
| `M` ná deze run | 2776 |
| trials geboekt bij het bevriezen | 6 |
| trials in deze run gedraaid | 6 |

De zes trials zijn bij het BEVRIEZEN van de pre-registratie geboekt — wie een parameterruimte vastlegt, heeft die kansen genomen. Deze run voert ze uit en boekt ze dus niet opnieuw; het oordeel gaat als amendement terug de ledger in, met `n_trials = 0`.

M0 telt niet mee: nul latente toestanden, nul geschatte parameters, drempels uit `conf/model/regime.yaml` die vóór de meting vastlagen. De ongeconditioneerde arm telt evenmin mee — die is de Phase 5-baseline zelf.

## 10. Wat hiermee NIET is getoetst

1. **De vraag of een regime-overlay het risico kan verlagen.** Dat kan dit boek per constructie niet meten: de soevereine vol-target herschaalt elke uniforme reductie weg (§2). Wat is gemeten is de cross-sectionele tilt.
2. **Een effect van de orde 0,08 Sharpe-eenheden.** De gemeten power daar is 0.106 (§6). Deze opzet kan dat effect niet zien, en dat lag vóór de run vast.
3. **De conditioneerders waarvan de bezetting de poort niet haalde.** Zij zijn `UNPROVEN — insufficient data` met de gemeten bezetting erbij, en NIET gefalsificeerd (no-go 8). Wat daar ontbreekt is data, geen model: op 495 trainbars per fold is een toestand met minder dan 100 verwachte observaties niet te schatten.
4. **M3 Markov-Switching GARCH.** Staat expliciet buiten deze fase (§2 van de fase-opdracht) en is niet gefit.
5. **Andere primaire signaaltracks.** `long_only_equal_weight` halteert op 2022-05-10 en handelt ~130 van 1.743 bars; een regime-experiment daarop meet de eerste zes maanden en daarna niets. De keuze voor `xs_momentum_risk_parity` stond in de pre-registratie.

Elke promotieclaim in dit rapport draagt de labels `IMPACT_UNCALIBRATED` en `SPREAD_ASSUMED` (1.0 bp). Er zijn geen promotieclaims.

