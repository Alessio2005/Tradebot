# Trend/carry-onderzoek op zes perps — rapport (2026-10-03)

## Conclusie

**Het doel ("een tevreden Sharpe en een winstgevende strategie") is met deze data niet gehaald, en het is er
ook niet aantoonbaar te halen.** Geen enkele kandidaat haalt de vooraf vastgelegde poorten A1–A6
(`PREREG.md` §3); de 60-bar-holdout is bewust ongelezen gelaten, omdat geen kandidaat hem verdient.

Wat wel waar is:

* Trend op deze zes munten is **licht winstgevend** over 2022–2026 (de bevroren `TREND_MIX`: +18 % per jaar bij
  21 % vol, vier van de vijf kalenderjaren positief, max drawdown −23 %).
* De **eerlijke** Sharpe-schatting is echter **0,3–0,5**, niet 0,85. Het gekozen lookback-ensemble staat
  bovenaan de verdeling van 54 redelijke specificaties (mediaan 0,45), en in het ongeziene venster 2024+
  zakt het van 0,85 naar 0,50 (mediaan van de familie: 0,26).
* Met 4,5 jaar dagdata op zes sterk gecorreleerde munten is de standaardfout van een Sharpe ≈ 0,5. Zelfs
  een *echte* Sharpe van 1,0 is dan pas na ≈ 9 jaar met 80 % power te bevestigen (≈ 4 jaar voor 50 %). Elke
  claim "Sharpe ≥ 1" uit deze data zou dus overfit zijn, hoe mooi het getal ook uit een zoektocht komt.

## Wat er gedaan is

| Stap | Wat | Uitkomst |
| --- | --- | --- |
| 0 | Engine herbouwd, T1/T2 uit de upload exact gereproduceerd | 0,99/0,64 en 1,20/0,74: identiek |
| A | Diagnose van de bestaande trend | 7-daagse lookback sterft bij 1 dag vertraging (−0,02); marktfactor is op dagbasis een random walk (variance ratios ≈ 1) |
| C | Screen van 17 voorspellers op discovery (≤ 2023) | **niets op \|t\| ≥ 3**; alleen zwakke aanwijzingen: XS-momentum 28–112 d (IC +0,03) en funding-crowding (IC −0,03) |
| C | Sleeves op discovery | `FUND_XS` faalt (−0,12), Donchian (0,11) en gesmoord signaal (0,65) vallen af; trend + XS-momentum lijkt goed (1,38) |
| D | **Eén** lezing van het verzegelde validatievenster (2024+) | `COMBO_TX` 0,27, `XS_MOM` −0,27, `TREND_MIX` 0,50: de bevroren kandidaat faalt |
| E | Multiversum van 54 trendspecs, niets geselecteerd | dev-mediaan 0,45, val-mediaan 0,26; gemiddeld ensemble 0,55 / 0,29 |
| F | Weekdag-effect en "veelvouden van 7" | geen effect (alle \|t\| < 1,2); verklaring verworpen |
| G | Volatiliteitsforecast | HAR verslaat EWMA(0,94) in alle vijf jaren: gemiddeld −6 % QLIKE (−2 %…−11 %) |
| H | Kitchen-sink ridge, walk-forward per kwartaal | OOS rang-IC −0,003; Sharpe −0,31 (TS) en −0,60 (XS): ML voegt niets toe |

Boekhouding: **M = 47** (17 vooraf + 30 nieuw van het budget van 40; zie `trials.jsonl`).

## De kandidaten tegen de poorten

Gemeten op 2022-01-01 → 2026-06-23, 20 % vol-doel, 6,5 bps, funding inbegrepen
(`final_gates.py`, `results/final_gates.json`).

| Poort | `TREND_MIX` (bevroren) | `COMBO_TX` (bevroren) | `TREND_MULTIVERSE_AVG` |
| --- | --- | --- | --- |
| A1 netto Sharpe ≥ 1,0 | 0,85 ✗ | 0,89 ✗ | 0,55 ✗ |
| A2 ≥ 0,7 in 22–23 én 24+ | 1,26 / 0,50 ✗ | 1,63 / 0,27 ✗ | 0,84 / 0,29 ✗ |
| A3 ≥ 4/5 jaren positief, geen < −10 % | ✓ | 2026H1 −10,6 % ✗ | ✓ |
| A4 max drawdown ≤ 25 % | −23,1 % ✓ | −31,3 % ✗ | −28,0 % ✗ |
| A5 2× kosten ≥ 0,8 én lag 2 ≥ 0,6 | 0,71 / 0,50 ✗ | 0,71 / 0,67 ✗ | 0,41 / 0,29 ✗ |
| A6 ondergrens 95 %-CI > 0 | −0,08 ✗ | −0,02 ✗ | −0,42 ✗ |
| DSR bij M = 47 (rapportage) | 0,32 | 0,36 | 0,14 |

## Wat de bevindingen betekenen

* **De ML-route is dicht, nu van twee kanten.** De wekelijkse meta-label-pijplijn gaf AUC 0,49–0,51
  (`inputs/repro_auc.json`); een lineaire ridge op alle 15 voorspellers geeft op een dagpaneel IC −0,003.
  De voorspellers dragen te weinig informatie, niet te weinig modelcomplexiteit.
* **Trend is fragiel in zijn specificatie.** Lookbacks (7,14,28,56,112) geven 1,00 op 2021-10 → 2023-12 en
  (5,10,20,40,80,160) geeft 0,30, bij vrijwel dezelfde logica. Een weekdag-verklaring ("veelvouden van 7")
  bleek geen stand te houden: ladders die één dag verschoven zijn doen het niet systematisch slechter. Het is
  ruis in een ruw Sharpe-oppervlak, geen structuur, en daarom is er geen lookbackset gekozen.
* **XS-momentum is geen sleeve maar een gok.** Discovery 0,59, validatie −0,27, en in 2026H1 −29,5 % bij 20 % vol:
  een klassieke momentum-crash op zes munten. De lage correlatie met trend (0,07) bleef bestaan, maar een
  sleeve met negatieve verwachting diversifieert niets.
* **Cash-and-carry is dood als edge.** C1 levert 2022+ ≈ 4 % per jaar op het totale notioneel, gelijk aan de
  risicovrije voet; de Sharpe van 5,85 in de upload is een rendement-op-totaal met bijna nul vol, geen
  overrendement, en basisrisico is zonder spotdata niet meetbaar.
* **De enige robuuste winst zit in het tweede moment.** HAR-volatiliteit verslaat EWMA(0,94) elk jaar. Dat
  verbetert de risicoschaling, maar het is geen Sharpe van 1.

## Beperkingen die de uitkomst kunnen vertekenen

* **Survivorship:** de zes munten zijn de overlevers van nu (geen LUNA, FTT). Dat helpt long-posities en maakt
  short-posities eerder gunstiger; de netto richting is niet te bepalen zonder de gesneuvelde munten.
* **Het validatievenster is nu verbruikt** voor deze ontwerpfamilie (één lezing, daarna zijn ook de
  multiversum-resultaten ermee beoordeeld). Elke volgende variant is ontwikkeling, geen validatie.
* **Eén regime-opeenvolging:** 2023 levert 57 % van de som van de jaarrendementen van `TREND_MIX`; zonder 2023
  zakt zijn Sharpe van 0,85 naar 0,48 (≈ 10 % per jaar).
* **Uitvoering:** beslissing en fill op dezelfde close (lag 1), 6,5 bps taker; maker-orders zouden ≈ 0,1
  Sharpe terugwinnen maar vragen fill-aannames die dagdata niet kan toetsen. Geen capaciteits- of
  marktimpactmodel (klein eigen kapitaal).
* **Vol-schaling:** de schaling is causaal (getest met een toekomst-vervangingstest,
  `test_lib.py`), maar hefboom tot 4× in rustige regimes is een staartrisico dat de backtest niet prijst.

## Wat ik aanraad

1. **Niet live op basis van dit.** Verwachte Sharpe ≈ 0,3 (zie het krimpoordeel: dev 0,55 ± 0,5 tegen een
   sceptische prior van 0,2 ± 0,3 → posterior ≈ 0,3). Dat is geen edge die de operationele risico's betaalt.
2. **De enige hefboom die het antwoord echt verandert is breedte**, en daar zit de blokkade: het
   egress-beleid van deze sessie weigert alle marktdata-hosts (Binance, Bybit, OKX, Kraken, CoinGecko,
   CryptoCompare, Yahoo: 403). Met `api.bybit.com` (de bron van de bestaande ingestie) toegestaan, kan dezelfde
   pijplijn draaien op tientallen perps vanaf 2019–2020, en kan de poort-set van §3 opnieuw voorgeregistreerd
   worden voor een groter universum. Instelling: *Network access* van de cloud-omgeving → *Custom* → host
   toevoegen onder *Allowed domains* (en de standaardlijst van package managers laten staan), zie
   <https://code.claude.com/docs/en/cloud-environments#network-access>.
3. **Spotdata** voor het basisrisico van carry, en **andere activaklassen** (de klassieke weg naar een
   gediversifieerde trend-Sharpe rond 1): dat is data-uitbreiding, geen modelwerk.

## Reproduceren

```text
python research/repro_zip.py            # stap 0: T1/T2 exact
python research/stage_a_diag.py         # diagnose bestaande trend
python research/stage_c_screen.py       # voorspellerscreen (discovery)
python research/stage_c_build.py        # sleeves (discovery)
python research/stage_c_trend_variants.py
python research/stage_c_combo.py
python research/stage_d_validate.py     # EENMALIGE validatielezing
python research/stage_e_multiverse.py   # 54 specs, niets geselecteerd
python research/stage_f_dow.py          # weekdag + ladder-uitlijning
python research/stage_g_volforecast.py  # HAR vs EWMA
python research/stage_h_ml_ridge.py     # kitchen-sink ridge
python research/final_gates.py          # poorten A1-A6, machtsberekening
python -m pytest research/test_lib.py   # causaliteit + boekhouding
```
