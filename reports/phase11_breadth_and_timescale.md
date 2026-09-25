# FASE 11 — BREEDTE EN TIJDSCHAAL: DE METING

> **Nul trials.** Dit rapport bevat geen gemiddeld rendement, geen Sharpe van een
> signaal en geen IC. Alles hieronder is een tweede moment, een eigenschap van
> een besluitpaneel of omzet (R-15).

**Faseopdracht:** `Prompts-fases/fase_11_breedte_en_tijdschaal.md`
**Bron van elk getal:** `artefacts/governance/phase11_breadth.json`, geproduceerd
door `apps/run_breadth_measurement.py` op `src/tradebot/validation/breadth.py`,
`src/tradebot/validation/signal_clock.py` en
`src/tradebot/validation/phase11_breadth_measurement.py`
**Parameters:** `conf/research/breadth.yaml`, gezet vóór de eerste meting;
bootstrap uit `conf/validation/inference.yaml` (10.000 replicaties, seed
20260905, ci 0,95, bloklengte gekalibreerd)
**Risicobeleid:** niet van toepassing. Niets hier gaat door de risicolaag.

---

## 1. Breedte per constructie, per venster (stap 2)

De vensters zijn afgeleid en niet opnieuw gedefinieerd: `W_FULL` is het
bruikbare venster (ex-ante volatiliteit en causale ADV op elke naam, zoals
H-10.1), `W_DEV` is `development_slice` met de bevroren split, en `W_GATE` is
het complement. Het poortsample is niet via `gate_slice` gelezen;
`holdout_lock.json` draagt nog `reads: []`.

| venster | bars | eerste | laatste |
|---|---:|---|---|
| `W_DEV` | 1.390 | 2021-11-15 | 2025-09-04 |
| `W_GATE` | 353 | 2025-09-05 | 2026-08-23 |
| `W_FULL` | 1.743 | 2021-11-15 | 2026-08-23 |

Twee grootheden per rij, elk onder haar eigen naam (AD-30). **Onafhankelijke
weddenschappen** is de breedte. Het **ontwerpeffect** is wat de repository tot
nu toe "N_eff" noemde (DI-35).

| venster | constructie | rang | ρ̄ | **onafh. weddenschappen** | 95 %-interval | ontwerpeffect | 95 %-interval |
|---|---|---:|---:|---:|---|---:|---|
| `W_DEV` | directioneel | 6 | +0,7403 | **1,601** | [1,516; 1,684] | 1,276 | [1,240; 1,312] |
| `W_DEV` | dollar-neutraal | 5 | −0,1903 | **4,353** | [4,144; 4,500] | 123,58 | [45,60; 335,91] |
| `W_DEV` | bèta-gehedged (EW) | 5 | −0,1872 | **4,615** | [4,437; 4,717] | 93,58 | [47,06; 167,98] |
| `W_GATE` | directioneel | 6 | +0,8200 | **1,369** | [1,296; 1,468] | 1,176 | [1,143; 1,221] |
| `W_GATE` | dollar-neutraal | 5 | −0,1830 | **3,801** | [3,142; 4,336] | 70,67 | [40,34; 137,77] |
| `W_GATE` | bèta-gehedged (EW) | 5 | −0,1752 | **3,763** | [3,271; 4,157] | 48,33 | [30,05; 82,69] |
| `W_FULL` | directioneel | 6 | +0,7485 | **1,576** | [1,503; 1,647] | 1,265 | [1,234; 1,297] |
| `W_FULL` | dollar-neutraal | 5 | −0,1931 | **4,364** | [4,172; 4,503] | 172,71 | [59,69; 501,85] |
| `W_FULL` | bèta-gehedged (EW) | 5 | −0,1891 | **4,573** | [4,415; 4,674] | 109,67 | [55,38; 188,98] |

Vier lezingen.

**1. Zes namen zijn directioneel 1,6 weddenschap.** Het interval [1,516; 1,684]
sluit elke lezing uit waarin het universum "een beetje breed" is. Het
ontwerpeffect van 1,276 is het getal dat fase 10 gebruikte, en voor haar doel
(een gepoolde t defleren) is het juist.

**2. Neutraliseren tegen het mandje verdriedubbelt de breedte, tot vlak onder
de rangbovengrens.** 4,353 van maximaal 5. Het residu heeft vijf bijna gelijk
verdeelde eigenwaarden: de altcoinbewegingen tegen elkaar zijn werkelijk
verschillend, zodra de gezamenlijke factor eruit is.

**3. Het ontwerpeffect is op die constructie geen getal.** Een puntschatting
van 123,58 met een interval van 45,60 tot 335,91 op zes namen meet de nabijheid
van de noemer tot nul en niets anders. In 2025 alleen geeft zij 465,55. Dat is
de kwantitatieve inhoud van DI-35: wie deze grootheid als breedte leest, leest
ruis die een orde van grootte beweegt.

**4. Het poortvenster is directioneel aantoonbaar smaller.** 1,369
[1,296; 1,468] op `W_GATE` tegen 1,601 [1,516; 1,684] op `W_DEV`: de intervallen
overlappen niet. Dollar-neutraal is het verschil kleiner en niet scheiden
(3,801 [3,142; 4,336] tegen 4,353 [4,144; 4,500]). Een hypothese waarvan het
bewijs op directionele breedte leunt, wordt op een smaller universum gepoort
dan zij is ontwikkeld.

### 1.1 Per kalenderjaar (`W_FULL`)

| jaar | bars | directioneel | dollar-neutraal | bèta-gehedged (EW) | ontwerpeffect, dollar-neutraal |
|---|---:|---|---|---|---:|
| 2021 | 47 | 1,635 [1,397; 1,933] | 3,950 [3,345; 4,171] | 4,136 [3,410; 4,369] | 33,05 |
| 2022 | 365 | 1,430 [1,337; 1,500] | 4,131 [3,515; 4,440] | 4,474 [3,987; 4,677] | 46,52 |
| 2023 | 365 | 1,900 [1,680; 2,148] | 3,925 [3,612; 4,124] | 4,371 [4,066; 4,518] | 65,04 |
| 2024 | 366 | 1,738 [1,538; 1,978] | 4,116 [3,584; 4,451] | 4,401 [3,927; 4,635] | 307,73 |
| 2025 | 365 | 1,406 [1,320; 1,522] | 4,292 [3,692; 4,651] | 4,462 [4,031; 4,717] | 465,55 |
| 2026 | 235 | 1,351 [1,252; 1,485] | 4,067 [3,580; 4,398] | 3,863 [3,435; 4,170] | 51,88 |

De directionele breedte beweegt met het regime: 1,90 in 2023, 1,35 in 2026. De
dollar-neutrale breedte blijft binnen elk jaar tussen 3,9 en 4,3, met overlappende
intervallen. **De breedte die een neutraal boek ziet, is stabiel; die van een
richtingsboek niet.** Het ontwerpeffect springt in dezelfde jaren tussen 33 en 466.

### 1.2 Afwijkingen van de nulmeting van de faseopdracht (R-10)

| grootheid | faseopdracht §3.1 | gemeten | verklaring |
|---|---|---|---|
| interval directioneel, PR, `W_DEV` | [1,518; 1,687] | [1,516; 1,684] | 10.000 replicaties met de seed van de inferentiekern tegen 2.000 met een verkenningsseed; bijlage A van de opdracht voorspelde dit |
| interval dollar-neutraal, PR, `W_DEV` | [4,143; 4,513] | [4,144; 4,500] | idem |
| bèta-gehedged tegen BTC | 2,825 | niet in dit artefact | de constructie staat niet in `conf/research/breadth.yaml`; de opdracht noemde haar als diagnostiek, en zij is niet opnieuw gemeten |

Elke puntschatting reproduceert de faseopdracht op drie decimalen.
