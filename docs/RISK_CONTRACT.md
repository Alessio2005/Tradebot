# RISK CONTRACT — L7 Independent Risk & Volatility Targeting

> **Geverifieerd tegen de codebase op 2026-09-01** (Phase 7/8, Stage E-3).
> De limieten komen uit `conf/risk/default.yaml`; sinds Stage D geldt
> dat ook voor de live-keten (`CircuitBreakerConfig.from_risk_config`,
> `ExecutionControllerConfig(risk=...)`). De GETALLEN in dit document
> zijn niet een voor een tegen de YAML gehertoetst.

> **Fase:** 4, stap 2 · **Status:** bindend vanaf `git_sha` van deze commit
> **Bindend brondocument:** `ARCHITECTUUR_AUDIT_2026-08-22.md` — secties 11.1, 14, 14.1, 19, 24
> **Implementatie:** `src/tradebot/risk/contract.py` (types), `risk/engine.py` (compositie)
> **Voorgeschiedenis:** `reports/phase4_entanglement_map.md`

Dit document legt het interfacecontract vast **voordat** er geïmplementeerd
wordt. Elke module in `risk/` die na deze commit wordt geschreven, voldoet
hieraan; elke afwijking is een contractschending en geen ontwerpvrijheid.

---

## 1. De handtekening

```
decide(desired_exposure, market_state, risk_state) -> RiskDecision
```

**Puur.** Geen I/O, geen wall-clock, geen globale state, geen mutatie van de
argumenten. Dezelfde drie inputs leveren altijd bit-identiek dezelfde output.
Dat is niet stijl maar de voorwaarde voor exit-criterium 2: de functionele helft
van de ontkoppelingstest voert twee verschillende alpha-units met een identieke
`a_t`-reeks door de engine en eist bit-identieke `permitted_exposure`.

De engine is **stateless**. Alle toestand die over bars heen leeft, zit in
`RiskState` en wordt expliciet doorgegeven en expliciet teruggegeven. Er is geen
`self.equity`, geen `self._returns_history` en geen `update()` die stilzwijgend
een interne buffer bijwerkt — dat patroon uit `risk/portfolio.py` is precies
wat de bit-identiteitseis onmogelijk maakte.

---

## 2. Input 1 — `desired_exposure`

`Mapping[str, float]`, symbool → `a_t ∈ [-1, +1]`.

Dit is het **enige** dat L4 levert (audit §11.1). Wat de engine hier ontvangt is
een *wens*, geen positie.

**Wat er expliciet NIET in zit, en wat de engine dus niet mag kennen:**

| Verboden input | Waarom |
|---|---|
| Modelnaam, unit-naam, strategie-identiteit | §14: L7 kent geen strategie-identiteit. Een limiet die van de naam afhangt, is een bypass met een omweg. |
| Verwachte alpha / edge / μ̂ | Bevinding E3. Positiegrootte die meeschaalt met modelovertuiging is de bypass die §14 uitsluit. |
| Meta-label-confidence, signal confidence | Bevinding E4/E6. Idem. |
| Backtest-performance, Sharpe, hitrate | Een limiet gekalibreerd op gerealiseerde performance is een lookahead-lek met kapitaalgevolgen. |
| Conviction-, override- of bypass-vlaggen | §14: er bestaat geen high-conviction-uitzondering. |

**Validatie is fail-fast.** Een waarde buiten `[-1, +1]`, `NaN` of `±inf` is
geen aanleiding om te clippen maar om te crashen: een alpha-unit die buiten zijn
contract levert, is kapot, en het stilzwijgend terugsnijden verbergt dat.

---

## 3. Input 2 — `market_state`

Uitsluitend **gemeten marktgrootheden**, niet-conditioneel op enige strategie.

| Veld | Type | Bron | Fail-fast bij |
|---|---|---|---|
| `asof_ts` | UTC timestamp | de bar | ontbrekend of niet-UTC |
| `sigma_hat` | symbool → ex-ante `σ̂_{t+1\|t}`, geannualiseerd | **L2**, `volatility/ewma.py` met λ uit `conf/model/volatility.yaml` | ontbrekend, `NaN`, `±inf`, `<= 0` |
| `adv_usd` | symbool → average daily volume in USD | L0/L3 | ontbrekend wanneer de ADV-limiet aanstaat |
| `cluster` | symbool → sector/cluster-label | gemeten; leeg ⇒ terugval op `conf/risk/` | onbekend symbool wanneer de clusterlimiet aanstaat |

**De clusterstructuur is een marktfeit, geen alleen-configuratie.** Normaal
levert de caller er geen en gebruikt de engine de labels uit `conf/risk/`. Maar
wanneer alle correlaties naar 1 gaan (scenario S2), *is* het universum één
cluster; een engine die dan blijft rekenen met de geconfigureerde labels meet
een diversificatie die er niet meer is. Wordt `cluster` wél geleverd, dan wint
de gemeten structuur — ook in de postconditiecontrole.

**`σ̂` is ex-ante en causaal.** Hij gebruikt uitsluitend informatie tot en met
`t`. De EWMA-estimator propageert `NaN` tijdens de burn-in en vult die bewust
niet op; de engine crasht op die `NaN` in plaats van hem te interpreteren.
Er is **geen** laatste-bekende-waarde, **geen** cross-sectionele mediaan en
**geen** constante vol. Dat is exit-criterium 4.

**Waarom geen regimemodel in `market_state`.** Bevinding E8: een HMM in het
besluitpad maakt de risicolaag afhankelijk van een gefit model met eigen
parameters, eigen burn-in en eigen faalmodi. Regime-informatie mag uitsluitend
binnenkomen als *gemeten* grootheid (bijv. een realized-vol-ratio), nooit als
modeloutput. `risk/hmm_regime.py` verlaat het besluitpad in stap 7.

---

## 4. Input 3 — `risk_state`

De toestand die over bars heen leeft. Expliciet, serialiseerbaar, persistent.

| Veld | Betekenis | Persistentie-eis |
|---|---|---|
| `equity` | huidige boekwaarde | — |
| `high_water_mark` | causale HWM: max van equity tot en met `t` | moet herstart overleven |
| `day_start_equity` | equity bij aanvang van de handelsdag | moet herstart overleven |
| `halted` | `bool` — de onherroepelijke `HALTED`-toestand | **moet herstart overleven** |
| `halt_reason` | waarom, met de bindende constraint | idem |
| `halted_at` | wanneer | idem |

**De HWM is causaal.** Hij wordt voorwaarts opgebouwd uit de equity-reeks tot en
met `t`. Een HWM berekend over de volledige sample is een lookahead-lek dat de
drawdown-breaker systematisch te laat laat vuren in de eerste helft van de
sample en te vroeg in de tweede.

**`halted` is een eenrichtingsdeur.** Geen enkele codepad zet hem terug op
`False`. Herstel vereist een expliciete, handmatige operatorhandeling op de
persistente store. Dat is exit-criterium 5.

---

## 5. Output — `RiskDecision`

| Veld | Type | Betekenis |
|---|---|---|
| `permitted_exposure` | `Mapping[str, float]` | wat L8 mag construeren |
| `binding_constraints` | `tuple[BindingConstraint, ...]` | het auditspoor, geordend op toepassingsvolgorde |
| `unconstrained` | `bool` | `True` **alleen** wanneer geen enkele limiet bond |
| `config_hash` | `str` | hash van de risicoconfiguratie die dit besluit produceerde |
| `risk_state_out` | `RiskState` | de nieuwe toestand na dit besluit |

### 5.1 Het auditspoor is machineleesbaar

Elke `BindingConstraint` draagt:

```
kind             welke limiet bond (enum, geen vrije tekst)
scope            symbool, of "__book__" voor een boekbrede limiet
measured         de GEMETEN waarde
threshold        de GECONFIGUREERDE drempel
config_key       de sleutel in conf/risk/ die die drempel zette
exposure_before  exposure vóór deze limiet
exposure_after   exposure na deze limiet
```

Dat is de invulling van exit-criterium 6: *"elke risicobeslissing is herleidbaar
tot een geconfigureerde drempel en een gemeten waarde"*. De vervanger van
`SizingDecision.reason = f"vol_mult={...} corr_mult={...}"` — een string die
niets vertelt over wélke drempel bond of waar hij vandaan kwam.

### 5.2 `unconstrained` is verplicht en expliciet

Uit deliverable 4: *"De engine geeft nooit `a_t` ongewijzigd door zonder
expliciete registratie dat geen enkele limiet bond."* Het contract dwingt dat af
als invariant:

```
unconstrained == (len(binding_constraints) == 0)
permitted_exposure == desired_exposure   ⟹   unconstrained is True
```

Een engine die stilletjes de wens doorgeeft, is niet te onderscheiden van een
engine die kapot is. Deze vlag maakt dat onderscheid meetbaar.

### 5.3 Monotoniteit

```
|permitted_exposure[s]| <= |desired_exposure[s]|   voor elk symbool s
sign(permitted_exposure[s]) ∈ {0, sign(desired_exposure[s])}
```

De risicolaag **verkleint uitsluitend**. Hij vergroot nooit een positie, en
draait nooit een teken om. Een risicolaag die exposure kan toevoegen, is geen
risicolaag maar een tweede alpha-unit.

---

## 6. Toepassingsvolgorde

De volgorde is **geconfigureerd**, niet impliciet in de code-volgorde
(stap 4-eis). Hij staat in `conf/risk/default.yaml` als `constraint_order` en de
engine leest hem daar; een limiet die in de config staat maar niet in de
volgorde, is een fout en geen stilzwijgende overslag.

De vastgelegde canonieke volgorde:

| # | Fase | Waarom hier |
|---|---|---|
| 1 | `halted` | Een gehalt boek heeft geen exposure. Alles daarna is betekenisloos. |
| 2 | `daily_loss_governor` | Onmiddellijke kill switch; zet exposure op nul en zet `halted`. |
| 3 | `drawdown_breaker` | Getrapte de-grossing: een multiplier, geen nul. |
| 4 | `vol_target` | De boekbrede schaal `min(MaxLeverage, σ_target/σ̂)`. |
| 5 | `per_asset_cap`, `adv_cap`, `concentration_cap`, `cluster_cap` | Per-symbool clamps. |
| 6 | `gross_cap`, `net_cap` | Boekbrede clamps, **als laatste**. |

**Waarom de boekbrede caps als laatste.** Alleen dan geldt de garantie
onvoorwaardelijk: wat er in stap 1-5 ook gebeurt, de uiteindelijke gross en net
exposure overschrijden hun cap niet. Draai je de volgorde om, dan kan een latere
per-symbool-stap de gross opnieuw over de cap tillen — precies de klasse fout
die `risk/portfolio.py::size_position` maakt door de HMM-cap ná de gross-cap toe
te passen.

---

## 7. Wat de engine bij twijfel doet

**Crashen.** Niet degraderen. Uit de faseregels:

> *"Een risicolaag die stilzwijgend degradeert naar 'geen limiet' is de
> gevaarlijkste module in het platform."*

| Situatie | Gedrag |
|---|---|
| `σ̂` ontbreekt, `NaN`, `inf` of `<= 0` | `DataContractError` |
| Limietconfiguratie ontbreekt of is onvolledig | `ConfigContractError` |
| `a_t` buiten `[-1, +1]` of niet-eindig | `DataContractError` |
| `risk_state` corrupt (HWM < 0, equity niet-eindig) | `DataContractError` |
| Symbool in `desired_exposure` zonder `σ̂` | `DataContractError` |
| `constraint_order` noemt een onbekende limiet | `ConfigContractError` |

Er is **nul** `try/except` in `risk/`. Het enige toegestane guard-idioom is
`utils.failfast.require`, statisch gehandhaafd door
`scripts/audit_fallbacks.py --strict`.

---

## 8. Configuratie

Alle drempels komen uit `conf/risk/default.yaml`, gevalideerd tegen
`schemas/config.py::RiskConfig` (`extra="forbid"`, `frozen=True`). Nul defaults
in functiehandtekeningen — nieuwe modules in `risk/` staan onder ratchet-budget
0 in `scripts/check_hardcoded_params.py`.

Elk besluit draagt de `config_hash` van de configuratie die het produceerde.
Een risicoconfiguratie zonder hash is niet auditbaar (stap 11).

**De GETALLEN staan niet in dit document, en dat is opzet.** Dit is het
interfacecontract; de drempels zijn beleid. Waar dat beleid vandaan komt — en
welke drempel door het inmiddels vervallen propfirm-mandaat werd gezet, welke
uit de fase-4-opruiming kwam, en welke een marktfeit is dat niet mag
verruimen — staat in `docs/RISK_MANDATE.md`. Dit contract is bij die
mandaatwijziging met geen letter veranderd: een ruimere drempel is geen ruimer
contract.

**Wat NIET in `conf/risk/` hoort**, en in stap 8 verdwijnt:
`min_signal_confidence` (bevinding E6, een alpha-parameter) en
`max_funding_cost_bps_day` (bevinding E7, een kostendrempel).

---

## 9. Wat dit contract onmogelijk maakt

Ter controle bij review — als een van deze constructies ooit weer kan, is het
contract geschonden:

1. Een alpha-unit die zijn eigen leverage bepaalt (`desired_exposure` is
   begrensd op `[-1, +1]` en draagt geen grootte).
2. Een engine die meer exposure teruggeeft dan gevraagd (§5.3).
3. Een limiet die bindt zonder dat het ergens staat (§5.2).
4. Een drempel die niet uit `conf/risk/` komt (§8).
5. Een `HALTED`-boek dat na een herstart weer handelt (§4).
6. Een besluit dat doorgaat zonder `σ̂` (§7).
7. Een risicolaag die twee identieke `a_t`-reeksen verschillend behandelt omdat
   ze van verschillende units komen (§1, §2).
