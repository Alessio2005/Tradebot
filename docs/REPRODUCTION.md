# REPRODUCTION — van een schone machine naar een gereproduceerd resultaat

> **Geverifieerd tegen de codebase op 2026-09-01** (Phase 7/8, Stage E-1).
> Elk commando hieronder is nagelopen op bestaan en op wat het schrijft. Wat
> NIET is gedaan, staat in §6 — lees dat voordat u dit als "getoetst"
> beschouwt.

Eén pad, van `git clone` tot een resultaat waarvan u kunt narekenen dat het
hetzelfde is als het gepubliceerde. Geen alternatieve routes: waar meerdere
manieren bestaan, staat hier de bindende.

---

## 1. Wat u nodig hebt

| | |
|---|---|
| Python | 3.13 (de referentie-interpreter; zie `pyproject.toml`) |
| Git | voor de historie en de `git_sha` die elk artefact draagt |
| DVC | voor de pijplijn; `pip install dvc` of via de dev-lock |
| Schijfruimte | ~2 GB voor de PIT-store en de afgeleide artefacten |

Er is **geen** externe databron nodig om de kernresultaten te reproduceren: de
gecertificeerde point-in-time store staat in git (AD-12), niet onder DVC.

## 2. Het pad

```bash
# 1 ── de repository
git clone <remote> tradebot && cd tradebot

# 2 ── een schone omgeving met de VASTGELEGDE versies
python -m venv .venv
.venv/Scripts/activate            # Windows
# source .venv/bin/activate       # Linux/macOS
pip install -r requirements.lock
pip install -r requirements-dev.lock
pip install -e .                  # editable install; zie stap 3

# 3 ── bewijs dat de omgeving klopt vóór u iets meet
python -c "import tradebot, pathlib; print(pathlib.Path(tradebot.__file__).resolve())"
#   moet eindigen op  <repo>/src/tradebot/__init__.py
python apps/doctor.py             # omgevingsdiagnose

# 4 ── de suite, als eerste meting
python -m pytest tests -q
#   VERWACHT: exact 4 failures, alle vier killgates op cm_carry / cm_tsmom.
#   Minder dan vier betekent dat een killgate is uitgeschakeld.

# 5 ── de authoritative baseline
python apps/run_phase5_baseline.py
#   schrijft artefacts/baseline/phase5_revaluation.json

# 6 ── de Phase 6-oordelen, via de pijplijn
dvc repro phase6_adequacy
dvc repro phase6_econometrics
dvc repro phase6_h1_competition
dvc repro phase6_h2_regime_benchmark
dvc repro phase6_h3_meta_labeling
```

Stap 6 kan ook in één keer met `dvc repro`; de stages hangen via hun artefacten
aan elkaar en DVC bepaalt zelf de volgorde.

## 3. Hoe u weet dat het gereproduceerd IS

Een run die "doorloopt" bewijst niets. Elk artefact draagt zijn herkomst, en de
reproductie is geslaagd wanneer die drie velden overeenkomen met het
gepubliceerde artefact:

```bash
python - <<'PY'
import json, pathlib
for name in ["phase6_h1_competition", "phase6_h2_regime_benchmark",
             "phase6_h3_meta_labeling"]:
    d = json.loads(pathlib.Path(f"artefacts/governance/{name}.json").read_text(encoding="utf-8"))
    print(f"{name:<28} git_sha={d['git_sha']}  "
          f"prereg={d['preregistration_id'][:16]}")
PY
```

Sterker nog, en dat is de eigenlijke toets: **draai dezelfde stage twee keer en
vergelijk de bestanden byte voor byte.**

```bash
cp artefacts/governance/phase6_h3_meta_labeling.json /tmp/run1.json
python apps/run_meta_labeling.py
diff -q /tmp/run1.json artefacts/governance/phase6_h3_meta_labeling.json && echo IDENTIEK
```

**Gemeten op 2026-09-01:** drie opeenvolgende runs van
`apps/run_meta_labeling.py` leverden een bit-identiek artefact
(`md5 7e78b12305533aa53f17ad7345e8fd46`). Dat is exit-criterium 17 van Phase 6
met bewijs in plaats van met een bewering.

## 4. De monitoringdrempels

```bash
python apps/freeze_monitoring.py --check
```

Deze moet groen zijn vóór er een paper-trading-periode start. Wijkt hij af, dan
zijn de drempels gewijzigd sinds het bevriezen en herstart de 60-daagse klok
(no-go 10 en 11).

## 5. Wat u NIET hoeft te draaien

`dvc repro build_features`, `tune_hparams` en `train_cpcv` horen bij het
Wave-tijdperk-model en zijn **niet** nodig voor de Phase 5/6-resultaten. Zij
draaien op de featurestack uit `features/pipeline.py`; de authoritative keten
gebruikt de Phase 2-registry (`features/registry.py`) en leest rechtstreeks uit
de PIT-store. Zie `reports/phase7_divergence_map.md` §6.2 — dat er twee stacks
zijn, is een openstaand defect en geen ontwerp.

## 6. Wat hier NIET mee is aangetoond

1. **Dit pad is niet op een verse machine gedraaid.** Exit-criterium E1 eist
   dat wél. Wat is gedaan: elk commando is nagelopen op bestaan, elk pad dat
   het noemt is gecontroleerd, en de reproduceerbaarheid van stap 3 is gemeten
   op DEZE machine. Wat niet is gedaan: een clone op een computer zonder
   Python-omgeving, zonder caches en zonder deze repository. **E1 is daarmee
   niet gehaald**, en dat staat hier in plaats van dat het pad "getoetst" wordt
   genoemd.
2. **`dvc pull` staat niet in het pad.** Er is geen geconfigureerde remote;
   `dvc pull` zou falen. De gecertificeerde store staat in git, dus het pad
   werkt zonder — maar wie een DVC-remote toevoegt, moet deze stap hier bijzetten.
3. **De Wave-tijdperk-stages zijn niet gereproduceerd.** Zie §5.

---

*Bij twijfel over wat er vandaag waar is en wat niet: `docs/PROJECT_STATE.md`.*
