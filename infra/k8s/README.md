# Tradebot — Kubernetes

Wat hier staat, kan draaien. De audit van 2026-10-24 (E-2,
`docs/research/2026-10-24-production-readiness-audit.md`) vond een live-Deployment
en twee CronJobs die `python -m apps.live_trader`, `apps.doctor` en
`apps.monitor_drift` aanriepen: modules die niet bestaan. De images bouwden op
Python 3.11 terwijl de package 3.13 eist. Die manifests zijn verwijderd.
`tests/unit/test_infra_contract.py` bewaakt dat het niet terugkomt.

## Bestanden

| Bestand | Doel |
| --- | --- |
| `namespace.yaml` | namespace `tradebot`, pod security `restricted` |
| `configmap-tradebot.yaml` | niet-geheime omgeving (TZ, numba-cache) |
| `pvc-state.yaml` | `tradebot-live-state` (staat van het boek), `tradebot-artefacts` (data en manifesten) |
| `networkpolicy.yaml` | default deny, plus HTTPS-egress voor het shadowboek |
| `cron-carry-shadow.yaml` | het carry-shadowboek (v8-T2), maandelijks |

## Volgorde

```bash
docker build -f infra/docker/Dockerfile.runner -t <registry>/tradebot/runner:1.0.0 .
kubectl apply -f infra/k8s/namespace.yaml
kubectl apply -f infra/k8s/configmap-tradebot.yaml
kubectl apply -f infra/k8s/pvc-state.yaml
kubectl apply -f infra/k8s/networkpolicy.yaml
kubectl apply -f infra/k8s/cron-carry-shadow.yaml
```

Eenmalig, na de eerste `binance_vision forward`-build, de beginstand van het boek
(de backtest tot `forward.start` in `conf/model/robust_book_v8.yaml`):

```bash
kubectl -n tradebot create job --from=cronjob/carry-shadow carry-init -- \
  python -I -m tradebot.carry init --state /opt/tradebot/state/carry.json
```

## Wat dit NIET is

- **Geen live handel.** Het shadowboek is paper: het model is de venue. Het legt
  elke maand per bar vast welke orders het boek zou hebben gestuurd, met dezelfde
  mechaniek als de backtest (bit-identiek, zie `tests/unit/test_carry.py`).
- **Geen dagelijks boek.** Binance publiceert funding en mark price als
  maandarchieven. Een dagelijks (of live) boek vraagt een datafeed en een
  venue-adapter van de beurs zelf (`src/tradebot/carry/venue.py` beschrijft het
  contract).
- **Geen promotiepoort.** De oude poort (“paper-trade-ci 7× groen”) toetste een
  run van 24 synthetische bars zonder signalen, dus zonder één order. Wat een
  promotie wel moet aantonen, staat in het auditrapport, §7.

## Noodstop

Het shadowboek houdt geen echte posities. Een plat boek met de staat op stil
(bijvoorbeeld voor een overgang naar een nieuwe staat):

```bash
python -I -m tradebot.carry unwind --state /opt/tradebot/state/carry.json --reason "<waarom>"
```
