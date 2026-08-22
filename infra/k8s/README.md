# Tradebot — Kubernetes manifests (Wave 11)

This directory contains **stub** manifests to deploy Tradebot v1.0 on any
Kubernetes 1.27+ cluster. The manifests are deliberately minimal: no Helm
chart, no operator, no ArgoCD glue. Adopt those layers when the on-call team
is ready to own the cluster surface.

## Files

| File                          | Purpose                                          |
|-------------------------------|--------------------------------------------------|
| `namespace.yaml`              | Restricted-pod-security namespace `tradebot`     |
| `configmap-tradebot.yaml`     | Non-secret runtime env (paths, logging, TZ)      |
| `secret-binance.yaml`         | Template for API keys + Slack webhook            |
| `pvc-state.yaml`              | RWO PVC for live-engine state + RWX for artefacts|
| `cron-research.yaml`          | Nightly `dvc repro` (02:00 UTC)                  |
| `cron-monitor-drift.yaml`     | Hourly drift monitor + alerting                  |
| `deployment-live.yaml`        | Live engine Deployment + ClusterIP Service       |
| `networkpolicy.yaml`          | Default-deny + Binance/Slack egress whitelist    |

## Apply order

```bash
kubectl apply -f infra/k8s/namespace.yaml
kubectl apply -f infra/k8s/configmap-tradebot.yaml
kubectl create -f infra/k8s/secret-binance.yaml   # edit first
kubectl apply -f infra/k8s/pvc-state.yaml
kubectl apply -f infra/k8s/networkpolicy.yaml
kubectl apply -f infra/k8s/cron-research.yaml
kubectl apply -f infra/k8s/cron-monitor-drift.yaml
kubectl apply -f infra/k8s/deployment-live.yaml
```

## Image build

Push to your registry:
```bash
docker build -f infra/docker/Dockerfile.runner -t <registry>/tradebot/runner:0.5.0 .
docker build -f infra/docker/Dockerfile.live   -t <registry>/tradebot/live:1.0.0  .
docker push <registry>/tradebot/runner:0.5.0
docker push <registry>/tradebot/live:1.0.0
```

Replace `tradebot/runner:0.5.0` and `tradebot/live:1.0.0` references in the
manifests with your registry URL.

## Promotion gate

`live.mode=paper` is the default in `deployment-live.yaml`. Promote to
`live.mode=live` only after:

1. `paper-trade-ci.yml` (Wave 12) has been green for **≥ 7 consecutive runs**.
2. `ShadowTrader` has logged **≥ 14 days** of challenger PnL alongside the
   incumbent champion.
3. `ChampionChallenger.evaluate()` reports `recommendation == "PROMOTE"`
   (Diebold-Mariano p-value < 0.05 **AND** challenger Sharpe > champion + 0.05).
4. The latest `MRMReport` is signed off (`approved_by` non-empty in the JSON).

Update the `live.mode` arg in `deployment-live.yaml` and roll out:
```bash
kubectl -n tradebot rollout restart deployment/tradebot-live
```

## Halting the engine in an emergency

```bash
kubectl -n tradebot scale deployment/tradebot-live --replicas=0
```

Pre-stop hook gracefully cancels open orders (60 s grace period). For an
unplanned outage, the circuit breaker should fire first; if it did not,
manually inspect `tradebot-live-state` PVC content for the last
`circuit_log.jsonl` entry.

## What is **not** here (intentional)

- Helm chart (use Kustomize overlays per env if needed)
- Horizontal Pod Autoscaler (live engine is intentionally single-replica)
- Service mesh integration (Istio/Linkerd — add later for mTLS at scale)
- Vertical Pod Autoscaler (memory budget is stable; over-allocation is fine)
- ArgoCD / Flux ApplicationSet (decide on GitOps tool first)
