"""Structural tests for Wave 10 (Docker), Wave 11 (K8s), Wave 13 (docs).

These tests are not exhaustive container-runtime tests — they assert that
the manifest files exist with the structural properties the blueprint
requires (image names, security contexts, doc sections, ADR count).
"""
from __future__ import annotations

import pathlib

import pytest
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Wave 10 — Docker
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rel_path",
    [
        "infra/docker/Dockerfile.research",
        "infra/docker/Dockerfile.runner",
        "infra/docker/Dockerfile.live",
        "infra/docker/.dockerignore",
        "infra/compose/docker-compose.dev.yml",
        "infra/compose/docker-compose.live.yml",
        "infra/compose/prometheus.yml",
    ],
)
def test_wave10_docker_files_exist(rel_path: str):
    assert (_ROOT / rel_path).exists(), f"Missing Wave 10 file: {rel_path}"


def test_wave10_dockerfile_live_non_root_and_healthcheck():
    content = (_ROOT / "infra/docker/Dockerfile.live").read_text(encoding="utf-8")
    assert "USER tradebot" in content
    assert "HEALTHCHECK" in content
    assert "TRADEBOT_LIVE_STATE_DIR" in content


def test_wave10_compose_live_has_engine_and_prometheus():
    text = (_ROOT / "infra/compose/docker-compose.live.yml").read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    services = data.get("services", {})
    assert "live_engine" in services
    assert "prometheus" in services
    assert "grafana" in services


# ---------------------------------------------------------------------------
# Wave 11 — Kubernetes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rel_path",
    [
        "infra/k8s/namespace.yaml",
        "infra/k8s/configmap-tradebot.yaml",
        "infra/k8s/secret-binance.yaml",
        "infra/k8s/pvc-state.yaml",
        "infra/k8s/cron-research.yaml",
        "infra/k8s/cron-monitor-drift.yaml",
        "infra/k8s/deployment-live.yaml",
        "infra/k8s/networkpolicy.yaml",
        "infra/k8s/README.md",
    ],
)
def test_wave11_k8s_files_exist(rel_path: str):
    assert (_ROOT / rel_path).exists(), f"Missing Wave 11 manifest: {rel_path}"


def test_wave11_deployment_live_is_single_replica_recreate():
    """Live engine must run with replicas=1 and strategy=Recreate to avoid
    two engines placing orders simultaneously."""
    manifest = (_ROOT / "infra/k8s/deployment-live.yaml").read_text(encoding="utf-8")
    docs = list(yaml.safe_load_all(manifest))
    deployment = next(d for d in docs if d and d.get("kind") == "Deployment")
    spec = deployment["spec"]
    assert spec["replicas"] == 1
    assert spec["strategy"]["type"] == "Recreate"


def test_wave11_namespace_has_restricted_pod_security():
    manifest = (_ROOT / "infra/k8s/namespace.yaml").read_text(encoding="utf-8")
    ns = yaml.safe_load(manifest)
    labels = ns["metadata"]["labels"]
    assert labels.get("pod-security.kubernetes.io/enforce") == "restricted"


# ---------------------------------------------------------------------------
# Wave 12 — CI/CD
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rel_path",
    [
        ".github/workflows/paper-trade-ci.yml",
        ".github/workflows/nightly-regression.yml",
        "scripts/paper_trade_report.py",
    ],
)
def test_wave12_workflows_and_scripts_exist(rel_path: str):
    assert (_ROOT / rel_path).exists(), f"Missing Wave 12 file: {rel_path}"


def test_wave12_paper_trade_workflow_schedules_weekly():
    text = (_ROOT / ".github/workflows/paper-trade-ci.yml").read_text(encoding="utf-8")
    assert "cron:" in text
    assert "0 2 * * 1" in text   # weekly Monday 02:00 UTC


# ---------------------------------------------------------------------------
# Wave 13 — Documentation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "rel_path",
    [
        "docs/architecture.md",
        "docs/data_dictionary.md",
        "docs/runbook.md",
        "docs/tca_methodology.md",
        "docs/model_risk_policy.md",
        "docs/adr/0006-hrp-over-markowitz.md",
        "docs/adr/0007-event-loop-asyncio.md",
        "docs/adr/0008-shadow-before-live.md",
        "docs/adr/0009-feature-store-design.md",
        "docs/adr/0010-audit-log-schema.md",
        "CHANGELOG.md",
    ],
)
def test_wave13_docs_exist(rel_path: str):
    p = _ROOT / rel_path
    assert p.exists(), f"Missing Wave 13 doc: {rel_path}"
    assert p.stat().st_size > 200, f"Doc too short: {rel_path}"


def test_wave13_runbook_lists_circuit_breaker_section():
    text = (_ROOT / "docs/runbook.md").read_text(encoding="utf-8")
    assert "CIRCUIT_BREAKER" in text or "circuit_breaker" in text.lower()
    assert "FEED_TIMEOUT" in text or "feed timeout" in text.lower()


def test_wave13_changelog_mentions_audit_fixes():
    text = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "AUDIT A-1" in text
    assert "AUDIT D-2" in text
    assert "AUDIT E-4" in text
