"""De infra kan draaien: wat een image of manifest aanroept, bestaat (audit 2026-10-24, E-2).

De vorige infra-tests toetsten dat de bestanden BESTONDEN. Ze waren groen terwijl elk
image faalde (Python 3.11 tegen `requires-python >= 3.13`, `COPY apps/` van een map die
er niet is) en elk manifest een module aanriep die niet bestaat. Hier staat wat er
moet kloppen om te kunnen bouwen en starten.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
import tomllib
import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILES = sorted(p for p in (ROOT / "infra/docker").glob("Dockerfile*")
                     if not p.name.endswith(".dockerignore"))
MANIFESTS = sorted((ROOT / "infra/k8s").glob("*.yaml"))


def _docs(path: Path) -> list[dict]:
    return [d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d]


def _containers(doc: dict) -> list[dict]:
    spec = doc.get("spec", {})
    pod = spec.get("jobTemplate", {}).get("spec", {}).get("template", {}).get("spec") or \
        spec.get("template", {}).get("spec") or {}
    return list(pod.get("containers", []))


def _pod(doc: dict) -> dict:
    spec = doc["spec"]
    return spec.get("jobTemplate", {}).get("spec", {}).get("template", {}).get("spec") or \
        spec["template"]["spec"]


def _modules(text: str) -> list[str]:
    flat = text.replace('"', " ").replace(",", " ")       # ook de JSON-vorm van CMD
    return re.findall(r"python\s+(?:-I\s+)?-m\s+([\w.]+)", flat)


def test_there_is_something_to_check():
    assert DOCKERFILES and MANIFESTS


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: p.name)
def test_every_image_runs_the_python_the_package_requires(dockerfile: Path):
    need = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    floor = need["project"]["requires-python"].removeprefix(">=")
    text = dockerfile.read_text(encoding="utf-8")
    version = re.search(r"ARG\s+PYTHON_VERSION=(\S+)", text).group(1)  # type: ignore[union-attr]
    assert tuple(map(int, version.split("."))) >= tuple(map(int, floor.split(".")))


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: p.name)
def test_every_image_installs_from_the_lock_and_copies_what_exists(dockerfile: Path):
    text = dockerfile.read_text(encoding="utf-8")
    assert "pip install -r requirements.lock" in text, "de omgeving is de lock"
    for line in re.findall(r"^COPY\s+(?!--from)(?:--\S+\s+)*(.+)$", text, flags=re.M):
        *sources, _dest = line.split()
        for src in sources:
            assert (ROOT / src).exists(), f"{dockerfile.name}: COPY {src} bestaat niet"
    for module in _modules(text):
        assert importlib.util.find_spec(module) is not None, module


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda p: p.name)
def test_the_build_context_keeps_what_the_image_copies(dockerfile: Path):
    ignore = dockerfile.with_name(dockerfile.name + ".dockerignore")
    assert ignore.is_file(), "BuildKit leest alleen <Dockerfile>.dockerignore naast de Dockerfile"
    patterns = {p.strip().rstrip("/") for p in ignore.read_text(encoding="utf-8").splitlines()
                if p.strip() and not p.startswith("#")}
    copied = {s.rstrip("/") for line in re.findall(r"^COPY\s+(?!--from)(.+)$",
                                                    dockerfile.read_text(encoding="utf-8"),
                                                    flags=re.M)
              for s in line.split()[:-1]}
    assert not copied & patterns


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda p: p.name)
def test_every_manifest_parses_and_runs_modules_that_exist(manifest: Path):
    for doc in _docs(manifest):
        for c in _containers(doc):
            cmd = " ".join(c.get("command", []) + c.get("args", []))
            for module in _modules(cmd):
                assert importlib.util.find_spec(module) is not None, f"{manifest.name}: {module}"


def test_every_claim_is_declared_and_every_pod_is_locked_down():
    claims = {d["metadata"]["name"] for d in _docs(ROOT / "infra/k8s/pvc-state.yaml")
              if d["kind"] == "PersistentVolumeClaim"}
    workloads = [d for m in MANIFESTS for d in _docs(m) if d["kind"] in ("CronJob", "Deployment")]
    assert workloads
    for doc in workloads:
        pod = _pod(doc)
        assert pod["securityContext"]["runAsUser"] != 0
        for vol in pod.get("volumes", []):
            if "persistentVolumeClaim" in vol:
                assert vol["persistentVolumeClaim"]["claimName"] in claims
        for c in pod["containers"]:
            assert c["securityContext"]["readOnlyRootFilesystem"] is True
            assert c["securityContext"]["allowPrivilegeEscalation"] is False


def test_every_workload_may_reach_the_network_it_needs():
    """Default deny + een egress-policy per component: een pod zonder policy kan niets."""
    policies = [d for d in _docs(ROOT / "infra/k8s/networkpolicy.yaml")
                if "Egress" in d["spec"].get("policyTypes", []) and d["spec"]["podSelector"]]
    allowed = {p["spec"]["podSelector"]["matchLabels"]["app.kubernetes.io/component"]
               for p in policies}
    for m in MANIFESTS:
        for doc in _docs(m):
            if doc["kind"] in ("CronJob", "Deployment"):
                labels = (doc["spec"].get("jobTemplate", {}).get("spec", {})
                          .get("template", {}).get("metadata", {}).get("labels") or
                          doc["spec"]["template"]["metadata"]["labels"])
                assert labels["app.kubernetes.io/component"] in allowed


def test_the_namespace_enforces_restricted_pod_security():
    ns = _docs(ROOT / "infra/k8s/namespace.yaml")[0]
    assert ns["metadata"]["labels"]["pod-security.kubernetes.io/enforce"] == "restricted"
