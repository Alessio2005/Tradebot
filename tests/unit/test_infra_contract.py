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
WORKFLOWS = sorted((ROOT / ".github/workflows").glob("*.yml"))


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
    assert DOCKERFILES and MANIFESTS and WORKFLOWS


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)
def test_every_workflow_runs_modules_that_exist(workflow: Path):
    """Een workflow die een niet-bestaande `tradebot`-module aanroept, faalt pas als hij
    draait: voor een geplande of handmatige workflow kan dat weken na de merge zijn."""
    for mod in _modules(workflow.read_text(encoding="utf-8")):
        if mod.startswith("tradebot"):
            assert importlib.util.find_spec(mod) is not None, f"{workflow.name}: {mod}"


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


# --------------------------------------------------------------------------- #
# De 24/7-dienst op één server: docker compose en systemd
# --------------------------------------------------------------------------- #
COMPOSE = ROOT / "infra/compose/docker-compose.yml"
UNIT = ROOT / "infra/systemd/tradebot-u1.service"


def test_compose_runs_the_service_hardened_restarting_and_health_checked():
    svc = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]["u1"]
    assert (ROOT / svc["build"]["dockerfile"]).is_file()
    for cmd in (svc["command"], svc["healthcheck"]["test"]):
        for module in _modules(" ".join(cmd)):
            assert importlib.util.find_spec(module) is not None, module
    assert "--healthcheck" in svc["healthcheck"]["test"]
    assert svc["restart"] == "unless-stopped" and svc["read_only"] is True
    assert svc["cap_drop"] == ["ALL"] and "no-new-privileges:true" in svc["security_opt"]
    assert any(v.endswith(":/var/lib/tradebot") for v in svc["volumes"])
    assert svc["env_file"] == ".env" and (ROOT / "infra/compose/.env.example").is_file()
    # Elk pad dat de dienst schrijft, ligt in het volume.
    cfg = yaml.safe_load((ROOT / "conf/live/u1.example.yaml").read_text(encoding="utf-8"))
    assert all(p.startswith("/var/lib/tradebot/") for p in cfg["paths"].values())


def test_the_env_example_names_every_secret_the_code_reads():
    names = set(re.findall(r"^([A-Z_]+)=", (ROOT / "infra/compose/.env.example").read_text(
        encoding="utf-8"), flags=re.M))
    src = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "src/tradebot/carry").glob(
        "*.py"))
    read = set(re.findall(r'"((?:KRAKEN|ALERT|TELEGRAM|HEARTBEAT)_[A-Z_]+)"', src))
    assert read and read <= names


def test_the_systemd_unit_runs_the_service_and_always_restarts():
    text = UNIT.read_text(encoding="utf-8")
    exec_start = re.search(r"^ExecStart=(.+)$", text, flags=re.M).group(1)
    for module in _modules(exec_start):
        assert importlib.util.find_spec(module) is not None, module
    for line in ("Restart=always", "User=tradebot", "ProtectSystem=strict",
                 "ReadWritePaths=/var/lib/tradebot", "StateDirectory=tradebot",
                 "EnvironmentFile=/etc/tradebot/u1.env", "NoNewPrivileges=true"):
        assert re.search(rf"^{re.escape(line)}$", text, flags=re.M), line
