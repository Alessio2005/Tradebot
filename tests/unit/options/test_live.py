"""De live-schakelaar: alleen alle vijf voorwaarden samen; in deze repository nooit aan."""
from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import pytest

from tradebot.options import cli
from tradebot.options.live import (
    CONDITIONS,
    LIVE_CONFIRM_PHRASE,
    LiveChecks,
    LiveRefused,
    account_hash,
    gather,
    is_paper_account,
    live_decision,
    require_paper,
)

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.killgate
@pytest.mark.parametrize("combo", [c for c in itertools.product((False, True), repeat=5)
                                   if not all(c)])
def test_every_incomplete_combination_refuses_live(combo):
    ok, missing = live_decision(LiveChecks(*combo))
    assert not ok
    assert missing == tuple(n for n, v in zip(CONDITIONS, combo, strict=True) if not v)


def test_only_all_five_together_allow_live():
    assert live_decision(LiveChecks(True, True, True, True, True)) == (True, ())


def test_gather_reads_every_condition(tmp_path):
    acct = "U7654321"
    env = {"TRADEBOT_OPTIONS_LIVE_CONFIRM": LIVE_CONFIRM_PHRASE}
    assert gather("paper", acct, tmp_path, {}) == LiveChecks(False, False, False, True, False)
    (tmp_path / "LIVE_APPROVED").write_text(account_hash(acct) + "\n", encoding="utf-8")
    (tmp_path / "paper_verdict.json").write_text(json.dumps({"verdict": "pass"}),
                                                 encoding="utf-8")
    assert live_decision(gather("live", acct, tmp_path, env))[0]
    assert not gather("live", "DU1", tmp_path, env).approved_file
    assert gather("paper", "DU1", tmp_path).real_account is False


def test_the_paper_service_only_takes_paper_accounts():
    assert is_paper_account("DU1234567") and not is_paper_account("U1234567")
    require_paper("DU1234567")
    with pytest.raises(LiveRefused, match="live staat uit"):
        require_paper("U1234567")


def test_nothing_in_the_repository_switches_live_on():
    pattern = re.compile(r"^\s*mode\s*[:=]\s*[\"']?live\b", re.IGNORECASE | re.MULTILINE)
    hits = []
    for base in ("conf", "infra"):
        for p in (ROOT / base).rglob("*"):
            if p.is_file() and p.suffix in {".yaml", ".yml", ".env", ".service", ".example",
                                            ".toml", ".json"}:
                if pattern.search(p.read_text(encoding="utf-8", errors="ignore")):
                    hits.append(p.relative_to(ROOT).as_posix())
    assert hits == []
    with pytest.raises(SystemExit):
        cli.main(["live"])


def test_the_options_service_is_paper_hardened_and_health_checked():
    import importlib.util

    import yaml
    services = yaml.safe_load((ROOT / "infra/compose/docker-compose.yml").read_text(
        encoding="utf-8"))["services"]
    gw, svc = services["ib-gateway"], services["options-paper"]
    assert gw["environment"]["TRADING_MODE"] == "paper" and gw["env_file"] == ".env.ib"
    assert svc["env_file"] == ".env.options"          # least privilege: eigen geheimen per dienst
    assert svc["command"][:4] == ["python", "-I", "-m", "tradebot.options"]
    assert svc["command"][4] == "paper" and "--healthcheck" in svc["healthcheck"]["test"]
    assert svc["read_only"] is True and svc["cap_drop"] == ["ALL"]
    assert svc["restart"] == "unless-stopped" and svc["depends_on"] == ["ib-gateway"]
    assert importlib.util.find_spec("tradebot.options.__main__") is not None
    ib_env = (ROOT / "infra/compose/.env.ib.example").read_text(encoding="utf-8")
    for name in ("TWS_USERID=", "TWS_PASSWORD="):
        assert re.search(rf"^{name}$", ib_env, flags=re.M), name
    opt_env = (ROOT / "infra/compose/.env.options.example").read_text(encoding="utf-8")
    assert re.search(r"^IB_ACCOUNT=$", opt_env, flags=re.M)
    assert "TWS_PASSWORD" not in opt_env and "KRAKEN" not in opt_env
    assert "TWS_" not in (ROOT / "infra/compose/.env.example").read_text(encoding="utf-8")
    unit = (ROOT / "infra/systemd/tradebot-options-paper.service").read_text(encoding="utf-8")
    for line in ("Restart=always", "User=tradebot", "ProtectSystem=strict",
                 "EnvironmentFile=/etc/tradebot/options.env", "NoNewPrivileges=true",
                 "ExecStart=/opt/tradebot/.venv/bin/python -I -m tradebot.options paper "
                 "--run-dir /var/lib/tradebot/options"):
        assert re.search(rf"^{re.escape(line)}$", unit, flags=re.M), line
    assert (ROOT / "docs/runbook-options.md").is_file()
