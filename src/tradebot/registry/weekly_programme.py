"""Het bevriezen van het wekelijkse programma, vóór de eerste fit (spec §9, §17.2-3).

Drie handelingen, in deze volgorde en elk precies één keer:
1. Het ongelezen holdout-slot krijgt de nieuwe split (2026-06-24).
2. De vier geplande trials worden als EERSTE entry in de (lege) ledger geboekt.
3. De preregistratie wordt bevroren met M = 4 in haar parameters, zodat
   `frozen_trial_count` en daarmee de DSR hem uit het bevroren artefact leest.
"""
from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..features.base import DataRegister
from ..features.registry import current_git_sha
from ..schemas.weekly_meta import WeeklyMetaConfig, weekly_meta_config
from ..utils.failfast import DataContractError, require
from ..utils.hashing import hash_config
from ..validation.holdout import HoldoutLock, freeze_holdout
from .hypothesis_ledger import HypothesisLedger, LedgerEntry
from .preregistration import freeze_preregistration, load_preregistration_spec

__all__ = ["PROGRAMME_UNIT", "book_and_freeze", "certified_data_hashes",
           "programme_parameters", "refreeze_unread_holdout"]

PROGRAMME_UNIT = "weekly_meta_programme"
_SERIES = (("ohlcv", "1d"), ("funding", "8h"), ("open_interest", "1d"))


def refreeze_unread_holdout(lock_path: Path, *, split_utc: str, git_sha: str) -> HoldoutLock:
    """Een nieuwe split, alleen als het oude slot nooit is gelezen."""
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    require(payload.get("reads") == [],
            "Het poortsample is al gelezen; een nieuwe split zou die lezing ongedaan maken.",
            DataContractError, reads=payload.get("reads"))
    lock_path.unlink()
    return freeze_holdout(split_utc=split_utc, out=lock_path, git_sha=git_sha)


def programme_parameters(cfg: WeeklyMetaConfig) -> dict[str, Any]:
    return {"weekly_meta": cfg.model_dump(mode="json"), "M": cfg.planned_trials,
            "M_seed_total": 0, "M_registered_total": cfg.planned_trials}


def certified_data_hashes(
    register: DataRegister, symbols: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    out = []
    for dataset, gran in _SERIES:
        for sym in symbols:
            out.append((f"crypto/{dataset}/{sym}/{gran}",
                        register.certified_hash("crypto", dataset, sym, gran)))
    return tuple(sorted(out))


def book_and_freeze(
    *, spec_path: Path, cfg: WeeklyMetaConfig, register: DataRegister,
    ledger_path: Path, prereg_dir: Path, git_sha: str,
) -> Path:
    """Boek de geplande trials als eerste ledger-entry en bevries daarna de preregistratie."""
    ledger = HypothesisLedger(ledger_path)
    require(ledger.total_n_hypotheses() == 0,
            "De programmaboeking hoort de eerste entry van de ledger te zijn; er staat al iets in.",
            DataContractError, total=ledger.total_n_hypotheses())
    params = programme_parameters(cfg)
    hashes = certified_data_hashes(register, cfg.symbols)
    prereg = load_preregistration_spec(spec_path, data_hashes=hashes, parameters=params)
    require(prereg.planned_trials == cfg.planned_trials,
            "planned_trials in de preregistratie wijkt af van de config.", DataContractError,
            spec=prereg.planned_trials, config=cfg.planned_trials)
    ledger.append(LedgerEntry.from_config(
        wave=1, unit=PROGRAMME_UNIT, market="crypto", config=params, git_sha=git_sha,
        data_hash=hash_config(dict(hashes)), preregistration_id=prereg.preregistration_id,
        n_trials=cfg.planned_trials, result="interim",
        notes="Vier geplande trials, geboekt vóór de eerste fit: referentie, logreg, forest, ensemble."))
    return freeze_preregistration(prereg, git_sha=git_sha,
                                  ledger_total_at_freeze=ledger.total_n_hypotheses(),
                                  directory=prereg_dir)


def main(argv: Sequence[str]) -> None:
    require(list(argv) == ["freeze"], "Gebruik: python -m tradebot.registry.weekly_programme freeze",
            DataContractError)
    root = Path.cwd()
    cfg = weekly_meta_config()
    sha = current_git_sha()
    refreeze_unread_holdout(root / "artefacts/governance/holdout_lock.json",
                            split_utc=cfg.holdout_split_utc, git_sha=sha)
    path = book_and_freeze(
        spec_path=root / "conf/research/preregistration_weekly_meta.yaml", cfg=cfg,
        register=DataRegister(root / "artefacts/governance/data_hashes.json"),
        ledger_path=root / "artefacts/governance/hypothesis_ledger.json",
        prereg_dir=root / "artefacts/governance", git_sha=sha)
    print(f"holdout split {cfg.holdout_split_utc}; preregistratie {path.name}; M = {cfg.planned_trials}")


if __name__ == "__main__":
    main(sys.argv[1:])
