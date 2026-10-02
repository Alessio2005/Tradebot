"""Het eindmodel van de wekelijkse strategie: één fit op de hele ontwikkelsample, met zijn verdict.

Dit leest de holdout NIET. Het fit het gepreregistreerde ensemble (logreg + forest,
Platt-gekalibreerd) op alle ontwikkelevents met de `k` en `d_star` uit het
campagne-artefact, en schrijft het model naast een kaart die het verdict van die
campagne draagt. Een model met een ander verdict dan PASS is een onderzoeksartefact en
wordt nergens gepromoveerd; de kaart zegt dat met zoveel woorden.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..data.weekly_market import WeeklyMarket, load_weekly_market
from ..features.registry import current_git_sha
from ..schemas.weekly_meta import WeeklyMetaConfig, weekly_meta_config
from ..train.light_models import CalibratedModel, fit_light_model
from ..train.weekly_dataset import build_weekly_dataset
from ..utils.failfast import DataContractError, require
from .holdout import development_slice

__all__ = ["fit_final_model", "load_final_model", "main"]

MODEL_PATH = "artefacts/models/weekly_meta_ensemble.joblib"
CARD_PATH = "artefacts/models/weekly_meta_ensemble.json"


def fit_final_model(
    market: WeeklyMarket, cfg: WeeklyMetaConfig, *, k: float, d_star: float, cost_rt: float,
) -> tuple[CalibratedModel, list[str], int]:
    """Fit op alle events van `market`; geeft het model, zijn featurekolommen en het aantal events."""
    wd = build_weekly_dataset(market, cfg, k=k, d_star=d_star, cost_rt=cost_rt)
    ds = wd.dataset
    rows = np.arange(len(ds.target), dtype=np.int64)
    model = fit_light_model(ds, rows, ds.target, "ensemble", cfg)
    return model, [str(c) for c in ds.features.columns], int(rows.size)


def load_final_model(path: Path) -> tuple[CalibratedModel, list[str]]:
    payload = joblib.load(path)
    return payload["model"], list(payload["feature_columns"])


def main() -> None:
    root = Path.cwd()
    cfg = weekly_meta_config()
    campaign = json.loads((root / "artefacts/governance/weekly_meta_campaign.json")
                          .read_text(encoding="utf-8"))
    lock = root / "artefacts/governance/holdout_lock.json"
    require(json.loads(lock.read_text(encoding="utf-8"))["reads"] == [],
            "Het poortsample is al gelezen; het eindmodel hoort er niets van te zien.",
            DataContractError)
    import pandas as pd
    full = load_weekly_market(root, cfg.symbols)
    dev_index = development_slice(pd.DataFrame(index=full.grid), lock_path=lock).index
    market = full.truncate(dev_index[-1] + pd.Timedelta(hours=1))
    model, columns, n_events = fit_final_model(
        market, cfg, k=float(campaign["k"]), d_star=float(campaign["d_star"]),
        cost_rt=float(campaign["costs"]["round_trip"]))
    out = root / MODEL_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": columns}, out)
    card: dict[str, Any] = {
        "model": "ensemble (logreg + forest, Platt)", "k": campaign["k"],
        "d_star": campaign["d_star"], "n_events_fitted": n_events,
        "feature_columns": columns, "fitted_through": str(market.grid[-1]),
        "preregistration_id": campaign["preregistration_id"],
        "campaign_verdict": campaign["verdict"]["status"],
        "campaign_binding": campaign["verdict"]["binding"],
        "promoted": campaign["verdict"]["status"] == "PASS",
        "holdout_read": False, "git_sha": current_git_sha(),
        "model_file": MODEL_PATH, "model_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "note": ("Onderzoeksartefact. Alleen een PASS-verdict maakt dit model een kandidaat; "
                 "elk ander verdict betekent dat het niet wordt gebruikt om te handelen."),
    }
    (root / CARD_PATH).write_text(json.dumps(card, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
    print(json.dumps({k: card[k] for k in ("campaign_verdict", "promoted", "n_events_fitted")}))


if __name__ == "__main__":
    sys.exit(main())
