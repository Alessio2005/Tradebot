from __future__ import annotations

import joblib
import numpy as np

from tests.weekly_fixtures import synthetic_market
from tradebot.schemas.weekly_meta import weekly_meta_config
from tradebot.validation.weekly_final_model import fit_final_model, load_final_model

CFG = weekly_meta_config().model_copy(
    update={"forest_n_estimators": 60, "forest_min_samples_leaf": 5})


def test_the_final_model_round_trips_to_identical_probabilities(tmp_path) -> None:
    market = synthetic_market(n=700)
    model, columns, n_events = fit_final_model(market, CFG, k=1.0, d_star=0.4, cost_rt=0.0013)
    assert n_events > 50
    assert columns[-1] == "side" or "side" in columns
    path = tmp_path / "m.joblib"
    joblib.dump({"model": model, "feature_columns": columns}, path)
    loaded, loaded_columns = load_final_model(path)
    x = np.random.default_rng(0).normal(size=(20, len(columns)))
    assert loaded_columns == columns
    assert np.allclose(loaded.predict_proba(x), model.predict_proba(x))
    p = model.predict_proba(x)[:, 1]
    assert ((p > 0.0) & (p < 1.0)).all()
