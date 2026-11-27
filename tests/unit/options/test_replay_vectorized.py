"""De replay geeft met `pricing.vectorized` aan en uit dezelfde uitkomst (`summary.json`).

Een kleine synthetische markt met een ex-dividenddag in de looptijd: de BAW-kernel, de
FD-startwaarde en het oude pad contract voor contract komen alle drie aan bod.
"""
from __future__ import annotations

import asyncio
import json
from datetime import date, time

from tests.unit.options.market_helpers import CFG
from tests.unit.options.synthetic_market import SyntheticSpec, SyntheticUnderlying, generate
from tradebot.options.engine.replay import run_replay

SPEC = SyntheticSpec(underlyings=(SyntheticUnderlying("TSTB", 50.0, strike_step=1.0,
                                                      dividend_day=1, dividend=0.3),),
                     start=date(2026, 12, 14), n_days=2, snapshot_times=(time(10, 0),),
                     strike_range=0.06, max_days=40, warmup_days=60)


def _summary(vectorized: bool, workers: int | str, out):
    pricing = CFG.pricing.model_copy(update={"vectorized": vectorized, "workers": workers})
    cfg = CFG.model_copy(update={"pricing": pricing})
    asyncio.run(run_replay(cfg, generate(SPEC), out))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary.pop("journal") == str(out / "journal.jsonl")
    return summary


def test_vectorized_on_and_off_give_the_same_summary(tmp_path):
    on = _summary(True, CFG.pricing.workers, tmp_path / "on")
    off = _summary(False, 1, tmp_path / "off")
    assert on == off
    assert on["positions_opened"] >= 1 and not on["halted"]
