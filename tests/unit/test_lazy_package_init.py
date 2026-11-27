"""`tradebot.risk` en `tradebot.schemas` laden hun submodules lui (PEP 562).

`from tradebot.risk import X` moet blijven werken voor elke naam in `__all__`, een onbekende
naam moet een gewone AttributeError geven, en `dir()` moet de luie namen tonen.
"""
from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize("package", ["tradebot.risk", "tradebot.schemas"])
def test_every_public_name_resolves_lazily(package):
    mod = importlib.import_module(package)
    for name in mod.__all__:
        assert getattr(mod, name) is not None
    assert set(mod.__all__) <= set(dir(mod))
    with pytest.raises(AttributeError, match="geen_naam"):
        _ = mod.geen_naam


def test_the_two_regimes_keep_their_meaning():
    from tradebot.risk import AccountRegime, Regime
    assert Regime.__module__ == "tradebot.risk.hmm_regime"
    assert AccountRegime.__module__ == "tradebot.risk.daily_loss_governor"
