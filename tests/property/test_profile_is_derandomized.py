"""Bewijst dat het derandomised Hypothesis-profiel daadwerkelijk BINDT.

Phase 6, stap 0 — sluit `reports/phase5_exit_report.md` §11.3.

WAAROM DEZE TEST BESTAAT
------------------------
`tests/conftest.py` registreert een profiel met ``derandomize=True`` en
``database=None``. Registreren is echter niet hetzelfde als toepassen, en de
manier waarop het mis kan gaan is stil:

  * `register_profile` zonder `load_profile` — het profiel bestaat en doet niets;
  * het profiel wordt geladen NA het importeren van de testmodules, waardoor de
    `@settings(...)`-decorators de oude defaults hebben ingevroren;
  * een per-test `@settings(...)` zet `derandomize` alsnog zelf.

In alle drie de gevallen blijft de suite groen terwijl de eigenschap die Phase 6
moest opleveren — reproduceerbaarheid — ontbreekt. Precies de klasse defect die
`reports/phase5_exit_report.md` §12 vijf keer in de eigen bewijsvoering vond.

Deze test kijkt daarom niet naar de registratie maar naar het EINDRESULTAAT: de
settings die de daadwerkelijk gedecoreerde property-tests dragen.
"""
from __future__ import annotations

import pytest

hypothesis = pytest.importorskip("hypothesis", reason="hypothesis not installed")

from tests.conftest import HYPOTHESIS_PROFILE_NAME

#: De tests waarvan §11.3 vermoedde dat hun non-derandomised configuratie de
#: onverklaarde zevende failure veroorzaakte.
_KERNEL_TESTS = (
    "test_cornish_fisher_var_is_negative",
    "test_har_rv_forecast_non_negative",
    "test_evt_gpd_var_less_than_cf_at_extreme",
    "test_microprice_between_bid_ask",
)


def test_the_loaded_profile_is_the_derandomised_one() -> None:
    """Het profiel is niet alleen geregistreerd maar ook geladen."""
    from hypothesis import settings

    assert settings._current_profile == HYPOTHESIS_PROFILE_NAME, (
        "het derandomised profiel is geregistreerd maar niet geladen; elke "
        "property-test draait dan alsnog op een procesafhankelijke seed"
    )


def test_the_default_settings_carry_derandomize_and_no_database() -> None:
    from hypothesis import settings

    assert settings.default.derandomize is True
    assert settings.default.database is None, (
        "de voorbeeldendatabase staat aan; de uitkomst van een run hangt dan "
        "af van wat een eerdere run op DEZE machine toevallig vond"
    )


@pytest.mark.parametrize("test_name", _KERNEL_TESTS)
def test_every_kernel_property_test_inherits_the_profile(test_name: str) -> None:
    """De decorator erft het profiel — dit is de eigenschap die telt.

    ``@settings(max_examples=..., deadline=...)`` neemt elk NIET opgegeven veld
    over van het profiel dat op decoratietijd geladen is. Deze test bewijst dat
    die overerving daadwerkelijk heeft plaatsgevonden voor de vier kernels.
    """
    from tests.property import test_hypothesis_kernels as mod

    fn = getattr(mod, test_name)
    inner_settings = fn._hypothesis_internal_use_settings
    assert inner_settings is not None, f"{test_name} draagt geen settings-object"
    assert inner_settings.derandomize is True, (
        f"{test_name} draait NIET derandomised; het profiel is te laat geladen "
        "of de decorator overschrijft het"
    )
    assert inner_settings.database is None, (
        f"{test_name} heeft alsnog een voorbeeldendatabase"
    )


def test_the_check_can_fail_for_the_right_reason() -> None:
    """Negatieve controle (§0.10): een niet-derandomised test wordt gedetecteerd.

    Zonder deze controle bewijst de bovenstaande parametrisatie niets — een
    assertie die nooit rood kan worden, meet niets. Hier wordt een functie
    gebouwd die het profiel EXPLICIET overschrijft, en aangetoond dat dezelfde
    inspectie hem afkeurt.
    """
    from hypothesis import given, settings
    from hypothesis import strategies as st

    @settings(derandomize=False, max_examples=1)
    @given(st.integers())
    def _not_derandomised(x: int) -> None:  # pragma: no cover - nooit gedraaid
        pass

    assert _not_derandomised._hypothesis_internal_use_settings.derandomize is False
