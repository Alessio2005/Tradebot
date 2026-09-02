"""tests/conftest.py — pytest fixtures + path setup.

Ensures src/ and repo root are importable from any test location.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC       = _REPO_ROOT / "src"

for p in (_SRC, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# --------------------------------------------------------------------------- #
# Hypothesis — Phase 6, stap 0.  Sluit `reports/phase5_exit_report.md` §11.3.
# --------------------------------------------------------------------------- #
# Phase 5 legde vast dat één tussentijdse suite-run op `b206895` **7** failures
# rapporteerde zonder namen, terwijl elf volledige runs er exact 6 gaven. Het
# onderzoek daar kon de zevende niet reproduceren en noteerde als meest
# waarschijnlijke — maar onbewezen — verklaring de non-derandomised
# Hypothesis-configuratie: een zeldzaam tegenvoorbeeld dat de random search
# doorgaans niet vindt.
#
# Dit profiel maakt die verklaring toetsbaar door beide bronnen van
# run-to-run-variatie weg te nemen:
#
#   derandomize=True  de seed wordt deterministisch afgeleid uit de testnaam in
#                     plaats van uit de entropiebron van het proces. Dezelfde
#                     test genereert daardoor in elke run dezelfde voorbeelden.
#
#   database=None     de voorbeeldendatabase in `.hypothesis/` is lokale,
#                     niet-versiebeheerde state (`.gitignore` regel 24). Met de
#                     database aan hangt de uitkomst van een run af van wat een
#                     EERDERE run op DEZE machine toevallig heeft gevonden —
#                     precies het soort verborgen toestand waardoor "6 op mijn
#                     machine, 7 op de jouwe" ontstaat. Uit betekent dat een
#                     failure die deze suite rapporteert, door de derandomised
#                     search zélf is gevonden en dus overal reproduceerbaar is.
#
# Een per-test `@settings(...)`-decorator erft elk veld dat hij niet zelf zet
# van het geladen profiel. `tests/property/test_hypothesis_kernels.py` zet
# alleen `max_examples` en `deadline`, dus `derandomize` en `database` komen
# hiervandaan. `tests/property/test_profile_is_derandomized.py` bewijst dat die
# overerving daadwerkelijk plaatsvindt in plaats van het aan te nemen.
#
# `HYPOTHESIS_PROFILE=dev` zet de database terug aan voor lokaal debuggen
# (shrink-replay van een gevonden tegenvoorbeeld). Dat profiel is NIET de
# default, juist omdat het de reproduceerbaarheid opgeeft die hierboven wordt
# gekocht.
try:  # pragma: no cover - hypothesis is een dev-dependency, geen src-dependency
    from hypothesis import settings as _hyp_settings
except ImportError:  # pragma: no cover
    _hyp_settings = None  # type: ignore[assignment]

#: Naam van het profiel dat de suite standaard draait.
HYPOTHESIS_PROFILE_NAME = "tradebot"

if _hyp_settings is not None:
    _hyp_settings.register_profile(
        HYPOTHESIS_PROFILE_NAME,
        derandomize=True,
        database=None,
        print_blob=True,
    )
    _hyp_settings.register_profile(
        "dev",
        derandomize=False,
        print_blob=True,
    )
    _hyp_settings.load_profile(
        os.environ.get("HYPOTHESIS_PROFILE", HYPOTHESIS_PROFILE_NAME)
    )


@pytest.fixture(autouse=True)
def _isolate_cb_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Wave 15 P0-5.5: redirect the circuit-breaker trip log to a tmp dir.

    Prevents production ``artefacts/circuit_breaker.log`` from affecting
    unit/integration tests that instantiate CircuitBreaker without an explicit
    ``cb_log_path`` in their config.

    STAGE D, C2: hetzelfde geldt nu voor de SOEVEREINE halt-state. Zonder deze
    tweede omleiding zou een test die de breaker laat trippen een echte
    `artefacts/risk/halt_state.json` achterlaten (no-go 15) -- en omdat die
    toestand per ontwerp NIET verloopt, zou elke volgende `CircuitBreaker` in de
    suite daarna weigeren te starten. Een testartefact zou dan de hele suite
    stilleggen, en de volgende ontwikkelaar zou dat niet op een test wijten.
    """
    import tradebot.live.circuit_breaker as _cb_module

    monkeypatch.setattr(
        _cb_module, "_CB_LOG_PATH", tmp_path / "circuit_breaker.log"
    )
    monkeypatch.setattr(
        _cb_module, "_HALT_STORE_PATH", tmp_path / "risk" / "halt_state.json"
    )
