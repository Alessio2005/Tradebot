"""Fail-fast contract primitives — Phase 0, deliverable 2.

Doctrine (ARCHITECTUUR_AUDIT_2026-08-22, sectie 26 acceptance criterion 5, en
sectie 28): het stilzwijgend degraderen van een statistisch model naar een
naieve benadering is het grootste operationele risico in deze codebase. Een
contractschending moet daarom *luid en onmiddellijk* crashen.

Deze module levert de exception-hierarchie waarmee dat wordt afgedwongen:

    TradebotError                 (bestaand, src/tradebot/exceptions.py)
     └── TradebotContractError    basis voor elke contractschending
          ├── DependencyMissingError   harde afhankelijkheid ontbreekt
          ├── DataContractError        schema / PIT / immutability geschonden
          ├── ConfigContractError      configuratie ongeldig of onvolledig
          └── CausalityViolationError  lookahead / niet-causale operatie

REGEL
-----
Geen enkele van deze excepties mag ergens worden gevangen buiten een
top-level applicatie-entrypoint (`apps/*.py::main`). Een `except` die logt en
doorgaat is een overtreding; een `except` die een default teruggeeft is een
ernstiger overtreding. `scripts/audit_fallbacks.py` handhaaft dit statisch.
"""
from __future__ import annotations

from typing import Any, NoReturn, TypeVar

from ..exceptions import TradebotError

__all__ = [
    "CausalityViolationError",
    "has_module",
    "ConfigContractError",
    "DataContractError",
    "DependencyMissingError",
    "TradebotContractError",
    "require",
    "require_dependency",
    "unreachable",
]


class TradebotContractError(TradebotError):
    """Basis voor elke schending van een expliciet platform-contract."""


class DependencyMissingError(TradebotContractError):
    """Een harde afhankelijkheid ontbreekt.

    Nooit op te vangen met een fallback naar een naievere implementatie.
    Zie `docs/DEPENDENCY_CONTRACT.md` voor het gevolg per dependency.
    """


class DataContractError(TradebotContractError):
    """Een datacontract is geschonden.

    Schema, timestamp-standaard (UTC ns), append-only immutability van de PIT
    store, of een ontbrekende verplichte reeks.
    """


class ConfigContractError(TradebotContractError):
    """Een configuratiecontract is geschonden.

    Onbekende sleutel, ontbrekende sleutel, of een waarde buiten het
    gevalideerde bereik van het Pydantic-schema.
    """


class CausalityViolationError(TradebotContractError):
    """Een niet-causale operatie is gedetecteerd.

    Lookahead-lek, een scaler gefit op de volledige sample, een centered
    rolling window, of een gewicht dat op `t` rendeert in plaats van `t+1`.
    """


_E = TypeVar("_E", bound=BaseException)


def require(
    condition: object,
    message: str,
    exc_type: type[TradebotContractError] = TradebotContractError,
    **context: Any,
) -> None:
    """Raise `exc_type` wanneer `condition` falsy is.

    Geeft bewust *geen* boolean terug: een aanroeper mag de uitkomst niet
    kunnen negeren. Dit is het enige toegestane guard-idioom in `src/`.

    Parameters
    ----------
    condition : wordt op truthiness beoordeeld.
    message   : wat er is geschonden, en waarom dat fataal is.
    exc_type  : subklasse van `TradebotContractError`.
    **context : extra sleutel/waarde-paren die aan het bericht worden geplakt,
        zodat de crash zelfstandig diagnosticeerbaar is zonder debugger.

    Raises
    ------
    TradebotContractError
        Of de opgegeven subklasse daarvan.

    Examples
    --------
    >>> require(1 > 0, "sanity")
    >>> require(False, "vol-schatting ontbreekt", DataContractError, symbol="BTCUSDT")
    Traceback (most recent call last):
        ...
    tradebot.utils.failfast.DataContractError: vol-schatting ontbreekt (symbol=BTCUSDT)
    """
    if condition:
        return
    if not (isinstance(exc_type, type) and issubclass(exc_type, TradebotContractError)):
        raise TypeError(
            f"exc_type moet een TradebotContractError-subklasse zijn, kreeg {exc_type!r}"
        )
    if context:
        rendered = ", ".join(f"{k}={v!r}" if isinstance(v, str) else f"{k}={v}"
                             for k, v in sorted(context.items()))
        message = f"{message} ({rendered})"
    raise exc_type(message)


def require_dependency(module_name: str, *, needed_for: str, install_hint: str = "") -> Any:
    """Importeer `module_name` of crash met `DependencyMissingError`.

    Dit is de *enige* toegestane manier om een optioneel ogende import te doen.
    Het verschil met `try: import x except ImportError: fallback` is
    fundamenteel: hier bestaat geen gedegradeerd pad.

    Parameters
    ----------
    module_name  : volledige modulenaam, bijv. "hmmlearn.hmm".
    needed_for   : wat er kapot gaat zonder deze dependency (verschijnt in de
        crash, zodat de operator weet wat hij verliest).
    install_hint : optionele installatie-instructie.

    Returns
    -------
    De geimporteerde module.

    Raises
    ------
    DependencyMissingError
        Wanneer de module niet importeerbaar is.
    """
    import importlib
    import importlib.util

    top = module_name.split(".", maxsplit=1)[0]
    # find_spec() op een top-level naam retourneert None i.p.v. te raisen, zodat
    # deze module zelf nul try/except-blokken bevat (exit criterium 1).
    if importlib.util.find_spec(top) is None:
        hint = install_hint or f"pip install {top}"
        raise DependencyMissingError(
            f"Harde afhankelijkheid {module_name!r} ontbreekt. Benodigd voor: "
            f"{needed_for}. Er is GEEN fallback: dit platform degradeert niet "
            f"stilzwijgend naar een naievere benadering. Installeer met: {hint}"
        )
    return importlib.import_module(module_name)


def has_module(module_name: str) -> bool:
    """True wanneer `module_name` importeerbaar is, zonder hem te importeren.

    Dit is GEEN toegestaan alternatief voor `require_dependency`. Het is
    uitsluitend bedoeld voor **capability-probes waarbij afwezigheid betekent dat
    de capability feitelijk niet in gebruik is** - en er dus geen model, geen
    schatting en geen statistische claim wordt gedegradeerd.

    Toegestane voorbeelden:
      * `has_module("torch")` voordat een torch-RNG wordt geseed. Zonder torch
        bestaat er geen torch-model om te seeden; er degradeert niets.
      * `has_module("mlflow")` om te bepalen of naast de JSONL-ledger ook naar
        MLflow wordt gespiegeld. De JSONL-ledger is de autoriteit.

    Verboden voorbeeld:
      * `if has_module("hmmlearn"): hmm() else: ema()` - dat is precies de
        stille degradatie die D-10 beschrijft. Gebruik `require_dependency`.

    Gebruikt `importlib.util.find_spec`, zodat er geen try/except aan te pas komt.
    """
    import importlib.util

    return importlib.util.find_spec(module_name.split(".", maxsplit=1)[0]) is not None


def unreachable(message: str) -> NoReturn:
    """Markeer een tak die per constructie onbereikbaar hoort te zijn."""
    raise TradebotContractError(f"Onbereikbare tak bereikt: {message}")
