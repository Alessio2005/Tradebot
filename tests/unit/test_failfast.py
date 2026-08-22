"""Phase 0 / Stap 3 - contracttests voor de fail-fast fundering.

Geschreven VOORDAT de excepties ergens werden toegepast, conform de
stapsgewijze uitvoering in de Phase 0 master-prompt.
"""
from __future__ import annotations

import pytest

from tradebot.exceptions import TradebotError
from tradebot.utils.failfast import (
    CausalityViolationError,
    ConfigContractError,
    DataContractError,
    DependencyMissingError,
    TradebotContractError,
    require,
    require_dependency,
    unreachable,
)

CONTRACT_SUBCLASSES = [
    DependencyMissingError,
    DataContractError,
    ConfigContractError,
    CausalityViolationError,
]


class TestHierarchy:
    def test_base_is_a_tradebot_error(self) -> None:
        """De nieuwe tak hangt onder de bestaande hierarchie."""
        assert issubclass(TradebotContractError, TradebotError)

    @pytest.mark.parametrize("exc", CONTRACT_SUBCLASSES)
    def test_each_subclass_inherits_from_contract_base(self, exc: type) -> None:
        assert issubclass(exc, TradebotContractError)

    @pytest.mark.parametrize("exc", CONTRACT_SUBCLASSES)
    def test_subclasses_are_distinct(self, exc: type) -> None:
        """Een DataContractError mag niet per ongeluk een ConfigContractError zijn."""
        others = [e for e in CONTRACT_SUBCLASSES if e is not exc]
        assert not any(issubclass(exc, o) for o in others)


class TestRequire:
    def test_truthy_condition_is_a_no_op(self) -> None:
        assert require(True, "moet niet raisen") is None
        assert require(1, "moet niet raisen") is None
        assert require([0], "niet-lege lijst is truthy") is None

    @pytest.mark.parametrize("falsy", [False, 0, "", [], {}, None])
    def test_falsy_condition_raises(self, falsy: object) -> None:
        with pytest.raises(TradebotContractError):
            require(falsy, "contract geschonden")

    def test_returns_none_never_a_boolean(self) -> None:
        """Een aanroeper mag de uitkomst niet als vlag kunnen gebruiken."""
        assert require(True, "x") is None

    @pytest.mark.parametrize("exc", CONTRACT_SUBCLASSES)
    def test_raises_requested_subclass(self, exc: type) -> None:
        with pytest.raises(exc):
            require(False, "boem", exc)  # type: ignore[arg-type]

    def test_message_is_preserved(self) -> None:
        with pytest.raises(DataContractError, match="ontbrekende vol-schatting"):
            require(False, "ontbrekende vol-schatting", DataContractError)

    def test_context_is_rendered_into_the_message(self) -> None:
        """De crash moet zelfstandig diagnosticeerbaar zijn."""
        with pytest.raises(DataContractError) as ei:
            require(False, "gat in de reeks", DataContractError,
                    symbol="BTCUSDT", n_missing=3)
        msg = str(ei.value)
        assert "gat in de reeks" in msg
        assert "symbol='BTCUSDT'" in msg
        assert "n_missing=3" in msg

    def test_rejects_non_contract_exception_type(self) -> None:
        """require() mag geen willekeurige exception kunnen smokkelen."""
        with pytest.raises(TypeError):
            require(False, "x", ValueError)  # type: ignore[arg-type]

    def test_rejects_non_type_exc(self) -> None:
        with pytest.raises(TypeError):
            require(False, "x", "DataContractError")  # type: ignore[arg-type]


class TestRequireDependency:
    def test_returns_module_when_present(self) -> None:
        mod = require_dependency("json", needed_for="een test")
        assert hasattr(mod, "loads")

    def test_returns_submodule_when_present(self) -> None:
        mod = require_dependency("os.path", needed_for="een test")
        assert hasattr(mod, "join")

    def test_missing_dependency_crashes(self) -> None:
        with pytest.raises(DependencyMissingError) as ei:
            require_dependency(
                "definitely_not_installed_xyzzy",
                needed_for="3-state Gaussian HMM regime-classificatie",
            )
        msg = str(ei.value)
        assert "3-state Gaussian HMM" in msg, "moet zeggen WAT er kapot gaat"
        assert "GEEN fallback" in msg
        assert "pip install definitely_not_installed_xyzzy" in msg

    def test_install_hint_is_used(self) -> None:
        with pytest.raises(DependencyMissingError, match="pip install tradebot\\[hmm\\]"):
            require_dependency("definitely_not_installed_xyzzy",
                               needed_for="x", install_hint="pip install tradebot[hmm]")

    def test_no_silent_none_return(self) -> None:
        """Er bestaat geen codepad waarin dit None teruggeeft."""
        with pytest.raises(DependencyMissingError):
            require_dependency("definitely_not_installed_xyzzy", needed_for="x")


class TestUnreachable:
    def test_always_raises(self) -> None:
        with pytest.raises(TradebotContractError, match="Onbereikbare tak"):
            unreachable("deze tak hoort dood te zijn")


class TestModuleIsItselfFallbackFree:
    def test_failfast_module_contains_no_try_except(self) -> None:
        """De handhaver mag zelf geen fallback bevatten."""
        import ast
        import inspect

        import tradebot.utils.failfast as ff

        tree = ast.parse(inspect.getsource(ff))
        handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]
        assert handlers == [], "failfast.py mag nul except-handlers bevatten"
