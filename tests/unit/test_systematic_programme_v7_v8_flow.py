"""De commandoregel van v7 en v8 (`programme_v7.main`, `programme_v8.main`).

v7 en v8 zijn `v6.Programme`-objecten: hun orkestratie (bevriezen, W_DEV, OOS-lezing, vooruitlezing)
is die van v6 en wordt getoetst in `test_systematic_programme_v6_flow.py` en in de smoke-toetsen van
v7 en v8. Wat ze zelf hebben is het CLI dat die orkestratie voor HUN programma aanroept, en dat is
wat hier staat: elk commando roept precies één v6-stap, met `prog=` het eigen programma (niet dat
van v6), en een ontbrekend of onbekend commando weigert voordat er iets draait.
"""
from __future__ import annotations

import json
import runpy
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tradebot.systematic import programme_v6 as v6
from tradebot.systematic import programme_v7 as v7
from tradebot.systematic import programme_v8 as v8
from tradebot.utils.failfast import DataContractError

PROGRAMMES = [pytest.param(v7, v7.V7, "programme_v7", id="v7"),
              pytest.param(v8, v8.V8, "programme_v8", id="v8")]


def _fake_steps(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Vervang de vier v6-stappen: ze onthouden hoe ze werden aangeroepen en geven een vast antwoord."""
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake(name: str, result: Any) -> Callable[..., Any]:
        def call(*args: Any, **kw: Any) -> Any:
            calls.append((name, {"args": args, **kw}))
            return result
        return call

    monkeypatch.setattr(v6, "freeze", fake("freeze", Path("artefacts/prereg.json")))
    monkeypatch.setattr(v6, "run_programme", fake("run", {"selected": "S2", "score": 1.5}))
    monkeypatch.setattr(v6, "read_oos", fake("oos", {"verdict": "archived"}))
    monkeypatch.setattr(v6, "read_forward", fake("forward", {"verdict": "eligible_for_capital"}))
    return calls


@pytest.mark.parametrize(("command", "step", "printed"), [
    ("freeze", "freeze", "artefacts/prereg.json\n"),
    ("run", "run", json.dumps({"selected": "S2", "score": 1.5}, indent=2) + "\n"),
    ("oos", "oos", json.dumps({"verdict": "archived"}, indent=2) + "\n"),
    ("forward", "forward", json.dumps({"verdict": "eligible_for_capital"}, indent=2) + "\n"),
])
@pytest.mark.parametrize(("module", "programme", "_name"), PROGRAMMES)
def test_main_runs_exactly_the_requested_step_for_its_own_programme(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], module: ModuleType,
        programme: v6.Programme, _name: str, command: str, step: str, printed: str) -> None:
    calls = _fake_steps(monkeypatch)
    module.main([command])
    assert calls == [(step, {"args": (), "prog": programme})]
    assert capsys.readouterr().out == printed


def test_the_two_programmes_are_not_the_same_programme() -> None:
    assert v7.V7 is not v8.V8 and v7.V7.name == "robust_book_v7" and v8.V8.name == "robust_book_v8"


@pytest.mark.parametrize("argv", [[], ["nonsense"], ["freeze", "run"], ["FREEZE"], ["oos-resume"]])
@pytest.mark.parametrize(("module", "programme", "name"), PROGRAMMES)
def test_main_refuses_bad_usage_before_running_anything(
        monkeypatch: pytest.MonkeyPatch, module: ModuleType, programme: v6.Programme, name: str,
        argv: list[str]) -> None:
    calls = _fake_steps(monkeypatch)
    usage = rf"Gebruik: python -I -m tradebot\.systematic\.{name} \{{freeze\|run\|oos\|forward\}}"
    with pytest.raises(DataContractError, match=usage):
        module.main(argv)
    assert calls == []


@pytest.mark.filterwarnings("ignore:.*found in sys.modules after import of package:RuntimeWarning")
@pytest.mark.parametrize(("module", "programme", "name"), PROGRAMMES)
def test_the_module_runs_as_a_script_and_hands_its_arguments_to_main(
        monkeypatch: pytest.MonkeyPatch, module: ModuleType, programme: v6.Programme,
        name: str) -> None:
    calls = _fake_steps(monkeypatch)
    monkeypatch.setattr(sys, "argv", [name, "oos"])
    runpy.run_module(f"tradebot.systematic.{name}", run_name="__main__")
    assert [step for step, _ in calls] == ["oos"]
    # Het scriptpad bouwt zijn eigen `Programme` (de module wordt opnieuw uitgevoerd): hetzelfde
    # programma, niet hetzelfde object.
    ran = calls[0][1]["prog"]
    assert (ran.name, ran.wave, ran.candidates) == (programme.name, programme.wave, programme.candidates)
    monkeypatch.setattr(sys, "argv", [name])
    with pytest.raises(DataContractError, match="Gebruik: python -I -m"):
        runpy.run_module(f"tradebot.systematic.{name}", run_name="__main__")
