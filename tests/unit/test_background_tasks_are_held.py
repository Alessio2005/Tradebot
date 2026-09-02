# tests/unit/test_background_tasks_are_held.py
"""Geen achtergrondtaak zonder sterke referentie. DI-7, Stage D-2.

WAAROM DIT EEN CONTRACT IS EN GEEN STIJLREGEL
==============================================
`asyncio` houdt zelf alleen een ZWAKKE referentie naar een draaiende task. De
sterke referentie is de returnwaarde van `create_task`. Gooit de aanroeper die
weg — `asyncio.create_task(coro())` als losse expressie — dan mag de garbage
collector de taak op elk moment opruimen. De CPython-documentatie zegt dit
letterlijk, en het gebeurt in de praktijk juist onder geheugendruk: precies
wanneer het systeem het al zwaar heeft.

In dit project raakte dat vijf plekken (gemeten 2026-09-01):

* vier in `live/feed.py` — de replay-, aggtrade-, bookticker- en fundinglus.
  Verdwijnt daar een taak, dan stopt de koersstroom zonder foutmelding en
  handelt de engine door op de laatst bekende bar;
* één in `monitoring/sharpe_monitor.py` — de taak die de circuit breaker laat
  afgaan. Verdwijnt die, dan komt de HALT niet, op het moment dat hij nodig is.

De fase-opdracht noemt er vijf "in `live/`"; het zijn er vier in `live/` en één
in `monitoring/`. Zie `reports/phase7_divergence_map.md` §7.

Ref: fase-opdracht Stage D-2 (DI-7); `docs/DEFERRED_ISSUES.md` DI-7.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "tradebot"


def _is_create_task(node: ast.AST) -> bool:
    """`asyncio.create_task(...)` of `loop.create_task(...)`."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return isinstance(func, ast.Attribute) and func.attr in {
        "create_task", "ensure_future",
    }


def discarded_tasks(path: Path) -> list[str]:
    """Elke `create_task`-aanroep waarvan de returnwaarde wordt weggegooid.

    Een aanroep telt als weggegooid wanneer hij de VOLLEDIGE expressie van een
    `Expr`-statement is: `asyncio.create_task(x())` op een eigen regel. Wordt de
    waarde toegekend, geretourneerd, ge-await of in een container gestopt, dan
    is er een sterke referentie en is er niets aan de hand.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    try:
        label = path.relative_to(ROOT).as_posix()
    except ValueError:          # een wegwerpbestand uit `tmp_path`
        label = path.name
    return [
        f"{label}:{node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and _is_create_task(node.value)
    ]


class TestNoTaskIsCreatedAndDropped:
    def test_no_module_in_src_discards_a_task(self) -> None:
        offenders = [
            hit for path in sorted(SRC.rglob("*.py"))
            for hit in discarded_tasks(path)
        ]
        assert not offenders, (
            f"Deze aanroepen gooien hun task weg: {offenders}. `asyncio` houdt "
            f"alleen een zwakke referentie; de GC mag de taak dan opruimen. "
            f"DI-7.")

    def test_the_feed_holds_every_task_it_starts(self) -> None:
        """De feed moet zijn taken niet alleen vasthouden maar ook opruimen."""
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from tradebot.live import feed as feed_module

        source = ast.parse(Path(feed_module.__file__).read_text(encoding="utf-8"))
        names = {
            node.name for node in ast.walk(source)
            if isinstance(node, ast.FunctionDef)
        }
        assert "_spawn" in names, (
            "De feed hoort zijn taken via één plek te starten, zodat er één "
            "plek is die de referentie vasthoudt.")
        assert "_on_task_done" in names, (
            "Een taak die valt terwijl niemand hem await, verdwijnt in stilte. "
            "Er hoort een done-callback te zijn die dat zichtbaar maakt.")


class TestTheDetectorCanGoRed:
    """Zonder deze twee is de test hierboven een poort die niets bewaakt."""

    def test_a_discarded_task_is_caught(self, tmp_path: Path) -> None:
        offender = tmp_path / "offender.py"
        offender.write_text(
            "import asyncio\n"
            "async def go() -> None:\n"
            "    asyncio.create_task(other())\n",
            encoding="utf-8")
        assert discarded_tasks(offender)

    def test_a_held_task_is_not_caught(self, tmp_path: Path) -> None:
        clean = tmp_path / "clean.py"
        clean.write_text(
            "import asyncio\n"
            "async def go() -> None:\n"
            "    task = asyncio.create_task(other())\n"
            "    self._tasks.add(task)\n",
            encoding="utf-8")
        assert not discarded_tasks(clean)
