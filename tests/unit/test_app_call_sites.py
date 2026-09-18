"""De aanroepen in `apps/` moeten binden op de signatuur die zij aanroepen.

WAAROM DEZE TEST BESTAAT
========================
`apps/feature_selection.py` riep stage 2 zo aan::

    filter_by_mda(X=X_sfi, y=y, feature_names=sfi_names,
                  min_tstat=min_tstat_mda, n_repeats=10, seed=seed)

`selection/mda.py::filter_by_mda(X, mda_result)` neemt twee argumenten. De
aanroep gooide dus bij ELKE run `TypeError`, en de `except Exception` eronder
ving dat op als "MDA failed ... — using SFI features". Stage 2 heeft nooit
gedraaid en de log noemde het een mislukking in plaats van een programmeerfout.

Niets ving dat op. `apps/` heeft 0,0 % dekking (docs/CODE_REGISTER.md) en staat
niet in de DVC-DAG, dus geen enkele test kwam hier ooit langs. De typecontrole
zag het wel -- `[call-arg]` -- maar `ci.yml` stond op dat moment al rood om 187
andere redenen, dus het signaal verdronk.

HET WAS NIET DE ENIGE. Dezelfde bugklasse zat op drie andere plekken, en een
ervan schreef naar de hypothese-ledger:

    apps/alpha_combine.py:78       ICWeightedCombiner()   mist `signal_names`
    apps/live_paper_trader.py:329  PortfolioControllerConfig()  mist `constraints`
    apps/paper_trade_runner.py:343 idem
    apps/run_eq_units.py:54        LedgerEntry.from_config()  mist git_sha,
                                   data_hash en preregistration_id -- `--stage`
                                   heeft dus nooit een entry gestaged

WAT DEZE TEST DOET
==================
Statisch, zonder de apps te importeren of te draaien. Voor elke aanroep in
`apps/` van iets dat uit `tradebot.*` is geimporteerd -- een functie, een
klasse, of een classmethod daarop -- wordt geprobeerd de argumenten te binden op
de echte signatuur. Twee kanten:

  * een keyword dat de signatuur niet kent;
  * een verplicht argument dat ontbreekt.

Bindt het niet, dan is het een aanroep die bij uitvoering crasht.

Wat de test NIET beoordeelt, en bewust niet: aanroepen met `*args`- of
`**kwargs`-uitpakking, functies die zelf `**kwargs` accepteren, en dynamische
aanroepen. Die zijn statisch niet te beoordelen, en een test die dat wél claimt
zou meer beweren dan hij meet. Typen worden ook niet getoetst -- dat is het werk
van mypy; deze test dekt de aanroepen af die mypy op `apps/` zou zien, zodat het
signaal niet opnieuw verdrinkt als de typecontrole om een andere reden rood
staat.
"""
from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

_PLACEHOLDER = object()

ROOT = Path(__file__).resolve().parents[2]
APPS = ROOT / "apps"


def _imported_tradebot_names(tree: ast.Module) -> dict[str, str]:
    """Map lokale naam -> volledig gekwalificeerde naam, voor tradebot-imports."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("tradebot"):
            for alias in node.names:
                local = alias.asname or alias.name
                out[local] = f"{node.module}.{alias.name}"
    return out


def _resolve(qualname: str) -> object | None:
    parts = qualname.split(".")
    for split in range(len(parts) - 1, 0, -1):
        try:
            obj: object = importlib.import_module(".".join(parts[:split]))
        except Exception:
            continue
        for attr in parts[split:]:
            obj = getattr(obj, attr, None)
            if obj is None:
                return None
        return obj
    return None


def _offending_calls(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported = _imported_tradebot_names(tree)
    problems: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            shown, qualname = node.func.id, imported.get(node.func.id)
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            # `LedgerEntry.from_config(...)` — een classmethod op een
            # geimporteerde tradebot-klasse. Zonder deze tak viel juist die
            # aanroep erdoorheen, en het was er een naar de hypothese-ledger.
            base = imported.get(node.func.value.id)
            shown = f"{node.func.value.id}.{node.func.attr}"
            qualname = f"{base}.{node.func.attr}" if base else None
        else:
            continue
        if qualname is None:
            continue
        target = _resolve(qualname)
        if target is None or not callable(target):
            continue
        try:
            sig = inspect.signature(target)
        except (TypeError, ValueError):
            continue
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            continue  # **kwargs slikt alles; niets te toetsen
        if any(kw.arg is None for kw in node.keywords):
            continue  # `**d` uitpakken is niet statisch te beoordelen
        if any(isinstance(a, ast.Starred) for a in node.args):
            continue  # `*seq` idem

        kwnames = [kw.arg for kw in node.keywords if kw.arg is not None]
        unknown = [k for k in kwnames if k not in sig.parameters]
        if unknown:
            problems.append(
                f"{path.name}:{node.lineno}: {shown}(...) krijgt "
                f"{unknown}, maar {qualname} heeft {list(sig.parameters)}"
            )
            continue

        # Ontbrekende VERPLICHTE argumenten -- dezelfde bugklasse, andere kant.
        # `ICWeightedCombiner()` en `LedgerEntry.from_config(...)` vielen hier
        # doorheen zolang alleen de keyword-namen werden getoetst.
        try:
            sig.bind(*[_PLACEHOLDER] * len(node.args), **dict.fromkeys(kwnames, _PLACEHOLDER))
        except TypeError as exc:
            problems.append(f"{path.name}:{node.lineno}: {shown}(...) — {exc}")
    return problems


def _app_files() -> list[Path]:
    return sorted(p for p in APPS.glob("*.py") if p.name != "__init__.py")


@pytest.mark.parametrize("app", _app_files(), ids=lambda p: p.name)
def test_every_keyword_argument_binds_on_the_real_signature(app: Path) -> None:
    problems = _offending_calls(app)
    assert not problems, (
        "Deze aanroepen binden niet op de signatuur die zij aanroepen en zouden "
        "bij uitvoering crashen:\n  " + "\n  ".join(problems)
    )


def test_the_guard_can_go_red() -> None:
    """De negatieve controle: op een aanroep die niet bindt, MOET hij afgaan."""
    broken = APPS / "_tmp_call_site_probe.py"
    broken.write_text(
        "from tradebot.selection.mda import filter_by_mda\n"
        "from tradebot.registry import LedgerEntry\n"
        "filter_by_mda(X=None, y=None, feature_names=[], min_tstat=2.0)\n"
        "LedgerEntry.from_config(wave=1, unit='u', market='m', config={})\n",
        encoding="utf-8",
    )
    try:
        problems = _offending_calls(broken)
    finally:
        broken.unlink()
    assert problems, "de wachter ziet een aanroep die niet bindt niet -- hij meet niets"
    joined = "\n".join(problems)
    assert "filter_by_mda" in joined, "de keyword-tak meet niets"
    assert "LedgerEntry.from_config" in joined, "de classmethod-tak meet niets"
