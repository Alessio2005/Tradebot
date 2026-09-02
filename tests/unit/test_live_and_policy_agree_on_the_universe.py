# tests/unit/test_live_and_policy_agree_on_the_universe.py
"""De live-keten mag geen naam handelen die de policy niet kent. D1.

GEMETEN 2026-09-02, bij de voorbereiding van D1.

`apps/live_paper_trader.py` handelt VIJF namen; `conf/risk/default.yaml` is
geschreven voor ZES. Het verschil is BTCUSDT. Zie
`reports/phase7_divergence_map.md` §6.4.

WAT HIER WEL EN NIET WORDT AFGEDWONGEN
=======================================
Dat de policy zes namen kent en de live-keten er vijf handelt, is geen defect
dat een test kan oplossen. `gross_cap`, `concentration_cap` en `cluster_cap`
zijn grenzen over een BOEK, en dezelfde grenswaarden binden anders op vijf namen
dan op zes. Of dat boek uit vijf of zes namen hoort te bestaan, is een besluit
van de eigenaar van het risicoregime en geen implementatiekeuze; deze test
neemt dat besluit niet.

Wat wel een invariant is, en hier wordt vastgelegd: **elke naam die live wordt
verhandeld, moet in de policy voorkomen.** Een live-symbool zonder
cluster-toewijzing valt buiten `cluster_cap`; een live-symbool dat de policy
helemaal niet kent, valt buiten elke boekgrens die op naam werkt. Dat is de
kant die stilzwijgend naar GEEN limiet degradeert, en die verbiedt audit §23
(no-go 7).

De andere kant -- de policy kent een naam die live niet wordt verhandeld -- is
onschadelijk: een limiet op een positie die niet bestaat, bindt niet.

Ref: `reports/phase7_divergence_map.md` §6.4 en §6.5; fase-opdracht Stage D-1.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.schemas.config import RiskConfig, load_config

POLICY = load_config(ROOT / "conf" / "risk" / "default.yaml", RiskConfig)
LIVE_APPS = ("live_paper_trader.py", "live_trader.py", "paper_trade_runner.py")


def _symbols_declared_in(path: Path) -> set[str]:
    """Elke `_SYMBOLS = [...]`-toekenning op moduleniveau, statisch gelezen.

    Statisch en niet via import: die apps bouwen bij import een halve engine op.
    """
    if not path.exists():
        return set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if not names & {"_SYMBOLS", "SYMBOLS"}:
            continue
        if isinstance(node.value, ast.List | ast.Tuple):
            found |= {
                el.value for el in node.value.elts
                if isinstance(el, ast.Constant) and isinstance(el.value, str)
            }
    return found


class TestEveryLiveSymbolIsKnownToThePolicy:
    def test_the_detector_finds_the_live_universe(self) -> None:
        """Zonder dit is een lege verzameling niet te onderscheiden van succes."""
        symbols = _symbols_declared_in(ROOT / "apps" / "live_paper_trader.py")
        assert symbols, "geen `_SYMBOLS` gevonden; de detector leest niets"
        assert "ETHUSDT" in symbols

    def test_no_live_app_trades_a_symbol_the_policy_does_not_know(self) -> None:
        offenders: dict[str, set[str]] = {}
        for name in LIVE_APPS:
            traded = _symbols_declared_in(ROOT / "apps" / name)
            unknown = traded - set(POLICY.clusters)
            if unknown:
                offenders[name] = unknown
        assert not offenders, (
            f"Deze apps handelen namen die `conf/risk/default.yaml` niet kent: "
            f"{offenders}. Een live-symbool zonder cluster-toewijzing valt "
            f"buiten `cluster_cap`; audit §23 verbiedt stilzwijgend degraderen "
            f"naar geen limiet (no-go 7).")

    def test_every_live_symbol_has_a_cluster(self) -> None:
        """`cluster_cap` werkt op naam. Een naam zonder cluster telt nergens mee."""
        for name in LIVE_APPS:
            for symbol in _symbols_declared_in(ROOT / "apps" / name):
                assert POLICY.clusters.get(symbol), (
                    f"{name} handelt {symbol}, dat geen cluster heeft in "
                    f"`conf/risk/default.yaml`.")

    def test_the_detector_can_go_red(self, tmp_path: Path) -> None:
        """Negatieve controle op de AST-lezer zelf."""
        offender = tmp_path / "offender.py"
        offender.write_text(
            '_SYMBOLS = ["ETHUSDT", "DOGEUSDT"]\n', encoding="utf-8")
        traded = _symbols_declared_in(offender)
        assert traded == {"ETHUSDT", "DOGEUSDT"}
        assert traded - set(POLICY.clusters) == {"DOGEUSDT"}
