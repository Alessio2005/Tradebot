"""De negatieve controle op `scripts/check_banned_methods.py` — Stage B-5.

Een ratchet die nul treffers meldt, zegt twee dingen tegelijk: *"er is geen
overtreding"* en *"ik kijk niet"*. Alleen dit bestand kan die twee uit elkaar
houden.

Twee eigenschappen worden bewaakt, en de tweede is de belangrijkste:

1. De scanner VINDT elke verboden constructie. Zonder dat is hij decoratie.
2. De scanner vindt ze ALLEEN in uitvoerbare code. Een AST-scanner die ook
   docstrings en commentaar vlagt, produceert vals alarm — en een ratchet met
   vals alarm wordt uitgezet. Vals alarm is hier duurder dan een gemist geval,
   omdat het de hele gate kost in plaats van één regel.

Punt 2 is niet hypothetisch: de eerste versie van de scanner vlagde
`selection/mda.py::rng.shuffle(block_indices)` — een blok-permutatie met embargo
voor MDA-feature-importance, de correcte techniek. `shuffle` is daarna beperkt
tot de import uit `sklearn`.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "check_banned_methods", ROOT / "scripts" / "check_banned_methods.py")
assert _SPEC and _SPEC.loader
scanner = importlib.util.module_from_spec(_SPEC)
# Registreren VOOR exec_module: `@dataclass` zoekt zijn eigen module op in
# `sys.modules` om annotaties te kunnen resolven, en krijgt anders None.
sys.modules[_SPEC.name] = scanner
_SPEC.loader.exec_module(scanner)


def _scan(tmp_path: Path, body: str) -> list:
    path = tmp_path / "candidate.py"
    path.write_text(body, encoding="utf-8")
    return scanner.scan_file(path, root=tmp_path)


# --------------------------------------------------------------------------- #
# De scanner vindt wat hij moet vinden
# --------------------------------------------------------------------------- #
VIOLATIONS = {
    "kfold_import": "from sklearn.model_selection import KFold\n",
    "kfold_call": "import sklearn.model_selection as m\nx = m.KFold(n_splits=5)\n",
    "shuffle_split": "from sklearn.model_selection import ShuffleSplit\n",
    "train_test_split": "from sklearn.model_selection import train_test_split\n",
    "grid_search": "from sklearn.model_selection import GridSearchCV\n",
    "cross_val_score": "from sklearn.model_selection import cross_val_score\n",
    "timeseries_split": "from sklearn.model_selection import TimeSeriesSplit\n",
    "sklearn_shuffle": "from sklearn.utils import shuffle\n",
    "shuffle_kwarg": "def f(**kw): ...\nf(shuffle=True)\n",
    "shuffle_kwarg_on_a_correct_splitter": (
        "from tradebot.cv.walk_forward import WalkForwardCV\n"
        "cv = WalkForwardCV(train_size=1, test_size=1, shuffle=True)\n"),
}


class TestTheScannerFindsEveryBannedConstruct:
    @pytest.mark.parametrize("body", VIOLATIONS.values(), ids=list(VIOLATIONS))
    def test_it_is_caught(self, tmp_path: Path, body: str) -> None:
        findings = _scan(tmp_path, body)
        assert findings, (
            f"de scanner vond niets in:\n{body}\nEen ratchet die dit doorlaat, "
            f"meldt nul treffers omdat hij niet kijkt.")

    def test_the_message_names_the_construct_and_the_reason(
        self, tmp_path: Path
    ) -> None:
        """Wie de gate rood ziet, moet niet hoeven zoeken waarom."""
        finding = _scan(tmp_path, VIOLATIONS["kfold_import"])[0]
        rendered = finding.render()
        assert "KFold" in rendered
        assert "RANDOM_CV" in rendered
        assert "tijd" in rendered

    def test_strict_mode_exits_non_zero_on_a_violation(
        self, tmp_path: Path
    ) -> None:
        """De ratchet moet ook echt een exit code geven, niet alleen printen."""
        (tmp_path / "bad.py").write_text(
            VIOLATIONS["kfold_import"], encoding="utf-8")
        rc = scanner.main(["--strict", "--root", str(tmp_path), str(tmp_path)])
        assert rc == 1

    def test_without_strict_it_reports_but_does_not_fail(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "bad.py").write_text(
            VIOLATIONS["kfold_import"], encoding="utf-8")
        assert scanner.main(["--root", str(tmp_path), str(tmp_path)]) == 0


# --------------------------------------------------------------------------- #
# En alleen dat — geen vals alarm
# --------------------------------------------------------------------------- #
CLEAN = {
    "docstring": '"""Wij gebruiken GEEN KFold of ShuffleSplit; zie audit 17.1."""\n',
    "comment": "# TimeSeriesSplit is hier verboden, gebruik purged_walk_forward\n",
    "string_literal": 'REPORT = {"train_test_split": "Purged CPCV with 10 folds"}\n',
    "variable_name": "KFold = None\ntrain_test_split = 3\n",
    "numpy_generator_shuffle": (
        "import numpy as np\n"
        "rng = np.random.default_rng(42)\n"
        "blocks = [1, 2, 3]\n"
        "rng.shuffle(blocks)\n"),
    "shuffle_false": "def f(**kw): ...\nf(shuffle=False)\n",
    "the_correct_splitter": (
        "from tradebot.validation.walk_forward import purged_walk_forward\n"),
}


class TestTheScannerDoesNotCryWolf:
    @pytest.mark.parametrize("body", CLEAN.values(), ids=list(CLEAN))
    def test_clean_code_is_clean(self, tmp_path: Path, body: str) -> None:
        findings = _scan(tmp_path, body)
        assert findings == [], (
            f"vals alarm op:\n{body}\nGevonden: "
            f"{[f.render() for f in findings]}\nEen ratchet met vals alarm wordt "
            f"uitgezet, en dan bewaakt hij niets meer.")

    def test_the_mda_block_permutation_is_not_flagged(self) -> None:
        """Het concrete geval waarop de eerste versie viel.

        `selection/mda.py` shuffelt BLOKKEN met embargo om
        feature-importance te meten. Dat is López de Prado's MDA en het heeft
        niets met een train/test-splitsing te maken.
        """
        findings = scanner.scan_file(
            ROOT / "src" / "tradebot" / "selection" / "mda.py", root=ROOT)
        assert findings == [], [f.render() for f in findings]

    def test_this_scanner_does_not_flag_itself(self) -> None:
        """Hij noemt elke verboden naam in zijn eigen tabel en docstring."""
        findings = scanner.scan_file(
            ROOT / "scripts" / "check_banned_methods.py", root=ROOT)
        assert findings == [], [f.render() for f in findings]


# --------------------------------------------------------------------------- #
# De ratchet zelf
# --------------------------------------------------------------------------- #
class TestTheRatchet:
    def test_the_allowlist_is_empty(self) -> None:
        """Elk item zou een pad zijn waarlangs een niet-temporele splitsing een
        promotiebesluit kan bereiken. Dat hoort een expliciet besluit te zijn."""
        assert scanner.ALLOWLIST == {}, (
            f"de allowlist is niet langer leeg: {scanner.ALLOWLIST}. Leg het "
            f"besluit vast in docs/ARCHITECTURAL_DECISIONS.md.")

    def test_the_repository_is_clean_today(self) -> None:
        """Exit-criterium B5: nul treffers over src/, apps/ en scripts/."""
        assert scanner.main(["--strict"]) == 0
