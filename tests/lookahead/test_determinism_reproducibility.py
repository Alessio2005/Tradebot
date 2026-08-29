"""D-1 POORT 6 — DETERMINISME EN REPRODUCEERBAARHEID.

    Dezelfde invoer, dezelfde code, dezelfde uitvoer. Bit voor bit. Ook in een
    ander proces, ook in een andere importvolgorde, ook met een andere
    PYTHONHASHSEED.

Deze poort is de fundering onder de vijf andere. Elke invariantietoets
vergelijkt twee berekeningen; is de berekening zelf niet deterministisch, dan
meten die vergelijkingen ruis en is een groene uitslag toeval. Een lek dat maar
in één op de tien runs zichtbaar is, is met een niet-deterministische pijplijn
niet te onderscheiden van een geslaagde toets.

DE DRIE MANIEREN WAAROP DIT PLATFORM NIET-DETERMINISTISCH KAN WORDEN
====================================================================
1. **Ongeseede randomness** in een feature, een fit of een bootstrap. Direct
   zichtbaar bij herhaling binnen één proces.
2. **PYTHONHASHSEED.** De hash van een `str` verschilt per proces. Loopt er
   ergens een `set` of een `dict` mee in een volgorde die het RESULTAAT bepaalt
   — een kolomvolgorde, een hash over een geïtereerde verzameling — dan
   verandert een artefacthash tussen twee runs zonder dat er iets aan de data of
   de code is gewijzigd. Binnen één proces is dit onzichtbaar.
3. **Importvolgorde.** Gemeten in Stage B-2 en geen hypothese: er stond een
   importcyclus tussen `execution/order_router.py` en `backtest/engine.py` die
   alleen toesloeg wanneer `order_router` als eerste werd geïmporteerd. De
   volledige suite liep groen omdat daar altijd wel iets `tradebot.backtest`
   eerder importeerde; `pytest tests/lookahead` crashte bij collectie. Dat is
   precies de deelverzameling die `research_gates.yml` moet draaien.

   `TestImportOrderIsIrrelevant` is de regressietest daarop.

WAAROM SUBPROCESSEN
===================
Punt 2 en 3 zijn per definitie niet binnen één interpreter te meten. Een test
die alleen in het lopende proces kijkt, kan determinisme over processen heen
niet bevestigen — en het artefact dat morgen in CI wordt vergeleken, is in een
ander proces gemaakt.

Ref: audit §7.2, §26; `fase_2_research_falsification.md` deliverable 6.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from tradebot.features.base import BaseFeature, InputSpec
from tradebot.features.registry import build_default_registry, current_git_sha

from .d1_harness import FEAT_CFG, ROOT, SYMBOLS, certified_frame, requires_store

pytestmark = pytest.mark.lookahead

REGISTRY = build_default_registry(FEAT_CFG)
FEATURE_IDS = [f.feature_id for f in REGISTRY.features]

#: Twee verschillende, expliciet gezette hash-seeds. `random` zou het probleem
#: soms verbergen; twee vaste, verschillende waarden nooit.
HASH_SEEDS = ("0", "982451653")


def _run(code: str, *, hash_seed: str) -> str:
    """Draai `code` in een VERS interpreterproces en geef stdout terug."""
    import os

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = hash_seed
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600,
        check=False,
    )
    assert proc.returncode == 0, (
        f"subprocess faalde (PYTHONHASHSEED={hash_seed}):\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    return proc.stdout.strip()


# --------------------------------------------------------------------------- #
# Binnen één proces
# --------------------------------------------------------------------------- #
@requires_store
class TestRepeatedComputationIsBitIdentical:
    @pytest.mark.parametrize("feature", REGISTRY.features, ids=FEATURE_IDS)
    def test_a_feature_gives_the_same_answer_twice(
        self, feature: BaseFeature
    ) -> None:
        source = certified_frame("BTCUSDT")
        first = feature.transform(source).values.to_numpy(dtype="float64")
        second = feature.transform(source).values.to_numpy(dtype="float64")
        assert (np.isnan(first) == np.isnan(second)).all()
        finite = ~np.isnan(first)
        assert (first[finite] == second[finite]).all(), (
            f"{feature.feature_id} levert bij herhaling een andere waarde op. "
            f"Er zit ongeseede randomness in; elke invariantietoets op deze "
            f"feature meet vanaf nu ruis.")

    @pytest.mark.parametrize("symbol", SYMBOLS)
    def test_the_matrix_hash_is_stable_within_a_process(self, symbol: str) -> None:
        source = certified_frame(symbol)
        pipeline = REGISTRY.pipeline()
        a = pipeline.transform(source)
        b = pipeline.transform(source)
        assert a.content_hash() == b.content_hash()


# --------------------------------------------------------------------------- #
# Over processen heen
# --------------------------------------------------------------------------- #
_MATRIX_HASH_SCRIPT = """
    import pandas as pd
    from pathlib import Path
    from tradebot.data.pit_store import PitStore
    from tradebot.features.base import DataRegister
    from tradebot.features.microstructure import build_certified_micro_frame
    from tradebot.features.registry import build_default_registry, current_git_sha
    from tradebot.schemas.config import DataConfig, FeatureConfig, load_config

    root = Path(".").resolve()
    data_cfg = load_config(root / "conf" / "data" / "default.yaml", DataConfig)
    feat_cfg = load_config(root / "conf" / "features" / "default.yaml", FeatureConfig)
    store = PitStore(root / data_cfg.pit_store_root)
    source = build_certified_micro_frame(
        store,
        DataRegister(root / "artefacts" / "governance" / "data_hashes.json"),
        symbol="BTCUSDT", granularity="1d",
        funding_granularity="8h", open_interest_granularity="1d",
        funding_tolerance=pd.Timedelta(hours=feat_cfg.funding_tolerance_hours),
        open_interest_tolerance=pd.Timedelta(
            hours=feat_cfg.open_interest_tolerance_hours),
        asset_class="crypto",
    )
    registry = build_default_registry(feat_cfg)
    matrix = registry.pipeline().transform(source)
    print(matrix.content_hash())
    print(registry.matrix_hash(
        data_hashes=source.data_hashes, git_sha=current_git_sha()))
    print(",".join(matrix.columns))
"""


@requires_store
class TestDeterminismAcrossProcesses:
    """Twee verse interpreters met VERSCHILLENDE PYTHONHASHSEED.

    Dit is de enige manier om te zien of er ergens een `set`- of `dict`-volgorde
    in een artefacthash meeloopt. Binnen één proces is die fout onzichtbaar, en
    in CI verschijnt hij als een hash-mismatch waarvan niemand de oorzaak kan
    reproduceren.
    """

    def test_the_matrix_hash_survives_a_different_hash_seed(self) -> None:
        first = _run(_MATRIX_HASH_SCRIPT, hash_seed=HASH_SEEDS[0])
        second = _run(_MATRIX_HASH_SCRIPT, hash_seed=HASH_SEEDS[1])
        assert first == second, (
            "de feature-matrix hasht anders onder een andere PYTHONHASHSEED. "
            "Er loopt een verzamelings-iteratievolgorde mee in de hash; het "
            "artefact is dan niet reproduceerbaar tussen twee runs van dezelfde "
            "code op dezelfde data.\n"
            f"seed {HASH_SEEDS[0]}: {first}\nseed {HASH_SEEDS[1]}: {second}")

    def test_the_subprocess_actually_produced_a_hash(self) -> None:
        """Controle op het instrument: lege uitvoer zou de test hierboven groen
        maken zonder iets te vergelijken."""
        out = _run(_MATRIX_HASH_SCRIPT, hash_seed=HASH_SEEDS[0]).splitlines()
        assert len(out) == 3
        assert len(out[0]) >= 16 and len(out[1]) >= 16
        assert out[2].count(",") >= 1


class TestImportOrderIsIrrelevant:
    """REGRESSIETEST OP DE IMPORTCYCLUS DIE IN STAGE B-2 IS GEVONDEN.

    `execution/order_router.py` importeert `Fill` uit `backtest/accounting.py`,
    wat het PAKKET `tradebot.backtest` initialiseert, wat `backtest/engine.py`
    laadt, wat terug importeert uit `order_router` — dat dan pas halverwege zijn
    eigen module is. Wie `order_router` als eerste importeerde, kreeg een
    ImportError; wie `tradebot.backtest` als eerste importeerde, merkte niets.

    De onderliggende laagfout (L9 importeert uit L10) staat als DI-18
    geregistreerd. Deze test bewaakt het symptoom, dat de duurdere helft is: een
    live proces dat met de verkeerde import begint, start niet op.
    """

    @pytest.mark.parametrize(
        "first",
        [
            "tradebot.execution.order_router",
            "tradebot.backtest",
            "tradebot.backtest.engine",
            "tradebot.execution.context",
            "tradebot.risk.engine",
            "tradebot.validation.gates",
        ],
    )
    def test_importing_any_entry_point_first_works(self, first: str) -> None:
        out = _run(
            f"""
            import {first}                       # noqa: F401
            from tradebot.backtest import EventDrivenEngine
            from tradebot.execution.order_router import ExecutionReport, OrderRouter
            print("ok")
            """,
            hash_seed=HASH_SEEDS[0],
        )
        assert out.endswith("ok"), (
            f"met `import {first}` als eerste tradebot-import komt de "
            f"authoritative engine of de order router niet meer omhoog")


# --------------------------------------------------------------------------- #
# Negatieve controle
# --------------------------------------------------------------------------- #
class _UnseededFeature(BaseFeature):
    """Bewust niet-deterministisch: een ongeseede generator.

    Dit is geen karikatuur. Het idioom ontstaat zodra iemand een jitter, een
    bootstrap, een dropout of een tie-breaker toevoegt zonder seed, en het is
    onzichtbaar in code review omdat er niets fout AAN te zien is.
    """

    name: ClassVar[str] = "deliberately_unseeded"
    input_spec: ClassVar[InputSpec] = InputSpec(datasets=("ohlcv",), columns=("close",))

    def __init__(self) -> None:
        super().__init__(params={})

    @property
    def burn_in_period(self) -> int:
        return 1

    @property
    def output_columns(self) -> tuple[str, ...]:
        return ("unseeded_jitter",)

    def _compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        close = frame["close"].astype("float64").to_numpy()
        out = close * (1.0 + np.random.default_rng().normal(0.0, 1e-6, close.size))
        out[0] = np.nan
        return pd.DataFrame({self.output_columns[0]: out}, index=frame.index)


@requires_store
class TestTheGateCanGoRed:
    def test_an_unseeded_feature_is_caught_on_repetition(self) -> None:
        source = certified_frame("BTCUSDT")
        leaker = _UnseededFeature()
        first = leaker.transform(source).values.to_numpy(dtype="float64")
        second = leaker.transform(source).values.to_numpy(dtype="float64")
        finite = ~np.isnan(first)
        assert (first[finite] != second[finite]).any(), (
            "de bewust ongeseede feature gaf twee keer hetzelfde antwoord; deze "
            "negatieve controle meet dan niets")

    def test_an_unseeded_feature_also_breaks_the_matrix_hash(self) -> None:
        """Het gevolg dat er in de praktijk toe doet: het ARTEFACT verandert.

        Een niet-deterministische feature maakt elke `content_hash` waardeloos,
        en daarmee elke claim dat een run is gereproduceerd.
        """
        source = certified_frame("BTCUSDT")
        leaker = _UnseededFeature()
        a = leaker.transform(source).values
        b = leaker.transform(source).values
        assert not a.equals(b)

    def test_the_git_sha_resolves_so_the_hash_means_something(self) -> None:
        """Een matrix_hash met een niet-resolvende git_sha is niet herleidbaar
        tot code, en dan is reproduceerbaarheid een bewering."""
        sha = current_git_sha()
        assert sha and sha not in ("unknown", "UNKNOWN", ""), (
            f"current_git_sha() gaf {sha!r}; elk artefact dat hem meeschrijft is "
            f"per audit §26 invalide")
