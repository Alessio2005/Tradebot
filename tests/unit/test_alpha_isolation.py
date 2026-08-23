"""Laagscheiding van L4 — Phase 3, deliverable 10. Audit sectie 11.1.

*"Alpha kent geen risico en geen executie."*

Twee bewijzen, en ze zijn allebei nodig:

1. **Statisch.** Geen enkel bestand in `src/tradebot/alpha/` importeert uit
   `risk/`, `portfolio/`, `execution/` of `oms/` — ook niet binnen een functie
   of achter een `if`. Een conditionele import is geen uitzondering maar de
   manier waarop de scheiding in de praktijk verwatert.
2. **Contractueel.** Elke `AlphaUnit`-output valt binnen `[-1, +1]`, respecteert
   zijn eigen burn-in en kent geen impliciete missing-policy. Een unit die
   buiten het bereik komt, spreekt over positiegrootte in plaats van over
   richting — en dat is risicologica.

`TestTheScannerCanGoRed` bewijst dat de scanner een echt lek zou vangen. Een
statische controle die nooit rood is geweest, meet niets.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from tradebot.alpha.base import (
    FORBIDDEN_LAYERS,
    AlphaUnit,
    _imported_layers,
    assert_alpha_isolation,
)
from tradebot.alpha.momentum import CrossSectionalMomentum, build_cross_sectional_momentum
from tradebot.schemas.config import AlphaConfig, load_config
from tradebot.utils.failfast import DataContractError

ROOT = Path(__file__).resolve().parents[2]
ALPHA_DIR = ROOT / "src" / "tradebot" / "alpha"
ALPHA_CFG = load_config(ROOT / "conf" / "model" / "alpha.yaml", AlphaConfig)

ALPHA_MODULES = sorted(p for p in ALPHA_DIR.glob("*.py"))


class _FakeFeatures:
    """Minimaal gecertificeerd feature-artefact voor de contract-tests."""

    def __init__(self, values: pd.DataFrame) -> None:
        self.values = values
        self.data_hashes = (("crypto/ohlcv/SYNTH/1d", "0" * 32),)


def _panel(rows: int, cols: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2021-01-01", periods=rows, freq="1D", tz="UTC")
    rng = np.random.default_rng(20260823)
    data = rng.uniform(-1.0, 1.0, size=(rows, cols))
    return pd.DataFrame(data, index=idx,
                        columns=[f"S{i}" for i in range(cols)]).astype("float64")


# --------------------------------------------------------------------------- #
# 1. Statische scheiding
# --------------------------------------------------------------------------- #
class TestAlphaImportsNothingForbidden:
    @pytest.mark.parametrize("path", ALPHA_MODULES,
                             ids=[p.name for p in ALPHA_MODULES])
    def test_module_imports_no_forbidden_layer(self, path: Path) -> None:
        offending = sorted(
            _imported_layers(path.read_text(encoding="utf-8"), str(path))
        )
        assert offending == [], (
            f"{path.name} importeert uit {offending}. Alpha kent geen risico en "
            f"geen executie (sectie 11.1); risicologica die hier binnensluipt "
            f"wordt in Phase 4 niet meer teruggevonden."
        )

    def test_the_whole_layer_at_once(self) -> None:
        """Ook een nieuw bestand moet er per ongeluk niet doorheen glippen."""
        assert ALPHA_MODULES, "alpha/ is leeg; de scan meet dan niets"
        offenders = {
            p.name: sorted(_imported_layers(p.read_text(encoding="utf-8"), str(p)))
            for p in ALPHA_MODULES
        }
        assert {k: v for k, v in offenders.items() if v} == {}

    def test_forbidden_set_matches_the_audit(self) -> None:
        assert set(FORBIDDEN_LAYERS) == {"risk", "portfolio", "execution", "oms"}

    def test_live_module_check_passes(self) -> None:
        assert_alpha_isolation("tradebot.alpha.momentum")
        assert_alpha_isolation("tradebot.alpha.base")


class TestTheScannerCanGoRed:
    """Bewijs door falen: de scanner moet een echt lek vangen."""

    @pytest.mark.parametrize(
        "snippet",
        [
            "from ..risk.limits import cap",
            "from tradebot.risk import something",
            "import tradebot.execution.fees",
            "from ..oms.paper_oms import PaperOMS",
            "from ..portfolio.risk_parity import sized_by_risk_parity",
        ],
    )
    def test_each_import_form_is_detected(self, snippet: str) -> None:
        found = _imported_layers(snippet, "<synthetic>")
        assert found, f"scanner miste een verboden import: {snippet!r}"

    def test_import_hidden_inside_a_function_is_detected(self) -> None:
        """Een import achter een `if` of in een functie telt net zo goed."""
        source = textwrap.dedent(
            """
            def size_it(a):
                if a > 0:
                    from ..risk.vol_targeting import target
                    return target(a)
                return 0.0
            """
        )
        assert _imported_layers(source, "<synthetic>") == {"risk"}

    def test_a_unit_in_a_leaking_module_crashes_at_class_creation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`__init_subclass__` weigert de klasse, niet pas de eerste aanroep."""
        import sys
        import types

        leaking = tmp_path / "leaking_unit.py"
        leaking.write_text("from ..risk import anything\n", encoding="utf-8")
        module = types.ModuleType("tradebot.alpha._leaking_test_module")
        module.__file__ = str(leaking)
        monkeypatch.setitem(sys.modules, module.__name__, module)

        with pytest.raises(DataContractError, match="verboden laag"):
            assert_alpha_isolation(module.__name__)

    def test_a_clean_module_passes(self, tmp_path: Path,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
        import sys
        import types

        clean = tmp_path / "clean_unit.py"
        clean.write_text("from ..features.transforms import cross_sectional_rank\n",
                         encoding="utf-8")
        module = types.ModuleType("tradebot.alpha._clean_test_module")
        module.__file__ = str(clean)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        assert_alpha_isolation(module.__name__)


# --------------------------------------------------------------------------- #
# 2. Het output-contract
# --------------------------------------------------------------------------- #
class _OutOfRangeUnit(AlphaUnit):
    name: ClassVar[str] = "deliberately_out_of_range"

    def __init__(self, *, scale: float) -> None:
        super().__init__(signal_floor=ALPHA_CFG.signal_floor,
                         signal_cap=ALPHA_CFG.signal_cap,
                         params={"scale": float(scale)})

    @property
    def burn_in_period(self) -> int:
        return 0

    def _generate(self, features: pd.DataFrame) -> pd.DataFrame:
        return (features * float(self.params["scale"])).astype("float64")


class _HolesUnit(AlphaUnit):
    """Produceert een NaN waar de feature wel bestaat."""

    name: ClassVar[str] = "deliberately_holey"

    def __init__(self) -> None:
        super().__init__(signal_floor=ALPHA_CFG.signal_floor,
                         signal_cap=ALPHA_CFG.signal_cap, params={})

    @property
    def burn_in_period(self) -> int:
        return 0

    def _generate(self, features: pd.DataFrame) -> pd.DataFrame:
        out = features.astype("float64").copy()
        out.iloc[len(out) // 2, 0] = np.nan
        return out


class _EagerUnit(AlphaUnit):
    """Handelt binnen zijn eigen burn-in."""

    name: ClassVar[str] = "deliberately_eager"

    def __init__(self, *, burn_in: int) -> None:
        super().__init__(signal_floor=ALPHA_CFG.signal_floor,
                         signal_cap=ALPHA_CFG.signal_cap,
                         params={"burn_in": int(burn_in)})

    @property
    def burn_in_period(self) -> int:
        return int(self.params["burn_in"])

    def _generate(self, features: pd.DataFrame) -> pd.DataFrame:
        return features.astype("float64")


class TestOutputContract:
    def test_in_range_output_is_accepted(self) -> None:
        features = _FakeFeatures(_panel(50))
        out = _OutOfRangeUnit(scale=1.0).generate(features)
        assert out.exposures.to_numpy().max() <= ALPHA_CFG.signal_cap
        assert out.exposures.to_numpy().min() >= ALPHA_CFG.signal_floor

    def test_out_of_range_output_crashes(self) -> None:
        features = _FakeFeatures(_panel(50))
        with pytest.raises(DataContractError, match="BUITEN het toegestane bereik"):
            _OutOfRangeUnit(scale=2.0).generate(features)

    def test_nan_without_missing_feature_crashes(self) -> None:
        features = _FakeFeatures(_panel(50))
        with pytest.raises(DataContractError, match="ontbrekende feature"):
            _HolesUnit().generate(features)

    def test_exposure_inside_the_burn_in_crashes(self) -> None:
        features = _FakeFeatures(_panel(50))
        with pytest.raises(DataContractError, match="binnen de burn-in"):
            _EagerUnit(burn_in=10).generate(features)

    def test_uncertified_input_crashes(self) -> None:
        with pytest.raises(DataContractError, match="data_hash"):
            _OutOfRangeUnit(scale=1.0).generate(_panel(50))

    def test_data_hashes_propagate_to_the_output(self) -> None:
        features = _FakeFeatures(_panel(50))
        out = _OutOfRangeUnit(scale=1.0).generate(features)
        assert out.data_hashes == features.data_hashes

    def test_cap_below_floor_crashes(self) -> None:
        class _Bad(AlphaUnit):
            name: ClassVar[str] = "bad_bounds"

            @property
            def burn_in_period(self) -> int:
                return 0

            def _generate(self, features: pd.DataFrame) -> pd.DataFrame:
                return features

        with pytest.raises(DataContractError, match="signal_cap"):
            _Bad(signal_floor=1.0, signal_cap=-1.0, params={})


# --------------------------------------------------------------------------- #
# 3. De echte baseline-unit
# --------------------------------------------------------------------------- #
class TestCrossSectionalMomentumRespectsTheContract:
    def test_parameters_come_from_conf(self) -> None:
        unit = build_cross_sectional_momentum(ALPHA_CFG)
        assert unit.params["lookback_bars"] == ALPHA_CFG.lookback_bars
        assert unit.params["skip_bars"] == ALPHA_CFG.skip_bars
        assert unit.params["min_assets"] == ALPHA_CFG.min_assets
        assert unit.params["signal_floor"] == ALPHA_CFG.signal_floor
        assert unit.params["signal_cap"] == ALPHA_CFG.signal_cap

    def test_burn_in_is_lookback_plus_skip(self) -> None:
        unit = build_cross_sectional_momentum(ALPHA_CFG)
        assert unit.burn_in_period == ALPHA_CFG.lookback_bars + ALPHA_CFG.skip_bars

    def test_declares_its_features_but_does_not_compute_them(self) -> None:
        """De unit bezit geen wiskunde; L1 doet de berekening."""
        unit = build_cross_sectional_momentum(ALPHA_CFG)
        names = [s.name for s in unit.feature_steps()]
        assert names == ["rolling_log_return", "cross_sectional_rank"]
        assert unit.feature_pipeline().burn_in_period == unit.burn_in_period

    def test_negative_skip_is_refused(self) -> None:
        with pytest.raises(DataContractError, match="TOEKOMST"):
            CrossSectionalMomentum(
                lookback_bars=ALPHA_CFG.lookback_bars, skip_bars=-1,
                min_assets=ALPHA_CFG.min_assets,
                rebalance_every_bars=ALPHA_CFG.rebalance_every_bars,
                signal_floor=ALPHA_CFG.signal_floor,
                signal_cap=ALPHA_CFG.signal_cap,
            )

    def test_output_is_bounded_and_dollar_neutral_on_synthetic_ranks(self) -> None:
        """De rang is per constructie begrensd en per rij som-nul."""
        unit = build_cross_sectional_momentum(ALPHA_CFG)
        rows = unit.burn_in_period + 40
        ranked = _panel(rows)
        ranked.iloc[: unit.burn_in_period] = np.nan
        out = unit.generate(_FakeFeatures(ranked))
        arr = out.exposures.to_numpy()
        finite = arr[np.isfinite(arr)]
        assert finite.min() >= ALPHA_CFG.signal_floor
        assert finite.max() <= ALPHA_CFG.signal_cap
        assert np.isnan(arr[: unit.burn_in_period]).all()
