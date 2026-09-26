"""Phase 2, deliverable 7 - BIT-NIVEAU DETERMINISME van de feature-laag.

Dit is exit criterium 3 van Phase 2.

Twee beweringen worden hier bewezen:

1. **Identieke `data_hash` + identieke featureconfiguratie -> identieke
   `feature_hash` EN een bit-identieke feature-matrix.** Zonder die eigenschap
   verwijst een `feature_hash` in de ledger naar niets specifieks en is elke
   reproductiepoging een gok.
2. **Elke wijziging die het artefact raakt, verandert de hash.** Een andere
   parameter, andere brondata of een andere commit levert gegarandeerd een
   andere `feature_hash` op. Een hash die NIET meebeweegt is gevaarlijker dan
   geen hash: hij wekt de indruk van provenance die er niet is.

De toetsen draaien op een synthetisch frame en zijn daarmee hermetisch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradebot.features.base import CertifiedFrame, FeaturePipeline
from tradebot.features.registry import (
    FEATURE_HASH_VERSION,
    FeatureRegistry,
    build_default_registry,
)
from tradebot.features.volatility import RealizedVolatility
from tradebot.schemas.config import FeatureConfig, load_config, validate_mapping
from tradebot.utils.failfast import DataContractError
from tradebot.utils.hashing import dataframe_content_hash

ROOT = Path(__file__).resolve().parents[2]
FEAT_CFG = load_config(ROOT / "conf" / "features" / "default.yaml", FeatureConfig)

#: Vaste seed: een determinismetoets op willekeurige data is een tegenspraak.
SYNTHETIC_SEED = 20260823
#: Genoeg bars om elke burn-in in de standaardregistry ruim te passeren.
SYNTHETIC_BARS = 400
GIT_SHA = "abc1234"
OTHER_GIT_SHA = "def5678"


def synthetic_frame(*, seed: int = SYNTHETIC_SEED) -> CertifiedFrame:
    """Deterministisch OHLCV + funding + open interest frame.

    Geen echte data nodig: dit toetst de HASH-RECEPTUUR en het determinisme van
    de pipeline, niet de marktinhoud. De causaliteit zelf wordt op de echte
    gecertificeerde reeksen getoetst in tests/lookahead/test_feature_causality.py.
    """
    rng = np.random.default_rng(seed)
    n = SYNTHETIC_BARS
    start = pd.Timestamp("2021-01-01", tz="UTC")
    event = pd.date_range(start, periods=n, freq="1D")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, size=n)))
    spread = np.abs(rng.normal(0.0, 0.01, size=n)) + 0.005
    frame = pd.DataFrame(
        {
            "event_ts_ns": event.astype("int64"),
            "asof_ts_ns": (event + pd.Timedelta(days=1)).astype("int64"),
            "open": close,
            "high": close * (1.0 + spread),
            "low": close * (1.0 - spread),
            "close": close,
            "volume": rng.uniform(1.0, 10.0, size=n),
            "funding_rate": rng.normal(0.0001, 0.00005, size=n),
            "open_interest": 1000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n))),
        },
        index=pd.DatetimeIndex(event, name="event_ts"),
    )
    return CertifiedFrame(
        frame=frame,
        symbol="SYNTH",
        data_hashes=(
            ("crypto/ohlcv/SYNTH/1d", "0" * 32),
            ("crypto/funding/SYNTH/8h", "1" * 32),
            ("crypto/open_interest/SYNTH/1d", "2" * 32),
        ),
    )


REGISTRY = build_default_registry(FEAT_CFG)


# --------------------------------------------------------------------------- #
# 1. Identieke input -> identiek artefact
# --------------------------------------------------------------------------- #
class TestIdenticalInputsGiveIdenticalOutputs:
    def test_feature_hash_is_stable_across_calls(self) -> None:
        source = synthetic_frame()
        first = REGISTRY.feature_hashes(
            data_hashes=source.data_hashes, git_sha=GIT_SHA
        )
        second = REGISTRY.feature_hashes(
            data_hashes=source.data_hashes, git_sha=GIT_SHA
        )
        assert first == second

    def test_two_registries_from_the_same_config_agree(self) -> None:
        """De registry is een functie van de config, niet van de aanroepvolgorde."""
        source = synthetic_frame()
        a = build_default_registry(FEAT_CFG)
        b = build_default_registry(FEAT_CFG)
        assert a.matrix_hash(
            data_hashes=source.data_hashes, git_sha=GIT_SHA
        ) == b.matrix_hash(data_hashes=source.data_hashes, git_sha=GIT_SHA)

    def test_matrix_is_bit_identical_across_runs(self) -> None:
        source = synthetic_frame()
        first = REGISTRY.pipeline().transform(source).values
        second = REGISTRY.pipeline().transform(synthetic_frame()).values
        pd.testing.assert_frame_equal(first, second, check_exact=True)
        assert dataframe_content_hash(
            first.reset_index()
        ) == dataframe_content_hash(second.reset_index())

    def test_hash_ignores_the_order_data_hashes_are_supplied_in(self) -> None:
        """De provenance is een VERZAMELING, geen lijst; hij wordt gesorteerd."""
        source = synthetic_frame()
        reversed_hashes = tuple(reversed(source.data_hashes))
        assert REGISTRY.matrix_hash(
            data_hashes=source.data_hashes, git_sha=GIT_SHA
        ) == REGISTRY.matrix_hash(data_hashes=reversed_hashes, git_sha=GIT_SHA)

    def test_every_column_is_float64(self) -> None:
        matrix = REGISTRY.pipeline().transform(synthetic_frame())
        for col in matrix.columns:
            assert matrix.values[col].dtype == np.dtype("float64"), col
        store = matrix.to_store_frame(drop_burn_in=True)
        assert store["event_ts_ns"].dtype == np.dtype("int64")
        assert store["asof_ts_ns"].dtype == np.dtype("int64")


# --------------------------------------------------------------------------- #
# 2. Elke relevante wijziging verandert de hash
# --------------------------------------------------------------------------- #
class TestAnyChangeChangesTheHash:
    def test_changed_hyperparameter_changes_the_hash(self) -> None:
        source = synthetic_frame()
        af = float(FEAT_CFG.annualisation_factor)
        base = RealizedVolatility(window=FEAT_CFG.volatility.realized_windows[0],
                                  annualisation_factor=af)
        other = RealizedVolatility(window=FEAT_CFG.volatility.realized_windows[1],
                                   annualisation_factor=af)
        registry = FeatureRegistry()
        assert registry.feature_hash(
            base, data_hashes=source.data_hashes, git_sha=GIT_SHA
        ) != registry.feature_hash(
            other, data_hashes=source.data_hashes, git_sha=GIT_SHA
        )

    def test_changed_input_data_hash_changes_the_hash(self) -> None:
        source = synthetic_frame()
        tampered = (("crypto/ohlcv/SYNTH/1d", "f" * 32),) + source.data_hashes[1:]
        assert REGISTRY.matrix_hash(
            data_hashes=source.data_hashes, git_sha=GIT_SHA
        ) != REGISTRY.matrix_hash(data_hashes=tampered, git_sha=GIT_SHA)

    def test_changed_git_sha_changes_the_hash(self) -> None:
        source = synthetic_frame()
        assert REGISTRY.matrix_hash(
            data_hashes=source.data_hashes, git_sha=GIT_SHA
        ) != REGISTRY.matrix_hash(data_hashes=source.data_hashes,
                                  git_sha=OTHER_GIT_SHA)

    def test_changed_config_changes_the_matrix_hash(self) -> None:
        """Exit criterium 5 in hash-vorm: parameters zitten NIET in de code."""
        source = synthetic_frame()
        raw = FEAT_CFG.model_dump(mode="python")
        raw["volatility"]["realized_windows"] = tuple(
            w + 1 for w in FEAT_CFG.volatility.realized_windows
        )
        tweaked = validate_mapping(FeatureConfig, raw, source="tweaked")
        assert build_default_registry(tweaked).matrix_hash(
            data_hashes=source.data_hashes, git_sha=GIT_SHA
        ) != REGISTRY.matrix_hash(data_hashes=source.data_hashes, git_sha=GIT_SHA)

    def test_feature_order_changes_the_matrix_hash(self) -> None:
        """De kolomvolgorde in de parquet is onderdeel van het artefact."""
        source = synthetic_frame()
        forward = FeatureRegistry()
        forward.register_all(REGISTRY.features)
        backward = FeatureRegistry()
        backward.register_all(tuple(reversed(REGISTRY.features)))
        assert forward.matrix_hash(
            data_hashes=source.data_hashes, git_sha=GIT_SHA
        ) != backward.matrix_hash(data_hashes=source.data_hashes, git_sha=GIT_SHA)

    def test_hash_recipe_version_is_part_of_the_payload(self) -> None:
        assert FEATURE_HASH_VERSION
        source = synthetic_frame()
        payload = FeatureRegistry._hash_payload(
            REGISTRY.features[0], data_hashes=source.data_hashes, git_sha=GIT_SHA
        )
        assert payload["hash_version"] == FEATURE_HASH_VERSION
        for key in ("feature_class_name", "hyperparameters", "input_data_hashes",
                    "git_sha"):
            assert key in payload, key


# --------------------------------------------------------------------------- #
# 3. Geen provenance -> geen hash
# --------------------------------------------------------------------------- #
class TestProvenanceIsMandatory:
    def test_empty_data_hashes_crashes(self) -> None:
        with pytest.raises(DataContractError, match="input-data_hash"):
            REGISTRY.feature_hash(REGISTRY.features[0], data_hashes=(),
                                  git_sha=GIT_SHA)

    def test_empty_git_sha_crashes(self) -> None:
        source = synthetic_frame()
        with pytest.raises(DataContractError, match="git_sha"):
            REGISTRY.feature_hash(REGISTRY.features[0],
                                  data_hashes=source.data_hashes, git_sha="")

    def test_certified_frame_without_hash_crashes(self) -> None:
        source = synthetic_frame()
        with pytest.raises(DataContractError, match="data_hash"):
            CertifiedFrame(frame=source.frame, symbol="SYNTH", data_hashes=())

    def test_duplicate_registration_crashes(self) -> None:
        registry = FeatureRegistry()
        af = float(FEAT_CFG.annualisation_factor)
        window = FEAT_CFG.volatility.realized_windows[0]
        registry.register(RealizedVolatility(window=window, annualisation_factor=af))
        with pytest.raises(DataContractError, match="al geregistreerd"):
            registry.register(
                RealizedVolatility(window=window, annualisation_factor=af)
            )

    def test_empty_registry_has_no_matrix_hash(self) -> None:
        with pytest.raises(DataContractError, match="Lege registry"):
            FeatureRegistry().matrix_hash(data_hashes=(("a", "b"),), git_sha=GIT_SHA)


# --------------------------------------------------------------------------- #
# 4. Manifest en pipeline-invarianten
# --------------------------------------------------------------------------- #
class TestManifestDescribesTheArtefact:
    def test_manifest_is_json_serialisable_and_complete(self) -> None:
        source = synthetic_frame()
        manifest = REGISTRY.manifest(data_hashes=source.data_hashes, git_sha=GIT_SHA)
        round_trip = json.loads(json.dumps(manifest, sort_keys=True))
        assert round_trip == manifest
        assert len(manifest["features"]) == len(REGISTRY)
        assert manifest["git_sha"] == GIT_SHA
        assert manifest["burn_in_period"] == REGISTRY.pipeline().burn_in_period
        for entry in manifest["features"]:
            assert entry["feature_hash"]
            assert entry["output_columns"]
            assert entry["burn_in_period"] >= 0

    def test_pipeline_burn_in_is_the_maximum_of_its_members(self) -> None:
        pipeline = REGISTRY.pipeline()
        assert pipeline.burn_in_period == max(
            int(f.burn_in_period) for f in REGISTRY.features
        )

    def test_pipeline_rejects_colliding_output_columns(self) -> None:
        af = float(FEAT_CFG.annualisation_factor)
        window = FEAT_CFG.volatility.realized_windows[0]
        a = RealizedVolatility(window=window, annualisation_factor=af)
        b = RealizedVolatility(window=window, annualisation_factor=af * 2.0)
        assert a.output_columns == b.output_columns
        with pytest.raises(DataContractError, match="Botsende outputkolommen"):
            FeaturePipeline([a, b])
