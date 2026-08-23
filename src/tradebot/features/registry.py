# src/tradebot/features/registry.py
"""L3 - FeatureRegistry en de onveranderlijke `feature_hash`.

Phase 2, deliverable 5.

De `feature_hash` beantwoordt precies een vraag: *welk artefact is dit, en waar
komt het vandaan?* Hij wordt berekend over vier componenten, conform stap 4:

    feature_hash = H( input_data_hashes
                    + feature_class_name
                    + hyperparameter_dict
                    + git_sha )

Aan die vier zijn drie afgeleiden toegevoegd die volledig door de klasse worden
bepaald - `output_columns`, `burn_in_period` en de modulenaam - zodat een
herschrijving van de berekening bij ongewijzigde parameters ook een andere hash
oplevert wanneer hij het schema of de opstartfase raakt.

De gevolgen zijn opzettelijk streng:

* Een ander venster in `conf/features/default.yaml` -> andere hash.
* Een herbouwde PIT-store met andere inhoud -> andere hash.
* ELKE commit -> andere hash, via `git_sha`.

Die laatste is bewust grof. `current_git_sha()` leest HEAD en niet de laatste
commit die `features/` raakt, dus ook een docs-commit verschuift alle hashes.
Dat is de conservatieve kant van de fout: een hash mag nooit beweren dat twee
artefacten gelijk zijn terwijl de codetoestand verschilt. Een artefact blijft
exact reproduceerbaar door de `git_sha` uit zijn manifest uit te checken.

`FEATURE_HASH_VERSION` maakt daarnaast de RECEPTUUR zelf zichtbaar: wijzigt de
samenstelling van de payload, dan verschuiven alle hashes zichtbaar in plaats
van onopgemerkt.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md secties 7.2, 19 (L3), 23 (Phase 2), 26.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..registry.lineage import get_git_sha
from ..schemas.config import FeatureConfig
from ..utils.failfast import DataContractError, require
from ..utils.hashing import DATA_HASH_LENGTH, hash_config
from .base import BaseFeature, FeaturePipeline
from .microstructure import FundingRateMean, FundingRateZScore, OpenInterestLogChange
from .momentum import EwmaReturnSpread, RollingLogReturn
from .volatility import (
    EwmaVolatility,
    ExpandingVolatility,
    ParkinsonVolatility,
    RealizedVolatility,
)

__all__ = [
    "FEATURE_HASH_VERSION",
    "FeatureDefinition",
    "FeatureRegistry",
    "build_default_registry",
    "current_git_sha",
]

#: Versie van de HASH-RECEPTUUR, niet van de features. Wijzigt de samenstelling
#: van de payload hieronder, dan verhoogt dit getal en verschuiven alle hashes
#: zichtbaar in plaats van onopgemerkt.
FEATURE_HASH_VERSION = "phase2.v1"

#: Lengte van de `feature_hash`, gelijk aan die van de `data_hash` (128 bit).
FEATURE_HASH_LENGTH = DATA_HASH_LENGTH


def current_git_sha() -> str:
    """De korte git-SHA van de werkende boom. Crasht buiten een git-repo.

    Er is geen fallback naar een lege string: een feature-artefact zonder
    codeversie is niet auditbaar, en een lege `git_sha` zou de hash stil laten
    samenvallen met die van elke andere commit.
    """
    sha = get_git_sha(short=True)
    require(
        bool(sha),
        "Lege git_sha. Een feature-artefact zonder codeversie is niet "
        "auditbaar; de hash zou samenvallen met die van elke andere commit.",
        DataContractError,
    )
    return sha


@dataclass(frozen=True)
class FeatureDefinition:
    """De onveranderlijke definitie van een geregistreerde feature."""

    feature_id: str
    name: str
    class_name: str
    module: str
    params: Mapping[str, Any]
    input_datasets: tuple[str, ...]
    input_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    burn_in_period: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "name": self.name,
            "class_name": self.class_name,
            "module": self.module,
            "params": {k: self.params[k] for k in sorted(self.params)},
            "input_datasets": list(self.input_datasets),
            "input_columns": list(self.input_columns),
            "output_columns": list(self.output_columns),
            "burn_in_period": int(self.burn_in_period),
        }


class FeatureRegistry:
    """Beheert feature-definities, versiebeheer en de `feature_hash`.

    De registry is een geordende, dubbelvrije verzameling. Registreren van
    dezelfde geparametriseerde feature crasht: twee identieke definities zouden
    dezelfde hash krijgen en het onderscheid tussen artefacten opheffen.
    """

    def __init__(self) -> None:
        self._features: list[BaseFeature] = []
        self._by_id: dict[str, BaseFeature] = {}

    # ------------------------------------------------------------- registratie
    def register(self, feature: BaseFeature) -> FeatureDefinition:
        require(
            isinstance(feature, BaseFeature),
            "Alleen een BaseFeature kan worden geregistreerd.",
            DataContractError,
            got=type(feature).__name__,
        )
        fid = feature.feature_id
        require(
            fid not in self._by_id,
            "Deze geparametriseerde feature is al geregistreerd. Twee identieke "
            "definities zouden dezelfde feature_hash krijgen en het onderscheid "
            "tussen artefacten opheffen.",
            DataContractError,
            feature_id=fid,
        )
        taken = set(self.output_columns)
        clash = [c for c in feature.output_columns if c in taken]
        require(
            not clash,
            "Botsende outputkolom met een reeds geregistreerde feature.",
            DataContractError,
            feature_id=fid,
            clashing=clash,
        )
        self._by_id[fid] = feature
        self._features.append(feature)
        return self.definition(feature)

    def register_all(self, features: Sequence[BaseFeature]) -> None:
        for f in features:
            self.register(f)

    # ---------------------------------------------------------------- toegang
    def __len__(self) -> int:
        return len(self._features)

    @property
    def features(self) -> tuple[BaseFeature, ...]:
        return tuple(self._features)

    @property
    def output_columns(self) -> tuple[str, ...]:
        cols: list[str] = []
        for f in self._features:
            cols.extend(f.output_columns)
        return tuple(cols)

    def pipeline(self) -> FeaturePipeline:
        """Alle geregistreerde features als een enkele causale pipeline."""
        return FeaturePipeline(self._features)

    @staticmethod
    def definition(feature: BaseFeature) -> FeatureDefinition:
        spec = type(feature).input_spec
        return FeatureDefinition(
            feature_id=feature.feature_id,
            name=feature.name,
            class_name=type(feature).__name__,
            module=type(feature).__module__,
            params=dict(feature.params),
            input_datasets=tuple(spec.datasets),
            input_columns=tuple(spec.columns),
            output_columns=tuple(feature.output_columns),
            burn_in_period=int(feature.burn_in_period),
        )

    def definitions(self) -> tuple[FeatureDefinition, ...]:
        return tuple(self.definition(f) for f in self._features)

    # ------------------------------------------------------------------ hash
    @staticmethod
    def _hash_payload(
        feature: BaseFeature,
        *,
        data_hashes: Sequence[tuple[str, str]],
        git_sha: str,
    ) -> dict[str, Any]:
        require(
            bool(data_hashes),
            "Een feature_hash zonder input-data_hash is betekenisloos: het "
            "artefact zou niet aan een gecertificeerde dataset te koppelen zijn.",
            DataContractError,
            feature=feature.name,
        )
        require(
            bool(git_sha),
            "Een feature_hash zonder git_sha is niet auditbaar.",
            DataContractError,
            feature=feature.name,
        )
        definition = FeatureRegistry.definition(feature)
        return {
            "hash_version": FEATURE_HASH_VERSION,
            "feature_class_name": definition.class_name,
            "feature_module": definition.module,
            "feature_name": definition.name,
            "hyperparameters": {k: definition.params[k] for k in sorted(definition.params)},
            "output_columns": list(definition.output_columns),
            "burn_in_period": definition.burn_in_period,
            "input_data_hashes": [list(pair) for pair in sorted(data_hashes)],
            "git_sha": git_sha,
        }

    def feature_hash(
        self,
        feature: BaseFeature,
        *,
        data_hashes: Sequence[tuple[str, str]],
        git_sha: str,
    ) -> str:
        """De onveranderlijke `feature_hash` van een enkele feature.

        `data_hashes` en `git_sha` zijn keyword-only en hebben GEEN default.
        Een default zou precies de situatie creeren die dit contract uitsluit:
        een artefact dat een provenance claimt die niemand heeft opgegeven.
        """
        payload = self._hash_payload(feature, data_hashes=data_hashes, git_sha=git_sha)
        return hash_config(payload, length=FEATURE_HASH_LENGTH)

    def feature_hashes(
        self, *, data_hashes: Sequence[tuple[str, str]], git_sha: str
    ) -> dict[str, str]:
        """`feature_id` -> `feature_hash`, voor elke geregistreerde feature."""
        return {
            f.feature_id: self.feature_hash(f, data_hashes=data_hashes, git_sha=git_sha)
            for f in self._features
        }

    def matrix_hash(
        self, *, data_hashes: Sequence[tuple[str, str]], git_sha: str
    ) -> str:
        """Hash over de VOLLEDIGE matrixdefinitie; de naam van het artefact.

        Bevat de individuele feature-hashes in registratievolgorde, zodat het
        herordenen van de kolommen - wat de kolomvolgorde in de parquet
        verandert - ook een ander artefact oplevert.
        """
        require(
            len(self._features) > 0,
            "Lege registry: er is geen matrix om te hashen.",
            DataContractError,
        )
        ordered = [
            self.feature_hash(f, data_hashes=data_hashes, git_sha=git_sha)
            for f in self._features
        ]
        return hash_config(
            {
                "hash_version": FEATURE_HASH_VERSION,
                "feature_hashes_in_order": ordered,
                "columns_in_order": list(self.output_columns),
                "git_sha": git_sha,
            },
            length=FEATURE_HASH_LENGTH,
        )

    def manifest(
        self, *, data_hashes: Sequence[tuple[str, str]], git_sha: str
    ) -> dict[str, Any]:
        """JSON-serialiseerbaar manifest: alles wat het artefact reproduceerbaar maakt."""
        return {
            "hash_version": FEATURE_HASH_VERSION,
            "git_sha": git_sha,
            "input_data_hashes": {k: v for k, v in sorted(data_hashes)},
            "matrix_hash": self.matrix_hash(data_hashes=data_hashes, git_sha=git_sha),
            "burn_in_period": max(int(f.burn_in_period) for f in self._features),
            "features": [
                {
                    **self.definition(f).as_dict(),
                    "feature_hash": self.feature_hash(
                        f, data_hashes=data_hashes, git_sha=git_sha
                    ),
                }
                for f in self._features
            ],
        }


def build_default_registry(cfg: FeatureConfig) -> FeatureRegistry:
    """Bouw de volledige Phase 2-registry uit `conf/features/default.yaml`.

    Dit is de ENIGE plek waar features worden geinstantieerd voor productie- en
    testgebruik. Elke parameter komt uit `cfg`; er staat geen enkel venster,
    decay of drempel als literal in deze functie. Daardoor is de verzameling
    features die de causaliteitstests doorlopen per constructie gelijk aan de
    verzameling die de feature store vult.
    """
    af = float(cfg.annualisation_factor)
    features: list[BaseFeature] = []

    # L3 - volatiliteit. Zie de DI-2-toelichting in features/volatility.py.
    features.extend(
        RealizedVolatility(window=w, annualisation_factor=af)
        for w in cfg.volatility.realized_windows
    )
    features.append(
        ExpandingVolatility(
            min_periods=cfg.min_expanding_periods, annualisation_factor=af
        )
    )
    features.append(
        EwmaVolatility(
            lam=cfg.volatility.ewma_lambda,
            burn_in_bars=cfg.volatility.ewma_burn_in_bars,
            annualisation_factor=af,
        )
    )
    features.append(
        ParkinsonVolatility(window=cfg.volatility.range_window, annualisation_factor=af)
    )

    # L3 - momentum.
    features.extend(
        RollingLogReturn(window=w, skip_bars=cfg.momentum.skip_bars)
        for w in cfg.momentum.lookback_windows
    )
    features.append(
        EwmaReturnSpread(
            fast_span=cfg.momentum.ewma_fast_span,
            slow_span=cfg.momentum.ewma_slow_span,
        )
    )

    # L3 - microstructuur en carry.
    features.extend(
        FundingRateMean(window=w) for w in cfg.microstructure.funding_windows
    )
    features.append(
        FundingRateZScore(min_periods=cfg.microstructure.funding_zscore_min_periods)
    )
    features.extend(
        OpenInterestLogChange(window=w) for w in cfg.microstructure.oi_change_windows
    )

    registry = FeatureRegistry()
    registry.register_all(features)
    return registry
