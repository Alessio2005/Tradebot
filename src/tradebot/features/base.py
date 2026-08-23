# src/tradebot/features/base.py
"""L3 - Feature Store & Causal Pipeline: het basiscontract.

Phase 2, deliverable 1.

Een feature is in dit platform geen wiskundige transformatie maar een
**gecertificeerd informatie-artefact**. Deze module legt vast waaraan het moet
voldoen voordat het die naam mag dragen:

1. **Gecertificeerde input.** Een `CertifiedFrame` draagt de `data_hash` van elke
   PIT-reeks waaruit hij is opgebouwd, en die hash wordt geverifieerd tegen het
   Phase 1 data-register (`artefacts/governance/data_hashes.json`). Data zonder
   register-entry is geen input maar een bewering.
2. **Expliciet outputschema.** De feature-matrix is geindexeerd op `asof_ts` -
   het moment waarop de waarde BEKEND was, niet het moment waarop de gebeurtenis
   plaatsvond. Voor een daily bar die om 00:00 opent en pas de volgende dag
   sluit, is de feature dus pas op die sluitingstijd beschikbaar. Wie op
   `event_ts` indexeert, bouwt per constructie een lookahead in.
3. **Burn-in blijft NaN.** Er is geen `fillna`. Geen sample-brede std (DI-2),
   geen `fillna(0)`, geen `bfill` over de opstartfase. Tijdens de burn-in bestaat
   er domweg nog geen schatting, en dat feit wordt doorgegeven in plaats van
   verborgen. `transform` DWINGT dit af: een feature die tijdens zijn eigen
   burn-in een eindige waarde produceert, crasht.
4. **Zuiverheid.** `_compute` is een pure functie van het inputframe. Geen
   toestand tussen aanroepen, geen globale statistieken, geen centrerende
   vensters. Daarmee is truncatie-invariantie een eigenschap van de constructie
   en niet van de discipline van de auteur.
5. **float64, overal.** Impliciete dtype-conversie maakt cross-platform
   determinisme onmogelijk; `transform` weigert elke andere dtype.

Ref: ARCHITECTUUR_AUDIT_2026-08-22.md secties 7.2, 8, 8.2, 10.2, 19 (L3), 26.
"""
from __future__ import annotations

import abc
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ..utils.failfast import CausalityViolationError, DataContractError, require
from ..utils.hashing import dataframe_content_hash
from ..utils.time import assert_utc_index

__all__ = [
    "ASOF_INDEX_NAME",
    "BaseFeature",
    "CertifiedFrame",
    "CertifiedPanel",
    "DATA_REGISTER_RELPATH",
    "DataRegister",
    "FEATURE_DTYPE",
    "FeatureMatrix",
    "FeaturePipeline",
    "FeatureResult",
    "InputSpec",
    "REQUIRED_TIME_COLUMNS",
    "load_certified_close_panel",
    "load_certified_series",
    "repo_root",
]

#: Naam van de index van elke feature-matrix: het BESCHIKBAARHEIDSMOMENT.
ASOF_INDEX_NAME = "asof_ts"

#: Verplichte tijdkolommen in elk gecertificeerd inputframe (sectie 7.2).
REQUIRED_TIME_COLUMNS = ("event_ts_ns", "asof_ts_ns")

#: Het enige toegestane dtype in de feature store. Geen float32, geen object.
FEATURE_DTYPE = np.dtype("float64")

#: Het Phase 1 data-register, relatief aan de repo-root.
DATA_REGISTER_RELPATH = Path("artefacts") / "governance" / "data_hashes.json"


def repo_root() -> Path:
    """Repo-root, afgeleid van de locatie van deze module."""
    return Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
class DataRegister:
    """Het gecertificeerde Phase 1 data-register.

    De enige geldige bron voor de vraag *welke data mag ik gebruiken*. Een
    reeks die hier niet in staat, of waarvan de gemeten inhoud afwijkt van de
    geregistreerde `data_hash`, is geen geldige feature-input.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else repo_root() / DATA_REGISTER_RELPATH
        require(
            self.path.is_file(),
            "Het Phase 1 data-register ontbreekt. Zonder register bestaat er "
            "geen gecertificeerde data en dus geen geldige feature. Genereer "
            "het met scripts/build_data_register.py.",
            DataContractError,
            path=str(self.path),
        )
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        require(
            isinstance(raw, dict) and bool(raw),
            "Het data-register is leeg of geen mapping.",
            DataContractError,
            path=str(self.path),
        )
        self._hashes: dict[str, str] = {str(k): str(v) for k, v in raw.items()}

    @property
    def hashes(self) -> Mapping[str, str]:
        """Onveranderlijke kijk op het register."""
        return MappingProxyType(self._hashes)

    @staticmethod
    def key(asset_class: str, dataset: str, symbol: str, granularity: str) -> str:
        return f"{asset_class}/{dataset}/{symbol}/{granularity}"

    def certified_hash(
        self, asset_class: str, dataset: str, symbol: str, granularity: str
    ) -> str:
        k = self.key(asset_class, dataset, symbol, granularity)
        require(
            k in self._hashes,
            "Deze reeks staat NIET in het gecertificeerde data-register. Er "
            "wordt geen feature op ongecertificeerde data berekend: het "
            "resultaat zou per definitie INVALID zijn (sectie 7.2).",
            DataContractError,
            series=k,
            register=str(self.path),
        )
        return self._hashes[k]

    def certify(
        self,
        *,
        asset_class: str,
        dataset: str,
        symbol: str,
        granularity: str,
        observed_hash: str,
    ) -> str:
        """Vergelijk de GEMETEN inhoudshash met de geregistreerde.

        Wijkt hij af, dan is de PIT-store gewijzigd zonder dat het register is
        herbouwd. Elk feature-resultaat zou dan een hash citeren die niet klopt,
        en dat is precies de situatie die sectie 7.2 als INVALID bestempelt.
        """
        expected = self.certified_hash(asset_class, dataset, symbol, granularity)
        require(
            observed_hash == expected,
            "De gemeten data_hash wijkt af van het register. De PIT-store is "
            "gewijzigd zonder dat het register is herbouwd, of het register "
            "hoort bij een andere DVC-revisie.",
            DataContractError,
            series=self.key(asset_class, dataset, symbol, granularity),
            observed=observed_hash,
            registered=expected,
        )
        return expected


# --------------------------------------------------------------------------- #
# Input- en outputcontract
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InputSpec:
    """Wat een feature nodig heeft: uit welke datasets, en welke kolommen."""

    datasets: tuple[str, ...]
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        require(bool(self.datasets), "InputSpec zonder dataset.", DataContractError)
        require(bool(self.columns), "InputSpec zonder kolommen.", DataContractError)


@dataclass(frozen=True)
class CertifiedFrame:
    """Inputframe plus de provenance van elke reeks waaruit het is opgebouwd.

    De index is `event_ts` (UTC, oplopend, uniek); `asof_ts_ns` bepaalt wanneer
    de rij BEKEND werd en wordt door `BaseFeature.transform` de index van de
    output. Beide tijdkolommen blijven aanwezig, zodat de feature-matrix zijn
    eigen herkomst kan meeschrijven naar de store.
    """

    frame: pd.DataFrame
    symbol: str
    data_hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_hashes", tuple(sorted(self.data_hashes)))
        require(bool(self.symbol), "CertifiedFrame zonder symbool.", DataContractError)
        require(
            bool(self.data_hashes),
            "CertifiedFrame zonder data_hash. Een feature op ongecertificeerde "
            "data is INVALID (sectie 7.2); er bestaat geen voorlopige variant.",
            DataContractError,
            symbol=self.symbol,
        )
        assert_utc_index(self.frame, name=f"CertifiedFrame({self.symbol})")
        require(
            bool(self.frame.index.is_unique),
            "Dubbele tijdstempels in het inputframe; een feature op t zou dan "
            "meerdere waarden hebben.",
            DataContractError,
            symbol=self.symbol,
        )
        missing = [c for c in REQUIRED_TIME_COLUMNS if c not in self.frame.columns]
        require(
            not missing,
            "Inputframe mist de verplichte tijdkolommen uit sectie 7.2.",
            DataContractError,
            symbol=self.symbol,
            missing=missing,
        )
        for col in REQUIRED_TIME_COLUMNS:
            require(
                pd.api.types.is_integer_dtype(self.frame[col]),
                f"{col} moet int64 UTC Unix nanoseconden zijn.",
                DataContractError,
                symbol=self.symbol,
                dtype=str(self.frame[col].dtype),
            )
        require(
            bool((self.frame["asof_ts_ns"] >= self.frame["event_ts_ns"]).all()),
            "asof_ts_ns ligt VOOR event_ts_ns: de waarde zou bekend zijn geweest "
            "voordat de gebeurtenis plaatsvond.",
            CausalityViolationError,
            symbol=self.symbol,
        )

    def truncate(self, cutoff: pd.Timestamp) -> CertifiedFrame:
        """Alles NA `cutoff` weggooien. Het instrument van de invariantietest."""
        return CertifiedFrame(
            frame=self.frame.loc[self.frame.index <= cutoff],
            symbol=self.symbol,
            data_hashes=self.data_hashes,
        )


@dataclass(frozen=True)
class CertifiedPanel:
    """Een BREED panel plus provenance: rijen `asof_ts`, kolommen symbolen.

    Phase 3 (L1). De cross-sectionele tegenhanger van `CertifiedFrame`. Een
    rangschikking over assets vereist dat de waarden van alle symbolen op
    hetzelfde BESCHIKBAARHEIDSmoment naast elkaar staan; daarom is de index
    `asof_ts` en niet `event_ts`.

    Een symbool dat op `t` nog niet bestond, staat op NaN. Dat is geen gat om te
    vullen maar het feit dat het instrument er niet was; het invullen ervan zou
    survivorship bias introduceren.
    """

    values: pd.DataFrame
    data_hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_hashes", tuple(sorted(self.data_hashes)))
        assert_utc_index(self.values, name="CertifiedPanel")
        require(
            bool(self.values.index.is_unique),
            "Dubbele tijdstempels in het panel.",
            DataContractError,
        )
        require(
            len(self.values.columns) > 0,
            "Panel zonder symbolen.",
            DataContractError,
        )
        require(
            bool(self.data_hashes),
            "CertifiedPanel zonder data_hash. Een cross-sectie op "
            "ongecertificeerde data is INVALID (sectie 7.2).",
            DataContractError,
        )
        for col in self.values.columns:
            require(
                self.values[col].dtype == FEATURE_DTYPE,
                "Elke panelkolom is float64; impliciete dtype-conversie maakt "
                "cross-platform determinisme onmogelijk.",
                DataContractError,
                column=str(col),
                dtype=str(self.values[col].dtype),
            )

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(str(c) for c in self.values.columns)

    def truncate(self, cutoff: pd.Timestamp) -> CertifiedPanel:
        """Alles NA `cutoff` weggooien. Het instrument van de invariantietest."""
        return CertifiedPanel(
            values=self.values.loc[self.values.index <= cutoff],
            data_hashes=self.data_hashes,
        )


@dataclass(frozen=True)
class FeatureResult:
    """De output van een enkele feature, met zijn burn-in en provenance."""

    feature_name: str
    values: pd.DataFrame
    burn_in_period: int
    params: Mapping[str, Any]
    symbol: str
    data_hashes: tuple[tuple[str, str], ...]
    event_ts_ns: pd.Series

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.values.columns)

    @property
    def n_rows(self) -> int:
        return int(len(self.values))

    @property
    def n_nan(self) -> int:
        return int(self.values.isna().to_numpy().sum())


@dataclass(frozen=True)
class FeatureMatrix:
    """De samengestelde feature-matrix van een enkel symbool.

    `values` bevat UITSLUITEND float64 feature-kolommen, geindexeerd op
    `asof_ts`. De tijd- en symboolkolommen komen er pas bij in
    `to_store_frame`, zodat de matrix zelf geen niet-feature-kolommen bevat die
    per ongeluk als model-input kunnen belanden.
    """

    values: pd.DataFrame
    symbol: str
    burn_in_period: int
    data_hashes: tuple[tuple[str, str], ...]
    event_ts_ns: pd.Series

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.values.columns)

    def drop_burn_in(self) -> FeatureMatrix:
        """Kap de burn-in af. Het alternatief is hem expliciet documenteren."""
        return FeatureMatrix(
            values=self.values.iloc[self.burn_in_period :],
            symbol=self.symbol,
            burn_in_period=self.burn_in_period,
            data_hashes=self.data_hashes,
            event_ts_ns=self.event_ts_ns.iloc[self.burn_in_period :],
        )

    def to_store_frame(self, *, drop_burn_in: bool) -> pd.DataFrame:
        """Parquet-payload: tijdkolommen, symbool, dan de feature-kolommen.

        `drop_burn_in` is keyword-only en heeft GEEN default: of de burn-in
        wordt afgekapt is een beleidskeuze die de aanroeper expliciet maakt.
        """
        src = self.drop_burn_in() if drop_burn_in else self
        out = pd.DataFrame(
            {
                "event_ts_ns": src.event_ts_ns.to_numpy(dtype="int64"),
                "asof_ts_ns": src.values.index.astype("int64").to_numpy(),
                "symbol": src.symbol,
            }
        )
        for col in src.values.columns:
            out[col] = src.values[col].to_numpy(dtype="float64")
        return out

    def content_hash(self) -> str:
        """Inhoudshash over de matrix, identiek berekend als elke `data_hash`."""
        return dataframe_content_hash(self.to_store_frame(drop_burn_in=False))


# --------------------------------------------------------------------------- #
# BaseFeature
# --------------------------------------------------------------------------- #
class BaseFeature(abc.ABC):
    """Abstract basiscontract voor elke feature in L3.

    Een subklasse levert:
      * `name`           - stabiele identiteit; gaat mee in de `feature_hash`.
      * `input_spec`     - welke datasets en kolommen hij nodig heeft.
      * `burn_in_period` - het aantal bars waarin per constructie GEEN schatting
        bestaat. Dit is geen schatting maar een bewering die `transform`
        controleert: eindige waarden binnen de burn-in zijn een contractbreuk.
      * `output_columns` - het expliciete outputschema.
      * `_compute`       - de pure transformatie.

    `transform` wordt door subklassen niet overschreven; hij is het contract.
    """

    name: ClassVar[str] = ""
    input_spec: ClassVar[InputSpec] = InputSpec(datasets=("ohlcv",), columns=("close",))

    def __init__(self, *, params: Mapping[str, Any]) -> None:
        require(
            bool(type(self).name),
            "Feature zonder `name`; zonder stabiele identiteit is de "
            "feature_hash betekenisloos.",
            DataContractError,
            cls=type(self).__name__,
        )
        require(
            isinstance(type(self).input_spec, InputSpec),
            "Feature zonder geldige `input_spec`.",
            DataContractError,
            cls=type(self).__name__,
        )
        self._params: Mapping[str, Any] = MappingProxyType(dict(params))

    def __repr__(self) -> str:
        rendered = ", ".join(f"{k}={v!r}" for k, v in sorted(self._params.items()))
        return f"{type(self).__name__}({rendered})"

    # ------------------------------------------------------------- interface
    @property
    def params(self) -> Mapping[str, Any]:
        """De hyperparameters. Onveranderlijk; gaan mee in de `feature_hash`."""
        return self._params

    @property
    def feature_id(self) -> str:
        """Identiteit van deze GEPARAMETRISEERDE feature.

        `name` alleen is niet genoeg: drie realized-vol vensters delen dezelfde
        klasse maar zijn drie verschillende features. Twee objecten met dezelfde
        `feature_id` zijn wel echte duplicaten.
        """
        rendered = ",".join(f"{k}={self._params[k]!r}" for k in sorted(self._params))
        return f"{self.name}({rendered})"

    @property
    @abc.abstractmethod
    def burn_in_period(self) -> int:
        """Aantal bars vanaf het begin waarin de output per constructie NaN is."""

    @property
    @abc.abstractmethod
    def output_columns(self) -> tuple[str, ...]:
        """Het expliciete outputschema, in vaste volgorde."""

    @abc.abstractmethod
    def _compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        """De pure transformatie. Uitsluitend causale operaties.

        Verboden: `center=True`, `shift(-k)`, `fillna` met een sample-brede
        statistiek, `bfill`, en elke aanroep die de volledige reeks samenvat
        voordat hij hem per rij toepast.
        """

    # ------------------------------------------------------------- uitvoering
    def transform(self, source: CertifiedFrame) -> FeatureResult:
        """Bereken de feature en dwing het volledige contract af."""
        require(
            isinstance(source, CertifiedFrame),
            "transform() eist een CertifiedFrame; een kale DataFrame draagt geen "
            "provenance en mag geen feature voeden.",
            DataContractError,
            got=type(source).__name__,
        )
        frame = source.frame
        missing = [c for c in self.input_spec.columns if c not in frame.columns]
        require(
            not missing,
            "Inputframe mist kolommen uit de input_spec van deze feature.",
            DataContractError,
            feature=self.name,
            missing=missing,
            available=sorted(frame.columns)[:12],
        )
        burn_in = int(self.burn_in_period)
        require(
            burn_in >= 0,
            "Negatieve burn_in_period.",
            DataContractError,
            feature=self.name,
            burn_in=burn_in,
        )

        out = self._compute(frame)
        self._validate_output(frame, out, burn_in)

        asof_index = pd.DatetimeIndex(
            pd.to_datetime(frame["asof_ts_ns"].to_numpy(), unit="ns", utc=True),
            name=ASOF_INDEX_NAME,
        )
        require(
            bool(asof_index.is_monotonic_increasing) and bool(asof_index.is_unique),
            "De asof-index is niet oplopend of niet uniek; de feature-matrix zou "
            "dan meerdere waarden op hetzelfde beschikbaarheidsmoment hebben.",
            DataContractError,
            feature=self.name,
            symbol=source.symbol,
        )
        values = pd.DataFrame(
            {c: out[c].to_numpy(dtype="float64") for c in self.output_columns},
            index=asof_index,
        )
        return FeatureResult(
            feature_name=self.name,
            values=values,
            burn_in_period=burn_in,
            params=self._params,
            symbol=source.symbol,
            data_hashes=source.data_hashes,
            event_ts_ns=pd.Series(
                frame["event_ts_ns"].to_numpy(dtype="int64"), index=asof_index
            ),
        )

    # -------------------------------------------------------------- contract
    def _validate_output(self, frame: pd.DataFrame, out: object, burn_in: int) -> None:
        require(
            isinstance(out, pd.DataFrame),
            "_compute moet een DataFrame teruggeven.",
            DataContractError,
            feature=self.name,
            got=type(out).__name__,
        )
        assert isinstance(out, pd.DataFrame)
        require(
            bool(out.index.equals(frame.index)),
            "De output van _compute heeft niet dezelfde index als de input. Een "
            "feature mag rijen niet herordenen, toevoegen of laten vallen.",
            DataContractError,
            feature=self.name,
            n_in=len(frame),
            n_out=len(out),
        )
        expected = tuple(self.output_columns)
        require(
            tuple(out.columns) == expected,
            "De output wijkt af van het gedeclareerde outputschema.",
            DataContractError,
            feature=self.name,
            declared=expected,
            produced=tuple(out.columns),
        )
        for col in expected:
            require(
                out[col].dtype == FEATURE_DTYPE,
                "Impliciete dtype-conversie: elke feature-kolom is float64. Een "
                "andere precisie maakt cross-platform determinisme onmogelijk.",
                DataContractError,
                feature=self.name,
                column=col,
                dtype=str(out[col].dtype),
            )
        arr = out.to_numpy(dtype="float64")
        require(
            not bool(np.isinf(arr).any()),
            "De feature bevat plus/min oneindig. Dat is geen waarde maar een "
            "deling door nul of een overflow, en het propageert stilzwijgend "
            "door elk later gewicht.",
            DataContractError,
            feature=self.name,
        )
        if len(out) == 0:
            return

        head = arr[: min(burn_in, len(arr))]
        n_finite_head = int(np.isfinite(head).sum()) if head.size else 0
        require(
            n_finite_head == 0,
            "Er bestaat een EINDIGE waarde binnen de burn-in. Dat kan alleen "
            "wanneer de opstartfase is opgevuld - met een sample-brede "
            "statistiek (DI-2), een constante of een backfill. Tijdens de "
            "burn-in bestaat er geen schatting, en dat feit wordt doorgegeven.",
            CausalityViolationError,
            feature=self.name,
            burn_in=burn_in,
            n_finite_in_burn_in=n_finite_head,
        )
        self._validate_post_burn_in_nan(frame, out, burn_in)

    def _validate_post_burn_in_nan(
        self, frame: pd.DataFrame, out: pd.DataFrame, burn_in: int
    ) -> None:
        """Na de burn-in is NaN uitsluitend toegestaan bij ONTBREKENDE input.

        Zonder deze regel kan een feature zijn eigen gaten verbergen: hij is dan
        NaN op willekeurige plekken en de consument vult die later op. Met deze
        regel is elke NaN in de store herleidbaar tot een ontbrekende bar.
        """
        in_cols = [c for c in self.input_spec.columns if c in frame.columns]
        input_nan = frame[in_cols].isna().any(axis=1).astype("float64")
        excused = input_nan.rolling(burn_in + 1, min_periods=1).max().to_numpy() > 0.0
        out_nan = out.isna().any(axis=1).to_numpy()
        offending = out_nan & ~excused
        offending[: min(burn_in, len(offending))] = False
        require(
            not bool(offending.any()),
            "NaN na de burn-in zonder ontbrekende input. Een feature die zijn "
            "eigen gaten produceert is niet reproduceerbaar en verplaatst het "
            "invulbesluit naar een consument die de oorzaak niet kent.",
            DataContractError,
            feature=self.name,
            n_offending=int(offending.sum()),
            first_offending_row=int(np.argmax(offending)) if offending.any() else -1,
        )


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class FeaturePipeline:
    """Compositie van features tot een matrix voor een enkel symbool.

    De pipeline voegt niets toe aan de wiskunde: hij dwingt af dat de leden
    dezelfde input delen, dat hun outputkolommen niet botsen, en dat de
    resulterende matrix een gezamenlijke burn-in kent (het maximum). Daardoor is
    een matrix die de burn-in afkapt in ELKE kolom bruikbaar, en niet alleen in
    de snelste.

    Let op de naamsbotsing: `tradebot.features.regime.FeaturePipeline` is een
    ANDER, ouder object (Level 2+, Phase 6). Deze klasse wordt bewust niet in
    `tradebot.features.__init__` geexporteerd; importeer hem via
    `tradebot.features.base`.
    """

    def __init__(self, features: Sequence[BaseFeature]) -> None:
        require(
            len(features) > 0,
            "Lege FeaturePipeline. Een pipeline zonder features levert een lege "
            "matrix die later als 'geen signaal' wordt gelezen.",
            DataContractError,
        )
        for f in features:
            require(
                isinstance(f, BaseFeature),
                "Elk lid van een FeaturePipeline moet een BaseFeature zijn.",
                DataContractError,
                got=type(f).__name__,
            )
        ids = [f.feature_id for f in features]
        require(
            len(set(ids)) == len(ids),
            "Dubbele feature in de pipeline: dezelfde klasse met dezelfde "
            "parameters. De feature_hash zou dan niet uniek naar een definitie "
            "verwijzen.",
            DataContractError,
            feature_ids=ids,
        )
        cols: list[str] = []
        for f in features:
            cols.extend(f.output_columns)
        require(
            len(set(cols)) == len(cols),
            "Botsende outputkolommen tussen features.",
            DataContractError,
            columns=cols,
        )
        self._features: tuple[BaseFeature, ...] = tuple(features)

    @property
    def features(self) -> tuple[BaseFeature, ...]:
        return self._features

    @property
    def burn_in_period(self) -> int:
        """De gezamenlijke burn-in: het MAXIMUM over alle leden."""
        return max(int(f.burn_in_period) for f in self._features)

    @property
    def output_columns(self) -> tuple[str, ...]:
        cols: list[str] = []
        for f in self._features:
            cols.extend(f.output_columns)
        return tuple(cols)

    def transform(self, source: CertifiedFrame) -> FeatureMatrix:
        results = [f.transform(source) for f in self._features]
        index = results[0].values.index
        frames: list[pd.DataFrame] = []
        for r in results:
            require(
                bool(r.values.index.equals(index)),
                "Features leverden verschillende indices; ze zijn niet op "
                "dezelfde input berekend.",
                DataContractError,
                feature=r.feature_name,
            )
            frames.append(r.values)
        values = pd.concat(frames, axis=1)
        require(
            tuple(values.columns) == self.output_columns,
            "De samengestelde matrix wijkt af van het gedeclareerde schema.",
            DataContractError,
            declared=self.output_columns,
            produced=tuple(values.columns),
        )
        return FeatureMatrix(
            values=values,
            symbol=source.symbol,
            burn_in_period=self.burn_in_period,
            data_hashes=source.data_hashes,
            event_ts_ns=results[0].event_ts_ns,
        )


# --------------------------------------------------------------------------- #
# Gecertificeerde inputopbouw
# --------------------------------------------------------------------------- #
def load_certified_series(
    store: Any,
    register: DataRegister,
    *,
    asset_class: str,
    dataset: str,
    symbol: str,
    granularity: str,
) -> tuple[pd.DataFrame, str]:
    """Laad een PIT-reeks en verifieer zijn inhoud tegen het data-register.

    Retourneert het frame met een UTC `event_ts`-index plus de gecertificeerde
    `data_hash`. Wijkt de gemeten hash af, dan crasht deze functie: er wordt
    geen feature berekend op data waarvan de herkomst niet vaststaat.
    """
    df = store.load(asset_class, dataset, symbol, granularity)
    observed = dataframe_content_hash(df.reset_index(drop=True))
    certified = register.certify(
        asset_class=asset_class,
        dataset=dataset,
        symbol=symbol,
        granularity=granularity,
        observed_hash=observed,
    )
    out = df.copy()
    out.index = pd.DatetimeIndex(
        pd.to_datetime(out["event_ts_ns"].to_numpy(), unit="ns", utc=True),
        name="event_ts",
    )
    return out, certified


def load_certified_close_panel(
    store: Any,
    register: DataRegister,
    *,
    symbols: Sequence[str],
    granularity: str,
    asset_class: str,
) -> CertifiedPanel:
    """Bouw het brede close-panel voor de cross-sectie, met provenance.

    Geindexeerd op `asof_ts`: de close van een daily bar is pas bekend wanneer
    die bar sluit. Symbolen met een latere listing krijgen NaN vooraan - het
    instrument bestond dan niet, en dat is informatie.
    """
    require(
        len(symbols) > 0,
        "Leeg symboluniversum; er valt niets te rangschikken.",
        DataContractError,
    )
    columns: dict[str, pd.Series] = {}
    hashes: list[tuple[str, str]] = []
    for symbol in symbols:
        df, certified = load_certified_series(
            store, register, asset_class=asset_class, dataset="ohlcv",
            symbol=symbol, granularity=granularity,
        )
        asof = pd.DatetimeIndex(
            pd.to_datetime(df["asof_ts_ns"].to_numpy(), unit="ns", utc=True),
            name=ASOF_INDEX_NAME,
        )
        columns[symbol] = pd.Series(
            df["close"].to_numpy(dtype="float64"), index=asof, name=symbol
        )
        hashes.append(
            (DataRegister.key(asset_class, "ohlcv", symbol, granularity), certified)
        )
    panel = pd.DataFrame(columns).sort_index()
    panel.index.name = ASOF_INDEX_NAME
    return CertifiedPanel(values=panel.astype("float64"), data_hashes=tuple(hashes))
