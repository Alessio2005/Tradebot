"""De gedeelde D-1-harnas: één definitie van "causaal", zes poorten.

Phase 7/8 Stage B-2. Dit bestand bevat GEEN tests. Het bevat de machinerie
waarmee de zes benoemde D-1-poorten meten, plus de bewust lekkende
referentiemodellen waarop elk van die poorten aantoonbaar rood moet worden.

WAAROM ÉÉN HARNAS EN NIET ZES KOPIEËN
=====================================
De causaliteitsdekking bestond al, verspreid over `test_feature_causality.py`
(310 tests), `test_baseline_causality.py`, `test_engine_causality.py` en
`test_filtered_only_enforcement.py`. De masterprompt (§B-2) schrijft daarom
**extraheren en hernoemen** voor, niet dupliceren: zes bestanden die elk hun
eigen `assert_bit_identical` meebrengen, zijn zes definities van causaliteit die
uit elkaar kunnen groeien, en dan bewaakt de strengste niets meer dan de
soepelste toelaat.

De helpers hieronder zijn woordelijk verplaatst uit `test_feature_causality.py`,
dat ze nu importeert in plaats van ze te bezitten.

DE LEKKENDE REFERENTIEMODELLEN
==============================
`FutureLeakingFeature` en `SampleWideScalingFeature` zijn geen curiositeiten maar
gereedschap. Elke poort in `tests/lookahead/` moet aantoonbaar rood worden op
minstens één ervan; een poort die dat niet doet, verklaart features groen zonder
iets te meten.

Zij dekken de twee klassen die in de praktijk voorkomen:

* het **grove** lek — ergens staat `shift(-1)`. Zichtbaar bij lezen.
* het **stille** lek — nergens staat een negatieve shift, elke regel oogt
  causaal, en tóch verandert elke waarde zodra er data achteraan wordt geplakt,
  omdat er met een sample-brede statistiek is genormaliseerd. Dit is de
  gevaarlijke van de twee, en het is precies DI-2.

Ref: audit §17.1; `fase_2_research_falsification.md`; `reports/phase3_exit_report.md` §2.
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from tradebot.data.pit_store import PitStore
from tradebot.features.base import BaseFeature, CertifiedFrame, DataRegister, InputSpec
from tradebot.features.microstructure import build_certified_micro_frame
from tradebot.features.volatility import log_returns
from tradebot.schemas.config import DataConfig, FeatureConfig, load_config

__all__ = [
    "DATA_CFG",
    "FEAT_CFG",
    "FUNDING_GRANULARITY",
    "FutureLeakingFeature",
    "GRANULARITY",
    "N_CUTS",
    "ROOT",
    "STORE",
    "SYMBOLS",
    "SampleWideScalingFeature",
    "TRUNCATION_SEED",
    "assert_bit_identical",
    "certified_frame",
    "cut_points",
    "leaking_references",
    "requires_store",
]

ROOT = Path(__file__).resolve().parents[2]
DATA_CFG = load_config(ROOT / "conf" / "data" / "default.yaml", DataConfig)
FEAT_CFG = load_config(ROOT / "conf" / "features" / "default.yaml", FeatureConfig)
STORE = PitStore(ROOT / DATA_CFG.pit_store_root)

SYMBOLS = list(DATA_CFG.symbols)
GRANULARITY = "1d"
FUNDING_GRANULARITY = "8h"

#: Vaste seed: een gefaalde invariantietest moet exact reproduceerbaar zijn.
TRUNCATION_SEED = 20260823
#: Drie snijpunten, zoals `fase_2_research_falsification.md` stap 5 voorschrijft.
N_CUTS = 3


def _have(dataset: str, symbol: str, granularity: str) -> bool:
    return bool(STORE.partitions("crypto", dataset, symbol, granularity))


requires_store = pytest.mark.skipif(
    not (
        _have("ohlcv", "BTCUSDT", GRANULARITY)
        and _have("funding", "BTCUSDT", FUNDING_GRANULARITY)
        and _have("open_interest", "BTCUSDT", GRANULARITY)
    ),
    reason=(
        "PIT-store is leeg. Draai eerst de Phase 1-ingestion "
        "(apps/ingest_crypto.py) en scripts/build_data_register.py."
    ),
)

_FRAME_CACHE: dict[str, CertifiedFrame] = {}


def certified_frame(symbol: str) -> CertifiedFrame:
    """Gecertificeerd inputframe (ohlcv + funding + open interest) per symbool."""
    if symbol not in _FRAME_CACHE:
        _FRAME_CACHE[symbol] = build_certified_micro_frame(
            STORE,
            DataRegister(ROOT / "artefacts" / "governance" / "data_hashes.json"),
            symbol=symbol,
            granularity=GRANULARITY,
            funding_granularity=FUNDING_GRANULARITY,
            open_interest_granularity=GRANULARITY,
            funding_tolerance=pd.Timedelta(hours=FEAT_CFG.funding_tolerance_hours),
            open_interest_tolerance=pd.Timedelta(
                hours=FEAT_CFG.open_interest_tolerance_hours
            ),
            asset_class="crypto",
        )
    return _FRAME_CACHE[symbol]


def cut_points(source: CertifiedFrame) -> list[pd.Timestamp]:
    """Drie snijpunten uit de tweede helft van de reeks, deterministisch geloot.

    De eerste helft wordt gemeden zodat elke feature zijn burn-in ruim voorbij
    is; een snijpunt binnen de burn-in zou twee NaN-blokken vergelijken en de
    toets betekenisloos maken.
    """
    idx = source.frame.index
    rng = np.random.default_rng(TRUNCATION_SEED)
    lo, hi = len(idx) // 2, len(idx) - 1
    positions = sorted(rng.choice(np.arange(lo, hi), size=N_CUTS, replace=False))
    return [idx[int(p)] for p in positions]


def assert_bit_identical(
    truncated: pd.DataFrame, full: pd.DataFrame, *, what: str, cut: pd.Timestamp
) -> None:
    """Elke waarde op of vóór `cut` moet bit-identiek zijn, NaN's inbegrepen.

    BIT-identiek, niet `np.allclose`. Een tolerantie zou de vraag verschuiven
    van *"gebruikt deze feature toekomstige data?"* naar *"hoeveel toekomstige
    data mag hij gebruiken?"*, en op die tweede vraag bestaat geen verdedigbaar
    antwoord.
    """
    expected = full.loc[full.index <= truncated.index[-1]]
    assert len(truncated) == len(expected), (
        f"{what}: rijaantal wijkt af na truncatie op {cut} "
        f"({len(truncated)} vs {len(expected)})"
    )
    assert truncated.index.equals(expected.index), f"{what}: index wijkt af na {cut}"
    for col in expected.columns:
        a = truncated[col].to_numpy(dtype="float64")
        b = expected[col].to_numpy(dtype="float64")
        same_nan = np.isnan(a) == np.isnan(b)
        assert same_nan.all(), (
            f"{what}/{col}: NaN-patroon verschuift na truncatie op {cut}; "
            f"{int((~same_nan).sum())} rij(en) afwijkend"
        )
        finite = ~np.isnan(a)
        diff = np.abs(a[finite] - b[finite])
        n_bad = int((diff != 0.0).sum())
        assert n_bad == 0, (
            f"{what}/{col}: {n_bad} waarde(n) veranderen door data NA {cut}. "
            f"max |diff| = {float(diff.max()) if diff.size else 0.0:.17g}. "
            f"Dit is een lookahead-lek; de feature is gefalsificeerd."
        )


# --------------------------------------------------------------------------- #
# De bewust lekkende referentiemodellen
# --------------------------------------------------------------------------- #
class FutureLeakingFeature(BaseFeature):
    """Het GROVE lek: gebruikt de close van de VOLGENDE bar (`shift(-1)`)."""

    name: ClassVar[str] = "deliberately_leaking_next_bar"
    input_spec: ClassVar[InputSpec] = InputSpec(datasets=("ohlcv",), columns=("close",))

    def __init__(self) -> None:
        super().__init__(params={})

    @property
    def burn_in_period(self) -> int:
        return 1

    @property
    def output_columns(self) -> tuple[str, ...]:
        return ("leak_next_close_ratio",)

    def _compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        close = frame["close"].astype("float64")
        # `ffill` dekt uitsluitend de LAATSTE rij, die na shift(-1) geen
        # opvolger meer heeft. Zonder die dekking crasht het NaN-contract van
        # BaseFeature al vóór de invariantietoets, en dan bewijst deze
        # negatieve controle niets over de toets zelf.
        out = ((close.shift(-1) / close) - 1.0).ffill()
        out.iloc[0] = np.nan
        return pd.DataFrame(
            {self.output_columns[0]: out.to_numpy(dtype="float64")}, index=frame.index
        )


class SampleWideScalingFeature(BaseFeature):
    """Het STILLE lek: normaliseert met het gemiddelde over de VOLLEDIGE sample.

    Dit is de subtiele variant, en in de praktijk de gevaarlijkste: er staat
    nergens een `shift(-1)`, elke rij oogt causaal, en toch verandert elke
    waarde zodra er data achteraan wordt geplakt.
    """

    name: ClassVar[str] = "deliberately_leaking_sample_scaler"
    input_spec: ClassVar[InputSpec] = InputSpec(datasets=("ohlcv",), columns=("close",))

    def __init__(self) -> None:
        super().__init__(params={})

    @property
    def burn_in_period(self) -> int:
        return 1

    @property
    def output_columns(self) -> tuple[str, ...]:
        return ("leak_sample_scaled_return",)

    def _compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        r = log_returns(frame["close"])
        out = r / r.std()
        return pd.DataFrame(
            {self.output_columns[0]: out.to_numpy(dtype="float64")}, index=frame.index
        )


def leaking_references() -> list[BaseFeature]:
    """De referentiemodellen waarop elke D-1-poort rood moet worden."""
    return [FutureLeakingFeature(), SampleWideScalingFeature()]
