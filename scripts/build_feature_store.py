"""Phase 2, deliverable 9 - bouw de gecertificeerde L3 feature store.

Leest `conf/features/default.yaml`, trekt de gecertificeerde reeksen uit de
PIT-store, berekent de feature-matrix per symbool en schrijft hem als
`{matrix_hash}.parquet` plus een manifest naar `features.store_root`.

Determinisme (exit criterium 3): twee runs op ongewijzigde data leveren
bit-identieke bestanden. De schrijfactie is daarom een no-op zodra het artefact
bestaat EN dezelfde inhoudshash heeft; wijkt hij af, dan crasht het script in
plaats van te overschrijven.

    python scripts/build_feature_store.py [--symbols BTCUSDT ...] [--keep-burn-in]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tradebot.data.pit_store import PitStore
from tradebot.features.base import DataRegister
from tradebot.features.positioning import build_certified_micro_frame
from tradebot.features.registry import build_default_registry, current_git_sha
from tradebot.schemas.config import DataConfig, FeatureConfig, load_config
from tradebot.utils.failfast import DataContractError, require
from tradebot.utils.hashing import dataframe_content_hash


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--granularity", default="1d")
    ap.add_argument("--funding-granularity", default="8h")
    ap.add_argument("--out", default=None, help="overschrijft features.store_root")
    ap.add_argument("--keep-burn-in", action="store_true",
                    help="burn-in NIET afkappen; alleen voor diagnostiek")
    args = ap.parse_args(argv)

    data_cfg = load_config(ROOT / "conf" / "data" / "default.yaml", DataConfig)
    feat_cfg = load_config(ROOT / "conf" / "features" / "default.yaml", FeatureConfig)
    out_root = Path(args.out) if args.out else ROOT / feat_cfg.store_root
    out_root.mkdir(parents=True, exist_ok=True)

    store = PitStore(ROOT / data_cfg.pit_store_root)
    register = DataRegister(ROOT / "artefacts" / "governance" / "data_hashes.json")
    registry = build_default_registry(feat_cfg)
    pipeline = registry.pipeline()
    git_sha = current_git_sha()

    for symbol in (args.symbols or list(data_cfg.symbols)):
        source = build_certified_micro_frame(
            store, register, symbol=symbol, granularity=args.granularity,
            funding_granularity=args.funding_granularity,
            open_interest_granularity=args.granularity,
            funding_tolerance=pd.Timedelta(hours=feat_cfg.funding_tolerance_hours),
            open_interest_tolerance=pd.Timedelta(
                hours=feat_cfg.open_interest_tolerance_hours),
            asset_class="crypto",
        )
        matrix = pipeline.transform(source)
        frame = matrix.to_store_frame(drop_burn_in=not args.keep_burn_in)
        content = dataframe_content_hash(frame)
        matrix_hash = registry.matrix_hash(
            data_hashes=source.data_hashes, git_sha=git_sha)
        target = out_root / f"{matrix_hash}.parquet"
        manifest = registry.manifest(
            data_hashes=source.data_hashes, git_sha=git_sha)
        # NaN-boekhouding: na het afkappen van de burn-in is elke resterende NaN
        # een ONTBREKENDE bar in de bron (open interest begint op sommige
        # symbolen later dan OHLCV). Machineleesbaar vastgelegd, zodat de regel
        # "burn-in afgekapt of expliciet gedocumenteerd" toetsbaar is.
        manifest.update({"symbol": symbol, "granularity": args.granularity,
                         "rows": int(len(frame)), "content_hash": content,
                         "burn_in_dropped": not args.keep_burn_in,
                         "nan_per_column": {c: int(frame[c].isna().sum())
                                            for c in matrix.columns}})
        if target.is_file():
            require(dataframe_content_hash(pd.read_parquet(target)) == content,
                    "Bestaand artefact met AFWIJKENDE inhoud onder dezelfde "
                    "matrix_hash. De feature store is append-only; overschrijven "
                    "zou een gepubliceerd resultaat stilzwijgend herschrijven.",
                    DataContractError, artefact=str(target), symbol=symbol)
        else:
            frame.to_parquet(target, index=False, compression="snappy")
        target.with_suffix(".json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(f"{symbol}: {len(frame):>5} rijen, {len(matrix.columns)} features -> "
              f"{target.name} (content_hash={content})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
