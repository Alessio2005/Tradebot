"""Beurzen voor het carryboek: wie het boek mag hosten, wat het kost, en wat dat doet.

Eigenaarsbesluit 2026-10-25: Binance (niet beschikbaar in Nederland) eruit, Kraken of Bybit
erin. `conf/execution/venues.yaml` legt per beurs vast wat een Nederlandse retailklant er
kan (spot, perps, lenen, onderpand), wat het kost (per tier), en waar dat vandaan komt.

* `load_venues` leest en valideert de profielen.
* `VenueProfile.blockers` zegt waarom een boek NIET op een beurs kan (geen perps, geen
  unified margin, munten die geen onderpand zijn, het retailplafond op perp-hefboom). Een
  boek met blokkades is op die beurs geen kandidaat.
* `basis_costs` geeft de `BasisCosts` van een beurs, voor een nieuwe trial die daar rekent.
* `fee_overlay` drukt een BEVROREN dagreeks uit tegen de fees van een andere beurs. De
  bevroren boeken rekenen 15,5 bp per paar benen (10 spot + 5,5 perp), en de framerij is
  een sluitende som, dus `net' = net + fees · (k − 1)` is exact voor de fee. Spread, impact
  en funding blijven die van Binance: dat is de aanname, niet de uitkomst.

    python -I -m tradebot.carry.venues     # -> artefacts/research/venue_comparison.json
"""
from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..systematic.harvest import BasisCosts
from ..systematic.programme_v2 import ROOT
from ..utils.failfast import ConfigContractError, DataContractError, require
from ..validation.serial_sharpe import period_sharpe

__all__ = ["BOOKS", "FROZEN_PAIR_FEE_BPS", "SCENARIOS", "VENUES_PATH", "BookNeeds", "FeeTier",
           "VenueProfile", "basis_costs", "compare", "cycle", "fee_overlay", "load_venues",
           "main"]

VENUES_PATH = Path("conf/execution/venues.yaml")
OUT = Path("artefacts/research/venue_comparison.json")
#: Wat de bevroren boeken per paar benen aan fee rekenen: 10 bp spot + 5,5 bp perp.
FROZEN_PAIR_FEE_BPS = 15.5


@dataclass(frozen=True)
class FeeTier:
    min_usd: float
    maker_bps: float
    taker_bps: float


@dataclass(frozen=True)
class VenueProfile:
    name: str
    label: str
    available_nl: bool
    perps_available_nl: bool
    spot_tiers: tuple[FeeTier, ...]
    perp_maker_bps: float | None
    perp_taker_bps: float | None
    funding_interval_hours: float | None
    retail_max_perp_leverage: float | None
    unified_margin: bool
    #: Wat als onderpand voor de perp telt; `None` = elke munt.
    collateral_assets: frozenset[str] | None
    sources: tuple[str, ...]

    def spot_fee_bps(self, capital_usd: float, *, maker: bool) -> float:
        """De spotfee van de hoogste tier die `capital_usd` haalt."""
        tier = [t for t in self.spot_tiers if capital_usd >= t.min_usd][-1]
        return tier.maker_bps if maker else tier.taker_bps

    def perp_fee_bps(self, *, maker: bool) -> float:
        fee = self.perp_maker_bps if maker else self.perp_taker_bps
        require(fee is not None, "Deze beurs biedt geen perps aan.", DataContractError,
                venue=self.name)
        assert fee is not None
        return float(fee)

    def blockers(self, req: BookNeeds) -> list[str]:
        """Waarom een boek hier niet kan (leeg = het kan, voor zover bekend)."""
        out: list[str] = []
        if not self.available_nl:
            out.append("niet beschikbaar voor Nederlandse klanten")
        if not self.perps_available_nl:
            out.append("geen perps voor Nederlandse/EER-klanten")
        if req.unified and not self.unified_margin:
            out.append("geen unified margin: lening, spot-onderpand en perps zitten niet "
                       "in één account")
        if req.collateral_coins and self.collateral_assets is not None:
            missing = sorted(set(req.collateral_coins) - self.collateral_assets)
            if missing:
                out.append("spot telt niet als perp-onderpand voor "
                           f"{', '.join(missing)}")
        cap = self.retail_max_perp_leverage
        if cap is not None and req.perp_leverage > cap + 1e-9:
            out.append(f"perp-hefboom {req.perp_leverage:.2f} boven het retailplafond "
                       f"{cap:.1f} (ESMA: perps als CFD); professionele status of een "
                       f"been van hooguit {req.max_leg_at(cap):.2f}x")
        return out


@dataclass(frozen=True)
class BookNeeds:
    """Wat een carryboek van een beurs vraagt.

    `perp_leverage`: perp-notional / het onderpand dat de short dekt. `collateral_coins`:
    de munten die als spot-onderpand voor de short moeten tellen (leeg = de short heeft
    eigen USD-marge). `unified`: lening en onderpand in één account (v6-v8)."""

    perp_leverage: float
    unified: bool
    collateral_coins: tuple[str, ...]
    leg: float

    def max_leg_at(self, cap: float) -> float:
        """Het grootste been (fractie van de equity) met aparte USD-marge onder `cap`:
        been + been / cap <= 1."""
        return 1.0 / (1.0 + 1.0 / cap)


def _profile(name: str, raw: Mapping[str, Any]) -> VenueProfile:
    tiers = tuple(FeeTier(float(t["min_usd"]), float(t["maker_bps"]), float(t["taker_bps"]))
                  for t in raw["spot_tiers"])
    require(len(tiers) >= 1 and tiers[0].min_usd == 0.0 and all(
        a.min_usd < b.min_usd for a, b in pairwise(tiers)),
        "De spot-tiers beginnen bij 0 en lopen strikt op.", ConfigContractError, venue=name)
    require(not raw["perps_available_nl"] or (raw["perp_maker_bps"] is not None
                                              and raw["perp_taker_bps"] is not None),
            "Een beurs met perps heeft perp-fees.", ConfigContractError, venue=name)
    return VenueProfile(
        name=name, label=str(raw["label"]), available_nl=bool(raw["available_nl"]),
        perps_available_nl=bool(raw["perps_available_nl"]), spot_tiers=tiers,
        perp_maker_bps=raw["perp_maker_bps"], perp_taker_bps=raw["perp_taker_bps"],
        funding_interval_hours=raw["funding_interval_hours"],
        retail_max_perp_leverage=raw["retail_max_perp_leverage"],
        unified_margin=bool(raw["unified_margin"]),
        collateral_assets=(None if raw["collateral_assets"] == "all"
                           else frozenset(raw["collateral_assets"])),
        sources=tuple(raw["sources"]))


def load_venues(path: Path | None = None) -> dict[str, VenueProfile]:
    doc = yaml.safe_load((path or ROOT / VENUES_PATH).read_text(encoding="utf-8"))
    return {name: _profile(name, raw) for name, raw in doc["venues"].items()}


def basis_costs(base: BasisCosts, venue: VenueProfile, *, capital_usd: float,
                maker: bool) -> BasisCosts:
    """`base` met de fees van `venue`; spread, impact en boekgrootte blijven."""
    return BasisCosts(
        perp=replace(base.perp, taker_fee=venue.perp_fee_bps(maker=maker) * 1e-4),
        spot=replace(base.spot, taker_fee=venue.spot_fee_bps(capital_usd, maker=maker) * 1e-4))


def fee_overlay(frame: pd.DataFrame, pair_fee_bps: float) -> pd.Series:
    """De netto dagrendementen van een bevroren boek tegen een andere fee per paar benen."""
    require({"net", "fees"} <= set(frame.columns), "Een dagreeks met net en fees.",
            DataContractError)
    return frame["net"] + frame["fees"] * (float(pair_fee_bps) / FROZEN_PAIR_FEE_BPS - 1.0)


#: (venue, aangehouden vermogen USD, maker?) per scenario.
SCENARIOS: dict[str, tuple[str, float, bool]] = {
    "binance_model": ("binance", 1e6, False),
    "kraken_entry_taker": ("kraken_eu", 0.0, False),
    "kraken_1m_taker": ("kraken_eu", 1e6, False),
    "kraken_1m_maker": ("kraken_eu", 1e6, True),
}

#: Wat elk bevroren carryboek vraagt: (programma, kandidaat, behoefte).
#: * v5-H3 (twee wallets, BTC/ETH): 0,70 spot, 0,30 USD-marge -> perp 0,70 / 0,30 = 2,33.
#: * v5-H3 met de spot ALS ONDERPAND in de futureswallet (Kraken EU: BTC en ETH tellen):
#:   dezelfde regels en hetzelfde been, maar de short is gedekt door 0,70 spot + 0,30 USD,
#:   dus perp / onderpand = 0,70. Zelfde dagreeks; alleen de walletindeling verschilt.
#: * v6-K1 (unified, 1x per been, top-50): spot van elke munt dekt de short.
#: * v7-S2, v8-T2: tot 2x per been, met een USD-lening in hetzelfde account.
ALTS = ("altcoins",)
MAJORS = ("BTC", "ETH")
BOOKS: dict[str, tuple[str, str, BookNeeds]] = {
    "H3_BASIS_MAJORS": ("robust_book_v5", "H3_BASIS_MAJORS",
                        BookNeeds(0.70 / 0.30, False, (), 0.70)),
    "H3_BASIS_MAJORS_COLLATERAL": ("robust_book_v5", "H3_BASIS_MAJORS",
                                   BookNeeds(0.70, False, MAJORS, 0.70)),
    "K1_CARRY_PM_1X": ("robust_book_v6", "K1_CARRY_PM_1X",
                       BookNeeds(1.25, False, MAJORS + ALTS, 1.0)),
    "S2_CARRY_PM_TRANCHE": ("robust_book_v7", "S2_CARRY_PM_TRANCHE",
                            BookNeeds(1.25, True, MAJORS + ALTS, 2.0)),
    "T2_CARRY_TRANCHE_JUMPCAP": ("robust_book_v8", "T2_CARRY_TRANCHE_JUMPCAP",
                                 BookNeeds(1.25, True, MAJORS + ALTS, 2.0)),
}


def cycle(root: Path, prog: str, cand: str) -> pd.DataFrame:
    """De bevroren dagreeks van een kandidaat over de hele cyclus 2020-04 .. 2026-09."""
    parts = [pd.read_csv(root / "artefacts/research" / prog / f"{cand}_daily_{w}.csv",
                         index_col=0, parse_dates=True)
             for w in ("backcast", "w_dev", "holdout")]
    return pd.concat(parts).sort_index()


def _summary(r: pd.Series) -> dict[str, Any]:
    eq = (1.0 + r).cumprod()
    years = len(r) / 365.0
    yearly = {str(y): float((1.0 + c).prod() - 1.0) for y, c in r.groupby(r.index.year)}
    sm = period_sharpe(r, "ME")
    return {"cagr": float(eq.iloc[-1] ** (1.0 / years) - 1.0),
            "sharpe_monthly": float(sm) if math.isfinite(sm) else None,
            "worst_year": min(yearly.values()), "years_negative": sorted(
                y for y, v in yearly.items() if v < 0.0), "yearly_return": yearly}


def compare(root: Path = ROOT, venues: Mapping[str, VenueProfile] | None = None) -> dict:
    """Elk bevroren carryboek, de hele cyclus 2020-04 .. 2026-09, per beursscenario."""
    venues = venues or load_venues(root / VENUES_PATH)
    out: dict[str, Any] = {
        "what": "Bevroren dagreeksen tegen de fees van een andere beurs (fee_overlay). "
                "Spread, impact en funding blijven die van Binance; het beleid (drempels) is "
                "niet opnieuw gekozen voor de hogere kosten.",
        "scenarios": {k: {"venue": v, "capital_usd": c, "maker": m,
                          "pair_fee_bps": venues[v].spot_fee_bps(c, maker=m)
                          + venues[v].perp_fee_bps(maker=m)}
                      for k, (v, c, m) in SCENARIOS.items()},
        "books": {}}
    for key, (prog, cand, needs) in BOOKS.items():
        frame = cycle(root, prog, cand)
        rec: dict[str, Any] = {"blockers": {name: v.blockers(needs)
                                            for name, v in venues.items()}}
        for scen, spec in out["scenarios"].items():
            rec[scen] = _summary(fee_overlay(frame, spec["pair_fee_bps"]))
        out["books"][key] = rec
    return out


def main(argv: Sequence[str]) -> dict:
    require(len(argv) == 0, "Gebruik: python -I -m tradebot.carry.venues", DataContractError)
    out = compare(ROOT)
    (ROOT / OUT).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{OUT}: {len(out['books'])} boeken x {len(out['scenarios'])} scenario's")
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
