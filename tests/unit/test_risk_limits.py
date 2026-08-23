"""L7 harde limieten — elke limiet zelfstandig, en de invarianten die ze delen.

Phase 4, stap 4. Elke functie hier krijgt drie soorten test: hij bindt wanneer
hij moet, hij bindt NIET wanneer hij niet moet, en wanneer hij bindt staat dat
machineleesbaar in het auditspoor met de configuratiesleutel erbij.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tradebot.risk.contract import BOOK_SCOPE, ConstraintKind
from tradebot.risk.limits import (
    apply_adv_cap,
    apply_cluster_cap,
    apply_concentration_cap,
    apply_gross_cap,
    apply_net_cap,
    apply_per_asset_cap,
    gross_exposure,
    net_exposure,
)
from tradebot.schemas.config import RiskConfig, load_config
from tradebot.utils.failfast import ConfigContractError, DataContractError

CONF = Path(__file__).resolve().parents[2] / "conf"

ALL_LIMITS = (
    lambda e: apply_per_asset_cap(e, max_position_pct=0.25),
    lambda e: apply_concentration_cap(e, max_concentration=0.40),
    lambda e: apply_gross_cap(e, cap=1.5),
    lambda e: apply_net_cap(e, cap=0.60),
)


class TestPerAssetCap:
    def test_binds_on_both_sides(self) -> None:
        out, bound = apply_per_asset_cap({"A": 0.9, "B": -0.9}, max_position_pct=0.25)
        assert out == {"A": 0.25, "B": -0.25}
        assert {c.scope for c in bound} == {"A", "B"}
        assert all(c.kind is ConstraintKind.PER_ASSET_CAP for c in bound)
        assert all(c.config_key == "risk.max_position_pct" for c in bound)

    def test_does_not_bind_below_the_cap(self) -> None:
        out, bound = apply_per_asset_cap({"A": 0.2}, max_position_pct=0.25)
        assert out == {"A": 0.2}
        assert bound == []

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, float("nan")])
    def test_invalid_config_crashes(self, bad: float) -> None:
        with pytest.raises(ConfigContractError):
            apply_per_asset_cap({"A": 0.1}, max_position_pct=bad)


class TestAdvCap:
    def test_limit_is_participation_times_adv_over_equity(self) -> None:
        out, bound = apply_adv_cap(
            {"A": 0.5}, {"A": 2_000_000.0}, participation_cap=0.01, equity=1_000_000.0
        )
        assert out["A"] == pytest.approx(0.02)
        assert bound[0].kind is ConstraintKind.ADV_CAP
        assert bound[0].config_key == "risk.adv_participation_cap"

    def test_a_deep_market_does_not_bind(self) -> None:
        out, bound = apply_adv_cap(
            {"A": 0.1}, {"A": 1e12}, participation_cap=0.01, equity=1e6
        )
        assert out == {"A": 0.1}
        assert bound == []

    def test_missing_adv_crashes_rather_than_exempting(self) -> None:
        """`risk/portfolio.py` documenteerde deze limiet maar dwong hem nooit af."""
        with pytest.raises(DataContractError):
            apply_adv_cap({"A": 0.5}, {}, participation_cap=0.01, equity=1e6)

    @pytest.mark.parametrize("bad", [0.0, float("nan"), -1.0])
    def test_invalid_adv_crashes(self, bad: float) -> None:
        with pytest.raises(DataContractError):
            apply_adv_cap({"A": 0.5}, {"A": bad}, participation_cap=0.01, equity=1e6)


class TestConcentrationCap:
    def test_solves_the_fixed_point_exactly(self) -> None:
        """Naief itereren convergeert te traag; de oplossing is gesloten."""
        out, _ = apply_concentration_cap({"A": 0.9, "B": -0.3, "C": 0.1},
                                         max_concentration=0.40)
        assert out["A"] == pytest.approx(0.2)
        assert out["B"] == pytest.approx(-0.2)
        assert out["C"] == pytest.approx(0.1)
        g = gross_exposure(out)
        assert max(abs(v) for v in out.values()) / g == pytest.approx(0.40)

    @pytest.mark.parametrize("seed", range(25))
    def test_the_invariant_holds_for_random_books(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        n = int(rng.integers(3, 9))
        book = {f"S{i}": float(rng.uniform(-1.0, 1.0)) for i in range(n)}
        out, _ = apply_concentration_cap(book, max_concentration=0.40)
        g = gross_exposure(out)
        if g > 1e-9:
            assert max(abs(v) for v in out.values()) <= 0.40 * g + 1e-9
        for s in book:
            assert abs(out[s]) <= abs(book[s]) + 1e-12

    def test_measured_explains_the_clip(self) -> None:
        """Het aandeel wordt gemeten tegen de UITEINDELIJKE gross.

        Anders zou B hier `measured=0.23` tegen `threshold=0.40` rapporteren en
        er in het auditspoor uitzien als een clip die niet had mogen gebeuren.
        """
        _, bound = apply_concentration_cap({"A": 0.9, "B": -0.3, "C": 0.1},
                                           max_concentration=0.40)
        for c in bound:
            assert c.measured > c.threshold

    def test_an_infeasible_cap_crashes_instead_of_zeroing_the_book(self) -> None:
        """n * cap < 1 heeft geen oplossing behalve nul; dat moet luid falen."""
        with pytest.raises(ConfigContractError):
            apply_concentration_cap({"A": 0.5, "B": 0.5}, max_concentration=0.40)

    def test_a_flat_book_is_left_alone(self) -> None:
        out, bound = apply_concentration_cap({"A": 0.0}, max_concentration=0.40)
        assert out == {"A": 0.0}
        assert bound == []


class TestClusterCap:
    LABELS = {"A": "l1", "B": "l1", "C": "oracle"}

    def test_the_whole_cluster_is_scaled_proportionally(self) -> None:
        out, bound = apply_cluster_cap(
            {"A": 0.5, "B": 0.5, "C": 0.2}, self.LABELS,
            max_cluster_concentration=0.60,
        )
        g = gross_exposure(out)
        assert (abs(out["A"]) + abs(out["B"])) / g == pytest.approx(0.60)
        assert out["A"] == pytest.approx(out["B"])  # proportioneel, niet willekeurig
        assert out["C"] == pytest.approx(0.2)       # het andere cluster blijft heel
        assert {c.scope for c in bound} == {"l1"}
        assert all(c.config_key == "risk.max_cluster_concentration" for c in bound)

    def test_an_unlabelled_symbol_crashes_rather_than_falling_into_an_other_bucket(
        self,
    ) -> None:
        with pytest.raises(ConfigContractError):
            apply_cluster_cap({"A": 0.5, "Z": 0.5}, self.LABELS,
                              max_cluster_concentration=0.60)

    def test_a_balanced_book_does_not_bind(self) -> None:
        out, bound = apply_cluster_cap(
            {"A": 0.3, "C": 0.3}, self.LABELS, max_cluster_concentration=0.60
        )
        assert bound == []
        assert out == {"A": 0.3, "C": 0.3}


class TestGrossCap:
    def test_scales_the_whole_book_proportionally(self) -> None:
        out, bound = apply_gross_cap({"A": 1.0, "B": -1.0}, cap=1.5)
        assert gross_exposure(out) == pytest.approx(1.5)
        assert out["A"] == pytest.approx(0.75)
        assert out["B"] == pytest.approx(-0.75)
        assert bound[0].scope == BOOK_SCOPE
        assert bound[0].measured == pytest.approx(2.0)
        assert bound[0].config_key == "risk.gross_cap"

    def test_does_not_bind_under_the_cap(self) -> None:
        out, bound = apply_gross_cap({"A": 0.5}, cap=1.5)
        assert out == {"A": 0.5}
        assert bound == []

    def test_an_infinite_cap_crashes(self) -> None:
        """`float('inf')` als default is hoe live/ zonder gross-limiet handelde."""
        with pytest.raises(ConfigContractError):
            apply_gross_cap({"A": 1.0}, cap=float("inf"))


class TestNetCap:
    def test_only_the_dominant_side_is_shrunk(self) -> None:
        out, bound = apply_net_cap({"A": 1.0, "B": 0.5, "C": -0.2}, cap=0.6)
        assert net_exposure(out) == pytest.approx(0.6)
        assert out["C"] == pytest.approx(-0.2)  # de tegenzijde blijft onaangeroerd
        assert out["A"] < 1.0 and out["B"] < 0.5
        assert bound[0].kind is ConstraintKind.NET_CAP

    def test_the_short_side_is_shrunk_when_net_is_negative(self) -> None:
        out, _ = apply_net_cap({"A": -1.0, "B": 0.1}, cap=0.6)
        assert net_exposure(out) == pytest.approx(-0.6)
        assert out["B"] == pytest.approx(0.1)

    def test_it_never_grows_the_opposite_side_to_meet_the_limit(self) -> None:
        """Sectie 5.3: de tegenzijde ophogen zou ook werken, en is verboden."""
        book = {"A": 1.0, "C": -0.2}
        out, _ = apply_net_cap(book, cap=0.6)
        for s in book:
            assert abs(out[s]) <= abs(book[s]) + 1e-12

    def test_a_market_neutral_book_does_not_bind(self) -> None:
        out, bound = apply_net_cap({"A": 1.0, "B": -1.0}, cap=0.6)
        assert bound == []
        assert out == {"A": 1.0, "B": -1.0}


class TestSharedInvariants:
    """Wat elke limiet moet doen, ongeacht welke het is."""

    BOOKS = [
        {"A": 1.0, "B": -1.0, "C": 0.5},
        {"A": 0.9, "B": 0.05, "C": -0.05},
        {"A": 0.0, "B": 0.0, "C": 0.0},
        {"A": -1.0, "B": -1.0, "C": -1.0},
    ]

    @pytest.mark.parametrize("limit_idx", range(len(ALL_LIMITS)))
    @pytest.mark.parametrize("book_idx", range(len(BOOKS)))
    def test_every_limit_only_shrinks_and_never_flips_a_sign(
        self, limit_idx: int, book_idx: int
    ) -> None:
        book = dict(self.BOOKS[book_idx])
        out, _ = ALL_LIMITS[limit_idx](book)
        for s, w in book.items():
            assert abs(out[s]) <= abs(w) + 1e-12, f"{s} groeide"
            if out[s] != 0.0:
                assert np.sign(out[s]) == np.sign(w), f"{s} klapte om"

    @pytest.mark.parametrize("limit_idx", range(len(ALL_LIMITS)))
    def test_the_source_book_is_never_mutated(self, limit_idx: int) -> None:
        # Drie assets: bij cap=0.40 is een boek van twee per constructie
        # onhaalbaar (2 x 0.40 < 1) en crasht de concentratielimiet terecht.
        book = {"A": 1.0, "B": -1.0, "C": 0.5}
        ALL_LIMITS[limit_idx](book)
        assert book == {"A": 1.0, "B": -1.0, "C": 0.5}

    @pytest.mark.parametrize("limit_idx", range(len(ALL_LIMITS)))
    def test_a_non_finite_exposure_crashes(self, limit_idx: int) -> None:
        with pytest.raises(DataContractError):
            ALL_LIMITS[limit_idx]({"A": float("nan")})

    @pytest.mark.parametrize("limit_idx", range(len(ALL_LIMITS)))
    def test_every_limit_is_pure(self, limit_idx: int) -> None:
        book = {"A": 0.9, "B": -0.4, "C": 0.2}
        first, _ = ALL_LIMITS[limit_idx](dict(book))
        second, _ = ALL_LIMITS[limit_idx](dict(book))
        assert first == second


class TestAgainstTheShippedConfig:
    def test_the_shipped_limits_are_all_feasible_for_the_live_universe(self) -> None:
        cfg = load_config(CONF / "risk/default.yaml", RiskConfig)
        book = {s: 1.0 for s in cfg.clusters}
        out, _ = apply_concentration_cap(book, max_concentration=cfg.max_concentration)
        out, _ = apply_cluster_cap(
            out, cfg.clusters, max_cluster_concentration=cfg.max_cluster_concentration
        )
        out, _ = apply_gross_cap(out, cap=cfg.gross_cap)
        out, _ = apply_net_cap(out, cap=cfg.net_cap)
        assert gross_exposure(out) <= cfg.gross_cap + 1e-9
        assert abs(net_exposure(out)) <= cfg.net_cap + 1e-9
        assert gross_exposure(out) > 0.0, "de geconfigureerde limieten sluiten het boek volledig"
