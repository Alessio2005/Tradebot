# tests/unit/test_data_sources_edges.py
"""`data/sources/`: het PIT-contract (`base`) en de reconstructie van S&P-500-lidmaatschap (`wiki_constituents`).

Geen netwerk: `requests.get` en `http_get_text` zijn vervangen.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import requests

from tradebot.data.sources import wiki_constituents as wiki
from tradebot.data.sources.base import (
    ASOF_COL,
    EVENT_COL,
    SourceMeta,
    http_get_text,
    read_csv_text,
    stamp_asof,
    validate_pit,
    write_market_parquet,
)


def _pit(dates, lag_days: int = 1, **cols) -> pd.DataFrame:
    df = pd.DataFrame({EVENT_COL: pd.to_datetime(dates, utc=True), **cols})
    return stamp_asof(df, pd.Timedelta(days=lag_days))


# --------------------------------------------------------------------------- #
# base
# --------------------------------------------------------------------------- #
class TestSourceMeta:
    def test_the_register_row_is_a_markdown_table_line(self) -> None:
        meta = SourceMeta("n", "http://u", "free", "1d", "2000-", "fixes survivorship", "note")
        assert meta.register_row() == "| n | http://u | free | 1d | 2000- | fixes survivorship | note |"


class _Response:
    def __init__(self, text: str = "ok", error: Exception | None = None) -> None:
        self.text, self._error = text, error

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error


class TestHttpGetText:
    def test_the_body_is_returned_with_the_contact_user_agent_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = {}

        def fake_get(url, headers=None, timeout=None):
            seen.update(url=url, headers=headers, timeout=timeout)
            return _Response("hello")
        monkeypatch.setattr(requests, "get", fake_get)
        assert http_get_text("http://x", timeout=7) == "hello"
        assert "User-Agent" in seen["headers"] and seen["timeout"] == 7
        http_get_text("http://x", headers={"User-Agent": "other"})
        assert seen["headers"] == {"User-Agent": "other"}

    def test_failures_are_retried_with_exponential_backoff_and_then_succeed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        script = [_Response(error=requests.HTTPError("429")), _Response(error=requests.ConnectionError("x")), _Response("fine")]
        sleeps: list[float] = []
        monkeypatch.setattr(requests, "get", lambda *a, **kw: script.pop(0))
        monkeypatch.setattr("time.sleep", sleeps.append)
        assert http_get_text("http://x") == "fine" and sleeps == [2.0, 4.0]

    def test_the_fourth_failure_is_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sleeps: list[float] = []
        monkeypatch.setattr(requests, "get", lambda *a, **kw: _Response(error=requests.Timeout("slow")))
        monkeypatch.setattr("time.sleep", sleeps.append)
        with pytest.raises(requests.Timeout):
            http_get_text("http://x")
        assert sleeps == [2.0, 4.0, 8.0]


class TestHelpers:
    def test_csv_text_is_parsed_with_pandas_options(self) -> None:
        df = read_csv_text("a;b\n1;2\n", sep=";")
        assert df.to_dict("records") == [{"a": 1, "b": 2}]

    def test_stamp_asof_adds_the_publication_lag_without_touching_the_input(self) -> None:
        raw = pd.DataFrame({EVENT_COL: ["2024-01-02"]})
        out = stamp_asof(raw, pd.Timedelta(hours=6))
        assert out[ASOF_COL].iloc[0] == pd.Timestamp("2024-01-02 06:00", tz="UTC") and ASOF_COL not in raw.columns


    def test_a_negative_publication_lag_is_refused(self) -> None:
        with pytest.raises(ValueError, match="lag must be >= 0"):
            stamp_asof(pd.DataFrame({EVENT_COL: ["2024-01-02"]}), pd.Timedelta(days=-1))


class TestValidatePit:
    def test_a_valid_frame_is_returned_sorted_by_asof(self) -> None:
        df = _pit(["2024-01-03", "2024-01-01", "2024-01-02"], x=[3, 1, 2])
        out = validate_pit(df, required=("x",))
        assert out["x"].tolist() == [1, 2, 3] and list(out.index) == [0, 1, 2]

    @pytest.mark.parametrize(("drop", "required", "message"), [
        (ASOF_COL, (), "missing column 'asof_ts'"),
        (EVENT_COL, (), "missing column 'event_ts'"),
        (None, ("nope",), "missing column 'nope'"),
    ])
    def test_missing_columns_are_a_contract_violation(self, drop, required, message) -> None:
        df = _pit(["2024-01-01"], x=[1])
        if drop:
            df = df.drop(columns=[drop])
        with pytest.raises(ValueError, match=message):
            validate_pit(df, required=required)

    def test_non_datetime_columns_are_refused(self) -> None:
        df = _pit(["2024-01-01"]).astype({EVENT_COL: str})
        with pytest.raises(ValueError, match="tz-aware UTC"):
            validate_pit(df)

    def test_only_utc_is_accepted(self) -> None:
        df = _pit(["2024-01-01"])
        df[EVENT_COL] = df[EVENT_COL].dt.tz_convert("Europe/Amsterdam")
        df[ASOF_COL] = df[ASOF_COL].dt.tz_convert("Europe/Amsterdam")
        with pytest.raises(ValueError, match="must be UTC"):
            validate_pit(df)

    def test_a_missing_timestamp_is_refused(self) -> None:
        df = _pit(["2024-01-01", "2024-01-02"])
        df.loc[1, ASOF_COL] = pd.NaT
        with pytest.raises(ValueError, match="contains NaT"):
            validate_pit(df)


class TestWriteMarketParquet:
    def test_a_validated_frame_is_written_under_its_market(self, tmp_path: Path) -> None:
        path = write_market_parquet(_pit(["2024-01-02", "2024-01-01"], x=[2, 1]), "equities", "sp500", root=tmp_path)
        assert path == tmp_path / "equities" / "sp500.parquet"
        assert pd.read_parquet(path)["x"].tolist() == [1, 2]

    def test_an_invalid_frame_is_never_written(self, tmp_path: Path) -> None:
        df = _pit(["2024-01-02"])
        df.loc[0, ASOF_COL] = df.loc[0, EVENT_COL] - pd.Timedelta(hours=1)
        with pytest.raises(ValueError, match="lookahead"):
            write_market_parquet(df, "equities", "bad", root=tmp_path)
        assert not (tmp_path / "equities").exists()


# --------------------------------------------------------------------------- #
# wiki_constituents
# --------------------------------------------------------------------------- #
def _page(changes_header: str = "multi", extra_tables: bool = False) -> str:
    current = pd.DataFrame({"Symbol": ["AAA", "BBB", "CCC"], "Security": ["a", "b", "c"]})
    rows = [["January 5, 2024", "AAA", "a", "ZZZ", "z", "x"],
            ["March 1, 2023", None, ".", "YYY", "y", "y"],
            ["not a date", "QQQ", "q", None, None, "r"]]
    if changes_header == "multi":
        cols = pd.MultiIndex.from_tuples([("Date", "Date"), ("Added", "Ticker"), ("Added", "Security"),
                                          ("Removed", "Ticker"), ("Removed", "Security"), ("Reason", "Reason")])
        changes = pd.DataFrame(rows, columns=cols)
    else:
        changes = pd.DataFrame([[r[0], r[1], r[3]] for r in rows], columns=["Date", "Added", "Removed"])
    html = current.to_html(index=False) + changes.to_html(index=False)
    return html + (pd.DataFrame({"unrelated": [1]}).to_html() if extra_tables else "")


class TestFetchMembershipEvents:
    def test_changes_and_current_members_become_point_in_time_events(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(wiki, "http_get_text", lambda url: _page())
        events = wiki.fetch_membership_events()
        by = {(r.symbol, r.action): r for r in events.itertuples()}
        assert set(by) == {("AAA", "add"), ("ZZZ", "remove"), ("YYY", "remove"), ("BBB", "add"), ("CCC", "add")}
        assert by[("AAA", "add")].event_ts == pd.Timestamp("2024-01-05", tz="UTC")
        assert by[("AAA", "add")].asof_ts == pd.Timestamp("2024-01-06", tz="UTC") and not by[("AAA", "add")].date_unknown
        for sym in ("BBB", "CCC"):                                           # zonder bekende toevoegdatum
            assert by[(sym, "add")].event_ts == pd.Timestamp("1990-01-01", tz="UTC") and by[(sym, "add")].date_unknown
        assert events[ASOF_COL].is_monotonic_increasing

    def test_a_header_without_a_ticker_sublevel_is_understood_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(wiki, "http_get_text", lambda url: _page("flat", extra_tables=True))
        events = wiki.fetch_membership_events()
        assert {"ZZZ", "YYY"} <= set(events.loc[events["action"] == "remove", "symbol"])

    def test_a_changed_page_layout_is_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(wiki, "http_get_text", lambda url: pd.DataFrame({"x": [1]}).to_html())
        with pytest.raises(ValueError, match="page layout changed"):
            wiki.fetch_membership_events()

    def test_a_changes_table_without_a_date_column_is_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        current = pd.DataFrame({"Symbol": ["AAA"]})
        changes = pd.DataFrame({"Added": ["AAA"], "Removed": ["ZZZ"]})
        monkeypatch.setattr(wiki, "http_get_text", lambda url: current.to_html() + changes.to_html())
        with pytest.raises(ValueError, match="No column matching"):
            wiki.fetch_membership_events()

    def test_the_source_is_documented_as_the_survivorship_fix(self) -> None:
        assert "survivorship" in wiki.META.survivorship and wiki.META.url == wiki._URL


class TestMembershipCalendar:
    def _events(self) -> pd.DataFrame:
        return pd.DataFrame({
            "symbol": ["AAA", "AAA", "BBB"], "action": ["add", "remove", "add"],
            EVENT_COL: pd.to_datetime(["2024-01-02", "2024-01-06", "2024-01-04"], utc=True)}
        ).pipe(stamp_asof, pd.Timedelta(days=1))

    def test_membership_starts_when_the_add_is_knowable_and_ends_when_the_removal_is(self) -> None:
        dates = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")
        cal = wiki.build_membership_calendar(self._events(), dates)
        assert cal["AAA"].tolist() == [False, False, True, True, True, True, False, False]
        assert cal["BBB"].tolist() == [False, False, False, False, True, True, True, True]

    def test_naive_dates_are_refused(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            wiki.build_membership_calendar(self._events(), pd.date_range("2024-01-01", periods=3))
