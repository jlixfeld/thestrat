"""Observable calendar reuse stays scoped to session and canonical query ranges."""

from datetime import UTC, datetime, timedelta

from polars.testing import assert_frame_equal

from thestrat import Session, TheStrat, session_for


def test_window_calls_reuse_calendar_and_session_assignment_invalidates():
    base = session_for("equity_us")
    calls = []

    def schedule(*, start, end):
        calls.append((start, end))
        return base.schedule(start=start, end=end)

    strat = TheStrat(session=Session(base.timezone, 570, schedule))
    stamp = datetime(2026, 3, 9, 14, tzinfo=UTC)
    first = strat.window(stamp, timeframes=["1h", "1d"])
    count = len(calls)
    assert_frame_equal(strat.window(stamp, timeframes=["1h", "1d"]), first)
    assert len(calls) == count
    strat.window(stamp + timedelta(minutes=1), timeframes=["1h", "1d"])
    assert len(calls) == count
    strat.session = Session(base.timezone, 540, schedule)
    changed = strat.window(stamp, timeframes=["1h", "1d"])
    assert len(calls) > count
    assert changed["window_start"].to_list() != first["window_start"].to_list()
    count = len(calls)
    strat.window(stamp + timedelta(days=40), timeframes=["1h"])
    assert len(calls) > count
