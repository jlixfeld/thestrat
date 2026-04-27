"""Instrument-type session presets for timeframe alignment.

Aggregating 1m bars into hour-and-above timeframes only makes sense if
each "day" starts at the right moment — otherwise daily/weekly/monthly
buckets split the actual trading session in half.

Each `InstrumentType` resolves to a `Session(timezone, anchor_minutes)`
where `anchor_minutes` is minutes-since-midnight in `timezone` for the
day boundary. Hour-and-above buckets (1h, 4h, 6h, 12h, 1d, 1w, 1m, 1q,
1y) align to that anchor.

Conventions used here come from the Strat methodology and standard
US market practice:

- **EQUITY_US**: 9:30 ET — the regular session open. Per the
  StratPlaybookMCP "Bottom of the Hour" concept, hourly buckets at
  9:30 / 10:30 / ... are preferred because news + open both happen
  at :30.
- **FUTURES_CME**: 18:00 ET — Globex Sunday open. Each CME futures
  trading day runs ~Sunday 18:00 ET → Friday 17:00 ET.
- **CRYPTO**: 00:00 UTC — 24/7 markets, no real session.
- **FX**: 17:00 ET — the conventional NY 5pm rollover.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True)
class Session:
    """Where each "day" starts for a given instrument's market.

    `timezone` is an IANA tz database name (e.g. "America/New_York", "UTC").
    `anchor_minutes` is minutes-since-midnight in that timezone for the
    day boundary — e.g. 570 for 9:30 ET, 1080 for 18:00 ET.
    """

    timezone: str
    anchor_minutes: int


class InstrumentType(StrEnum):
    """Coarse instrument categories that share session conventions."""

    EQUITY_US = "equity_us"
    FUTURES_CME = "futures_cme"
    CRYPTO = "crypto"
    FX = "fx"


SESSIONS: dict[InstrumentType, Session] = {
    InstrumentType.EQUITY_US: Session(timezone="America/New_York", anchor_minutes=570),
    InstrumentType.FUTURES_CME: Session(timezone="America/New_York", anchor_minutes=1080),
    InstrumentType.CRYPTO: Session(timezone="UTC", anchor_minutes=0),
    InstrumentType.FX: Session(timezone="America/New_York", anchor_minutes=1020),
}


def session_for(instrument_type: InstrumentType) -> Session:
    """Look up the canonical Session for an InstrumentType."""
    return SESSIONS[instrument_type]
