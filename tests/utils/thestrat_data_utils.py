"""
Test data utilities for TheStrat module tests.

Provides consistent data generation functions across all test modules.
"""

from datetime import datetime, timedelta

import polars as pl


def create_timestamp_series(
    start: str = "2023-01-01", periods: int = 10, freq_minutes: int = 60, timezone: str | None = None
) -> list[datetime]:
    """
    Create a series of timestamps for test data.

    Args:
        start: Start date string
        periods: Number of timestamps to generate
        freq_minutes: Frequency in minutes between timestamps
        timezone: Timezone string (optional)

    Returns:
        List of datetime objects
    """
    if len(start) == 10:  # Date only (YYYY-MM-DD)
        start_dt = datetime.fromisoformat(f"{start} 00:00:00")
    else:  # DateTime string (YYYY-MM-DD HH:MM:SS)
        start_dt = datetime.fromisoformat(start)

    timestamps = [start_dt + timedelta(minutes=i * freq_minutes) for i in range(periods)]

    if timezone:
        import pytz

        tz = pytz.timezone(timezone)
        timestamps = [tz.localize(ts) for ts in timestamps]

    return timestamps


def create_ohlc_data(
    periods: int = 10,
    base_price: float = 100.0,
    start: str = "2023-01-01",
    freq_minutes: int = 60,
    symbol: str | None = None,
    timezone: str | None = None,
) -> pl.DataFrame:
    """
    Create realistic OHLC test data.

    Args:
        periods: Number of bars to generate
        base_price: Starting price level
        start: Start date string
        freq_minutes: Minutes between bars
        symbol: Optional symbol name
        timezone: Optional timezone

    Returns:
        Polars DataFrame with OHLC data
    """
    timestamps = create_timestamp_series(start, periods, freq_minutes, timezone)

    # Generate realistic price movements
    data = {
        "timestamp": timestamps,
        "open": [base_price + i * 0.1 + (i % 5) * 0.05 for i in range(periods)],
        "high": [base_price + i * 0.1 + (i % 5) * 0.05 + 0.5 for i in range(periods)],
        "low": [base_price + i * 0.1 + (i % 5) * 0.05 - 0.5 for i in range(periods)],
        "close": [base_price + i * 0.1 + (i % 5) * 0.05 + 0.2 for i in range(periods)],
        "volume": [1000 + i * 10 + (i % 3) * 50 for i in range(periods)],
    }

    if symbol:
        data["symbol"] = [symbol] * periods

    return pl.DataFrame(data)


def create_trend_data(periods: int = 10) -> pl.DataFrame:
    """Create trending data with clear swing points."""
    timestamps = create_timestamp_series("2023-01-01", periods, freq_minutes=1440)  # Daily

    # Create data with clear peaks and valleys
    # Pattern: up-up-down-down-up-up-peak-down-down-up
    base_prices = [100, 102, 104, 103, 101, 103, 105, 110, 107, 105][:periods]

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": base_prices,
            "high": [p + 1.0 for p in base_prices],  # High is always base + 1
            "low": [p - 1.0 for p in base_prices],  # Low is always base - 1
            "close": [p + 0.5 for p in base_prices],  # Close is always base + 0.5
            "volume": [1000] * periods,
        }
    )


def create_pattern_data() -> pl.DataFrame:
    """Create data with specific pattern characteristics."""
    timestamps = create_timestamp_series("2023-01-01", 15, freq_minutes=1440)  # Daily

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100 + i * 0.5 for i in range(15)],
            "high": [101 + i * 0.5 for i in range(15)],
            "low": [99 + i * 0.5 for i in range(15)],
            "close": [100.2 + i * 0.5 for i in range(15)],
            "volume": [1000] * 15,
        }
    )


def create_gap_data() -> pl.DataFrame:
    """Create data with gap patterns."""
    timestamps = create_timestamp_series("2023-01-01", 5, freq_minutes=1440)  # Daily

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100, 105, 95, 103, 97],  # Gap up, gap down, normal, gap down
            "high": [102, 107, 97, 105, 99],
            "low": [98, 103, 93, 101, 95],
            "close": [101, 106, 96, 104, 98],
            "volume": [1000] * 5,
        }
    )


def create_price_analysis_data() -> pl.DataFrame:
    """Create data for price analysis testing."""
    timestamps = create_timestamp_series("2023-01-01", 5, freq_minutes=1440)  # Daily

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100, 102, 104, 106, 108],
            "high": [102, 104, 106, 108, 110],
            "low": [98, 100, 102, 104, 106],
            "close": [99, 103, 105, 107, 109],  # Various positions within range
            "volume": [1000] * 5,
        }
    )


def create_ath_atl_data() -> pl.DataFrame:
    """Create data for ATH/ATL testing."""
    timestamps = create_timestamp_series("2023-01-01", 8, freq_minutes=1440)  # Daily

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100, 102, 98, 105, 103, 110, 108, 95],
            "high": [101, 103, 99, 106, 104, 112, 109, 96],  # ATH at index 5 (112)
            "low": [99, 101, 97, 104, 102, 109, 107, 94],  # ATL at index 7 (94)
            "close": [100.5, 102.5, 98.5, 105.5, 103.5, 111.5, 108.5, 95.5],
            "volume": [1000] * 8,
        }
    )


def create_large_dataset(periods: int = 1000, freq_minutes: int = 1) -> pl.DataFrame:
    """Create large dataset for performance testing."""
    timestamps = create_timestamp_series("2023-01-01", periods, freq_minutes)

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0 + i * 0.001 for i in range(periods)],
            "high": [100.5 + i * 0.001 for i in range(periods)],
            "low": [99.5 + i * 0.001 for i in range(periods)],
            "close": [100.2 + i * 0.001 for i in range(periods)],
            "volume": [1000 + i for i in range(periods)],
        }
    )


def create_market_hours_data(symbol: str = "AAPL") -> pl.DataFrame:
    """Create realistic market hours data."""
    # 6.5 hours of market data (9:30 AM - 4:00 PM)
    timestamps = create_timestamp_series("2023-01-03 09:30:00", 390, freq_minutes=1)

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": [symbol] * 390,
            "open": [150.0 + i * 0.01 + (i % 10) * 0.05 for i in range(390)],
            "high": [150.5 + i * 0.01 + (i % 10) * 0.05 for i in range(390)],
            "low": [149.5 + i * 0.01 + (i % 10) * 0.05 for i in range(390)],
            "close": [150.2 + i * 0.01 + (i % 10) * 0.05 for i in range(390)],
            "volume": [1000 + i * 5 + (i % 20) * 100 for i in range(390)],
        }
    )


def create_crypto_data(symbol: str = "BTC-USD") -> pl.DataFrame:
    """Create 24/7 crypto data."""
    # 2 days * 24 hours
    timestamps = create_timestamp_series("2023-01-01", 48, freq_minutes=60, timezone="UTC")

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": [symbol] * 48,
            "open": [40000 + i * 50 + (i % 4) * 100 for i in range(48)],
            "high": [40200 + i * 50 + (i % 4) * 100 for i in range(48)],
            "low": [39800 + i * 50 + (i % 4) * 100 for i in range(48)],
            "close": [40100 + i * 50 + (i % 4) * 100 for i in range(48)],
            "volume": [10 + i * 0.5 for i in range(48)],
        }
    )


def create_forex_data(symbol: str = "EUR/USD") -> pl.DataFrame:
    """Create forex market data."""
    # 5 days * 48 half-hour bars per day
    timestamps = create_timestamp_series("2023-01-02", 240, freq_minutes=30)

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": [symbol] * 240,
            "open": [1.0500 + i * 0.0001 + (i % 8) * 0.0005 for i in range(240)],
            "high": [1.0510 + i * 0.0001 + (i % 8) * 0.0005 for i in range(240)],
            "low": [1.0490 + i * 0.0001 + (i % 8) * 0.0005 for i in range(240)],
            "close": [1.0505 + i * 0.0001 + (i % 8) * 0.0005 for i in range(240)],
            "volume": [100000 + i * 1000 for i in range(240)],
        }
    )


def create_dst_transition_data(transition_type: str = "spring_forward", timezone: str = "US/Eastern") -> pl.DataFrame:
    """
    Create data spanning DST transition periods for aggregation testing.

    Args:
        transition_type: 'spring_forward' (2AM -> 3AM skip) or 'fall_back' (2AM -> 1AM repeat)
        timezone: Timezone for the DST transition

    Returns:
        DataFrame with minute-level data across DST transition
    """
    from datetime import datetime

    import pytz

    tz = pytz.timezone(timezone)

    if transition_type == "spring_forward":
        # Spring forward: March 12, 2023 for US/Eastern
        # Generate data from 1:00 AM to 4:00 AM (skipping 2:00-3:00 AM)
        timestamps = []

        # 1:00 AM - 1:59 AM (60 minutes)
        for i in range(60):
            naive_time = datetime(2023, 3, 12, 1, i)
            aware_time = tz.localize(naive_time)
            timestamps.append(aware_time.replace(tzinfo=None))  # Store as naive for polars

        # Skip 2:00 AM - 2:59 AM (DST transition happens)

        # 3:00 AM - 3:59 AM (60 minutes)
        for i in range(60):
            naive_time = datetime(2023, 3, 12, 3, i)
            aware_time = tz.localize(naive_time)
            timestamps.append(aware_time.replace(tzinfo=None))

    else:  # fall_back
        # Fall back: November 5, 2023 for US/Eastern
        # Generate data with repeated 1:00-2:00 AM hour
        timestamps = []

        # 12:00 AM - 12:59 AM (60 minutes)
        for i in range(60):
            naive_time = datetime(2023, 11, 5, 0, i)
            aware_time = tz.localize(naive_time)
            timestamps.append(aware_time.replace(tzinfo=None))

        # First 1:00 AM - 1:59 AM (60 minutes, before fall back)
        for i in range(60):
            naive_time = datetime(2023, 11, 5, 1, i)
            aware_time = tz.localize(naive_time, is_dst=True)  # Before fall back
            timestamps.append(aware_time.replace(tzinfo=None))

        # Second 1:00 AM - 1:59 AM (60 minutes, after fall back)
        for i in range(60):
            naive_time = datetime(2023, 11, 5, 1, i)
            aware_time = tz.localize(naive_time, is_dst=False)  # After fall back
            timestamps.append(aware_time.replace(tzinfo=None))

        # 2:00 AM - 2:59 AM (60 minutes)
        for i in range(60):
            naive_time = datetime(2023, 11, 5, 2, i)
            aware_time = tz.localize(naive_time)
            timestamps.append(aware_time.replace(tzinfo=None))

    num_bars = len(timestamps)
    base_price = 150.0

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [base_price + i * 0.01 for i in range(num_bars)],
            "high": [base_price + 0.5 + i * 0.01 for i in range(num_bars)],
            "low": [base_price - 0.5 + i * 0.01 for i in range(num_bars)],
            "close": [base_price + 0.2 + i * 0.01 for i in range(num_bars)],
            "volume": [1000 + i * 10 for i in range(num_bars)],
        }
    )


def create_multi_timezone_data(timezones: list = None) -> dict:
    """
    Create identical market data in multiple timezones for aggregation testing.

    Args:
        timezones: List of timezone strings

    Returns:
        Dictionary mapping timezone -> DataFrame
    """
    if timezones is None:
        timezones = ["UTC", "US/Eastern", "US/Pacific", "Asia/Tokyo"]

    base_price = 100.0
    num_bars = 48  # 2 days of hourly data

    timezone_data = {}

    for tz_str in timezones:
        timestamps = create_timestamp_series("2023-06-15 12:00:00", num_bars, freq_minutes=60, timezone=tz_str)

        timezone_data[tz_str] = pl.DataFrame(
            {
                "timestamp": timestamps,
                "timezone": [tz_str] * num_bars,
                "open": [base_price + i * 0.1 for i in range(num_bars)],
                "high": [base_price + 0.5 + i * 0.1 for i in range(num_bars)],
                "low": [base_price - 0.5 + i * 0.1 for i in range(num_bars)],
                "close": [base_price + 0.2 + i * 0.1 for i in range(num_bars)],
                "volume": [1000 + i * 50 for i in range(num_bars)],
            }
        )

    return timezone_data


def create_edge_case_data(case_type: str = "identical_prices") -> pl.DataFrame:
    """
    Create specific edge case scenarios for testing.

    Args:
        case_type: Type of edge case - 'identical_prices', 'extreme_gaps', 'single_bar', 'missing_volume'

    Returns:
        DataFrame with the specified edge case scenario
    """
    timestamps = create_timestamp_series("2023-01-01", 10, freq_minutes=60)

    if case_type == "identical_prices":
        # All bars have identical OHLC - no price movement
        price = 100.0
        return pl.DataFrame(
            {
                "timestamp": timestamps,
                "open": [price] * 10,
                "high": [price] * 10,
                "low": [price] * 10,
                "close": [price] * 10,
                "volume": [1000] * 10,
            }
        )

    elif case_type == "extreme_gaps":
        # Large gaps between bars
        return pl.DataFrame(
            {
                "timestamp": timestamps,
                "open": [100, 200, 50, 300, 25, 400, 10, 500, 5, 600],
                "high": [105, 205, 55, 305, 30, 405, 15, 505, 10, 605],
                "low": [95, 195, 45, 295, 20, 395, 5, 495, 1, 595],
                "close": [102, 202, 52, 302, 27, 402, 12, 502, 7, 602],
                "volume": [1000] * 10,
            }
        )

    elif case_type == "single_bar":
        # Minimal dataset with just one bar
        return pl.DataFrame(
            {
                "timestamp": [timestamps[0]],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000],
            }
        )

    elif case_type == "missing_volume":
        # Dataset without volume column
        return pl.DataFrame(
            {
                "timestamp": timestamps,
                "open": [100.0 + i for i in range(10)],
                "high": [101.0 + i for i in range(10)],
                "low": [99.0 + i for i in range(10)],
                "close": [100.5 + i for i in range(10)],
            }
        )

    else:
        raise ValueError(f"Unknown edge case type: {case_type}")


def create_long_term_data(days: int = 365, freq_minutes: int = 60, symbol: str = "SPY") -> pl.DataFrame:
    """
    Create long-term dataset for aggregation testing across multiple timeframes.

    Args:
        days: Number of days of data to generate
        freq_minutes: Frequency in minutes between bars
        symbol: Symbol name

    Returns:
        DataFrame with long-term OHLC data suitable for all timeframe aggregations
    """
    periods = days * 24 * 60 // freq_minutes  # Calculate total bars needed
    timestamps = create_timestamp_series("2023-01-01", periods, freq_minutes)

    # Generate realistic price movements with trends and volatility
    import random

    random.seed(42)  # For reproducible test data

    base_price = 400.0  # Starting price similar to SPY
    prices = [base_price]

    # Generate price series with random walk + trend
    for i in range(1, periods):
        # Daily trend component (very small)
        trend = 0.001 * (i % 1440)  # Small daily trend

        # Random component
        random_change = random.uniform(-0.02, 0.02)  # +/- 2% random change

        # Volatility clustering (higher vol after big moves)
        if abs(random_change) > 0.015:
            random_change *= 1.5

        new_price = prices[-1] * (1 + trend + random_change)
        new_price = max(new_price, 1.0)  # Prevent negative prices
        prices.append(new_price)

    # Create OHLC from price series
    opens = prices[:-1]  # Use current price as next bar's open
    closes = prices[1:]  # Use next price as current bar's close

    highs = []
    lows = []
    volumes = []

    for i in range(len(opens)):
        open_price = opens[i]
        close_price = closes[i]

        # High is max of open/close + small random amount
        high = max(open_price, close_price) + random.uniform(0, 0.01) * open_price

        # Low is min of open/close - small random amount
        low = min(open_price, close_price) - random.uniform(0, 0.01) * open_price

        # Volume varies randomly
        volume = random.randint(1000000, 5000000)

        highs.append(high)
        lows.append(low)
        volumes.append(volume)

    return pl.DataFrame(
        {
            "timestamp": timestamps[1:],  # Skip first timestamp since we use prices[1:]
            "symbol": [symbol] * len(opens),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )
