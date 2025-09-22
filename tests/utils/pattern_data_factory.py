"""
Pattern Data Factory for TheStrat Signal Testing.

Provides clean, maintainable test data generation for specific signal patterns.
Each pattern has carefully crafted OHLC data designed to trigger the expected signal.
"""

from datetime import datetime, timedelta

from polars import DataFrame


class PatternDataFactory:
    """Factory class for generating test data that triggers specific signal patterns."""

    # Pattern configurations with carefully designed OHLC data
    PATTERN_CONFIGS = {
        # Rev Strat patterns: 1-2D-2U (inside, down, up)
        "1-2D-2U": {
            "description": "Inside bar, followed by 2D (down), then 2U (up) - long reversal",
            "open": [100.0, 101.5, 99.0, 100.5, 103.0],
            "high": [102.0, 101.8, 100.0, 102.0, 105.0],
            "low": [99.0, 100.5, 98.5, 100.0, 102.5],
            "close": [101.0, 101.0, 99.5, 101.5, 104.0],
        },
        # 1-2U-2D (inside, up, down)
        "1-2U-2D": {
            "description": "Inside bar, followed by 2U (up), then 2D (down) - short reversal",
            "open": [100.0, 100.5, 102.0, 99.0, 96.0],
            "high": [102.0, 101.8, 103.5, 100.0, 97.0],
            "low": [99.0, 100.0, 101.5, 97.5, 95.0],
            "close": [101.0, 101.5, 103.0, 98.0, 96.5],
        },
        # 3-bar reversal patterns: 3-1-2U (outside, inside, up)
        "3-1-2U": {
            "description": "3-bar (outside), inside bar, then 2U (up) - long reversal",
            "open": [100.0, 99.0, 100.5, 102.0, 105.0],
            "high": [102.0, 103.5, 101.8, 103.0, 107.0],
            "low": [99.0, 98.0, 100.0, 101.5, 104.0],
            "close": [101.0, 103.0, 101.0, 102.5, 106.0],
        },
        # 3-2D-2U (outside, down, up)
        "3-2D-2U": {
            "description": "3-bar (outside), 2D (down), then 2U (up) - long reversal",
            "open": [100.0, 99.0, 98.0, 99.5, 102.0],
            "high": [102.0, 103.5, 99.0, 101.0, 104.0],
            "low": [99.0, 98.0, 97.5, 99.0, 101.5],
            "close": [101.0, 103.0, 98.5, 100.5, 103.5],
        },
        # 2D-1-2U (down, inside, up)
        "2D-1-2U": {
            "description": "2D (down), inside bar, then 2U (up) - long reversal",
            "open": [100.0, 98.0, 98.5, 100.0, 103.0],
            "high": [102.0, 99.5, 99.8, 102.0, 105.0],
            "low": [99.0, 97.5, 98.0, 99.5, 102.5],
            "close": [101.0, 98.0, 99.0, 101.5, 104.0],
        },
        # 2D-2U (down, up)
        "2D-2U": {
            "description": "2D (down) followed by 2U (up) - long reversal",
            "open": [100.0, 98.0, 100.5, 103.0, 106.0],
            "high": [102.0, 99.5, 102.0, 105.0, 108.0],
            "low": [99.0, 97.5, 100.0, 102.5, 105.5],
            "close": [101.0, 98.0, 101.5, 104.0, 107.0],
        },
        # Short reversal patterns
        "3-1-2D": {
            "description": "3-bar (outside), inside bar, then 2D (down) - short reversal",
            "open": [100.0, 99.0, 100.5, 98.0, 95.0],
            "high": [102.0, 103.5, 101.8, 99.0, 96.0],
            "low": [99.0, 98.0, 100.0, 97.5, 94.0],
            "close": [101.0, 103.0, 101.0, 98.5, 95.5],
        },
        "3-2U-2D": {
            "description": "3-bar (outside), 2U (up), then 2D (down) - short reversal",
            "open": [100.0, 99.0, 102.0, 99.5, 96.0],
            "high": [102.0, 103.5, 104.0, 100.0, 97.0],
            "low": [99.0, 98.0, 101.5, 98.0, 95.0],
            "close": [101.0, 103.0, 103.5, 99.0, 96.5],
        },
        "2U-1-2D": {
            "description": "2U (up), inside bar, then 2D (down) - short reversal",
            "open": [100.0, 102.0, 102.5, 98.0, 95.0],
            "high": [102.0, 104.0, 103.8, 99.0, 96.0],
            "low": [99.0, 101.5, 102.0, 97.5, 94.0],
            "close": [101.0, 103.0, 103.0, 98.5, 95.5],
        },
        "2U-2D": {
            "description": "2U (up) followed by 2D (down) - short reversal",
            "open": [100.0, 102.0, 98.0, 95.0, 92.0],
            "high": [102.0, 104.0, 99.5, 96.0, 93.0],
            "low": [99.0, 101.5, 97.5, 94.0, 91.0],
            "close": [101.0, 103.0, 98.5, 95.5, 92.5],
        },
        # Continuation patterns
        "2U-2U": {
            "description": "2U (up) followed by another 2U (up) - long continuation",
            "open": [100.0, 102.0, 104.0, 106.0, 108.0],
            "high": [102.0, 104.0, 106.0, 108.0, 110.0],
            "low": [99.0, 101.5, 103.5, 105.5, 107.5],
            "close": [101.0, 103.0, 105.0, 107.0, 109.0],
        },
        "2U-1-2U": {
            "description": "2U (up), inside bar, then 2U (up) - long continuation",
            "open": [100.0, 102.0, 102.5, 104.0, 106.0],
            "high": [102.0, 104.0, 103.8, 106.0, 108.0],
            "low": [99.0, 101.5, 102.0, 103.5, 105.5],
            "close": [101.0, 103.0, 103.0, 105.0, 107.0],
        },
        "2D-2D": {
            "description": "2D (down) followed by another 2D (down) - short continuation",
            "open": [100.0, 98.0, 96.0, 94.0, 92.0],
            "high": [102.0, 99.5, 97.0, 95.0, 93.0],
            "low": [99.0, 97.5, 95.5, 93.5, 91.5],
            "close": [101.0, 98.0, 96.5, 94.5, 92.5],
        },
        "2D-1-2D": {
            "description": "2D (down), inside bar, then 2D (down) - short continuation",
            "open": [100.0, 98.0, 98.5, 96.0, 94.0],
            "high": [102.0, 99.5, 99.8, 97.0, 95.0],
            "low": [99.0, 97.5, 98.0, 95.5, 93.5],
            "close": [101.0, 98.0, 99.0, 96.5, 94.5],
        },
        # Context reversal patterns (require complex market context)
        "3-2U": {
            "description": "3-bar followed by 2U - context-dependent reversal pattern",
            "open": [95.0, 97.0, 100.0, 102.0, 104.0],
            "high": [97.0, 99.0, 103.0, 104.0, 106.0],
            "low": [94.0, 96.0, 99.0, 101.5, 103.5],
            "close": [96.5, 98.5, 102.5, 103.5, 105.0],
        },
        "3-2D": {
            "description": "3-bar followed by 2D - context-dependent reversal pattern",
            "open": [95.0, 98.0, 100.0, 99.0, 97.0],
            "high": [98.0, 100.0, 103.0, 100.5, 98.0],
            "low": [94.5, 97.5, 98.5, 98.0, 96.5],
            "close": [97.5, 99.5, 102.5, 99.0, 97.5],
        },
    }

    @classmethod
    def create(cls, pattern_type: str, extend_data: bool = True) -> DataFrame:
        """
        Create test data for a specific signal pattern.

        Args:
            pattern_type: The pattern type (e.g., "1-2D-2U", "2U-2D")
            extend_data: Whether to extend the pattern with additional bars for reliable detection

        Returns:
            Polars DataFrame with OHLC data designed to trigger the pattern

        Raises:
            ValueError: If pattern_type is not supported
        """
        if pattern_type not in cls.PATTERN_CONFIGS:
            available_patterns = list(cls.PATTERN_CONFIGS.keys())
            raise ValueError(f"Pattern '{pattern_type}' not supported. Available patterns: {available_patterns}")

        config = cls.PATTERN_CONFIGS[pattern_type].copy()

        # Extract the OHLC data
        base_data = {
            "open": config["open"],
            "high": config["high"],
            "low": config["low"],
            "close": config["close"],
        }

        if extend_data:
            # Add additional bars to ensure we have enough data for signal detection
            for key in ["open", "high", "low", "close"]:
                last_val = base_data[key][-1]
                # Add 15 more bars with trending data after the pattern
                extension = [last_val + i for i in range(1, 16)]
                base_data[key].extend(extension)

        data_length = len(base_data["open"])

        # Create DataFrame with timestamps and volume
        data = DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 9, 30) + timedelta(minutes=i * 5) for i in range(data_length)],
                "open": base_data["open"],
                "high": base_data["high"],
                "low": base_data["low"],
                "close": base_data["close"],
                "volume": [1000] * data_length,
            }
        )

        return data

    @classmethod
    def get_pattern_description(cls, pattern_type: str) -> str:
        """
        Get the description for a specific pattern.

        Args:
            pattern_type: The pattern type

        Returns:
            Human-readable description of the pattern
        """
        if pattern_type not in cls.PATTERN_CONFIGS:
            return f"Unknown pattern: {pattern_type}"

        return cls.PATTERN_CONFIGS[pattern_type]["description"]

    @classmethod
    def get_available_patterns(cls) -> list[str]:
        """Get list of all available pattern types."""
        return list(cls.PATTERN_CONFIGS.keys())

    @classmethod
    def is_context_pattern(cls, pattern_type: str) -> bool:
        """
        Check if a pattern requires complex market context for reliable detection.

        These patterns may not be consistently detected with simple test data
        and require more sophisticated market conditions.

        Args:
            pattern_type: The pattern type to check

        Returns:
            True if the pattern is context-dependent
        """
        context_patterns = ["3-2U", "3-2D", "2D-1-2U", "2D-1-2D", "2U-1-2D"]
        return pattern_type in context_patterns

    @classmethod
    def create_default_mixed_pattern_data(cls) -> DataFrame:
        """
        Create default test data with mixed signals for general testing.

        Returns:
            DataFrame with alternating patterns suitable for general signal testing
        """
        base_data = {
            "open": [
                100.0,
                102.0,
                98.0,
                105.0,
                97.0,
                108.0,
                95.0,
                110.0,
                93.0,
                112.0,
                91.0,
                115.0,
                89.0,
                118.0,
                87.0,
                120.0,
                85.0,
                122.0,
                83.0,
                125.0,
            ],
            "high": [
                101.0,
                103.0,
                99.0,
                106.0,
                98.0,
                109.0,
                96.0,
                111.0,
                94.0,
                113.0,
                92.0,
                116.0,
                90.0,
                119.0,
                88.0,
                121.0,
                86.0,
                123.0,
                84.0,
                126.0,
            ],
            "low": [
                99.0,
                101.0,
                97.0,
                104.0,
                96.0,
                107.0,
                94.0,
                109.0,
                92.0,
                111.0,
                90.0,
                114.0,
                88.0,
                117.0,
                86.0,
                119.0,
                84.0,
                121.0,
                82.0,
                124.0,
            ],
            "close": [
                100.5,
                102.5,
                98.5,
                105.5,
                97.5,
                108.5,
                95.5,
                110.5,
                93.5,
                112.5,
                91.5,
                115.5,
                89.5,
                118.5,
                87.5,
                120.5,
                85.5,
                122.5,
                83.5,
                125.5,
            ],
        }

        data_length = len(base_data["open"])

        return DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 9, 30) + timedelta(minutes=i * 5) for i in range(data_length)],
                "open": base_data["open"],
                "high": base_data["high"],
                "low": base_data["low"],
                "close": base_data["close"],
                "volume": [1000] * data_length,
            }
        )
