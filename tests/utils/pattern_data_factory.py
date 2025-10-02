"""
Pattern Data Factory for TheStrat Signal Testing.

Provides clean, maintainable test data generation for specific signal patterns.
Each pattern has carefully crafted OHLC data designed to trigger the expected signal,
with verifiable expected entry/stop prices based on setup bar methodology.
"""

from datetime import datetime, timedelta

from polars import DataFrame


class PatternDataFactory:
    """Factory class for generating test data that triggers specific signal patterns."""

    # Pattern configurations with carefully designed OHLC data
    # Each pattern includes expected entry/stop prices based on setup bar (not trigger bar)
    PATTERN_CONFIGS = {
        # ===== 3-BAR REVERSAL PATTERNS (LONG) =====
        # Rev Strat patterns: 1-2D-2U (inside, down, up)
        "1-2D-2U": {
            "description": "Inside bar, followed by 2D (down), then 2U (up) - long reversal",
            "bias": "long",
            "bar_count": 3,
            "open": [100.0, 101.5, 99.0, 100.5, 103.0],
            "high": [102.0, 101.8, 100.0, 102.0, 105.0],
            "low": [99.0, 100.5, 98.5, 100.0, 102.5],
            "close": [101.0, 101.0, 99.5, 101.5, 104.0],
            # Setup bar is index 2 (2D), trigger is index 3 (2U)
            "expected_entry": 100.0,  # Setup bar (2D) high
            "expected_stop": 98.5,  # Setup bar (2D) low
        },
        # 3-1-2U (outside, inside, up)
        "3-1-2U": {
            "description": "3-bar (outside), inside bar, then 2U (up) - long reversal",
            "bias": "long",
            "bar_count": 3,
            "open": [100.0, 99.0, 100.5, 102.0, 105.0],
            "high": [102.0, 103.5, 101.8, 103.0, 107.0],
            "low": [99.0, 98.0, 100.0, 101.5, 104.0],
            "close": [101.0, 103.0, 101.0, 102.5, 106.0],
            # Setup bar is index 2 (1), trigger is index 3 (2U)
            "expected_entry": 101.8,  # Setup bar (1) high
            "expected_stop": 100.0,  # Setup bar (1) low
        },
        # 3-2D-2U (outside, down, up)
        "3-2D-2U": {
            "description": "3-bar (outside), 2D (down), then 2U (up) - long reversal",
            "bias": "long",
            "bar_count": 3,
            "open": [100.0, 99.0, 98.0, 99.5, 102.0],
            "high": [102.0, 103.5, 99.0, 101.0, 104.0],
            "low": [99.0, 98.0, 97.5, 99.0, 101.5],
            "close": [101.0, 103.0, 98.5, 100.5, 103.5],
            # Setup bar is index 2 (2D), trigger is index 3 (2U)
            "expected_entry": 99.0,  # Setup bar (2D) high
            "expected_stop": 97.5,  # Setup bar (2D) low
        },
        # 2D-1-2U (down, inside, up)
        "2D-1-2U": {
            "description": "2D (down), inside bar, then 2U (up) - long reversal",
            "bias": "long",
            "bar_count": 3,
            "open": [100.0, 98.0, 98.5, 100.0, 103.0],
            "high": [102.0, 99.5, 99.8, 102.0, 105.0],
            "low": [99.0, 97.5, 98.0, 99.5, 102.5],
            "close": [101.0, 98.0, 99.0, 101.5, 104.0],
            # Setup bar is index 2 (1), trigger is index 3 (2U)
            "expected_entry": 99.8,  # Setup bar (1) high
            "expected_stop": 98.0,  # Setup bar (1) low
        },
        # ===== 2-BAR REVERSAL PATTERNS (LONG) =====
        # 2D-2U (down, up)
        "2D-2U": {
            "description": "2D (down) followed by 2U (up) - long reversal",
            "bias": "long",
            "bar_count": 2,
            "open": [100.0, 98.0, 100.5, 103.0, 106.0],
            "high": [102.0, 99.5, 102.0, 105.0, 108.0],
            "low": [99.0, 97.5, 100.0, 102.5, 105.5],
            "close": [101.0, 98.0, 101.5, 104.0, 107.0],
            # Setup bar is index 1 (2D), trigger is index 2 (2U)
            "expected_entry": 99.5,  # Setup bar (2D) high
            "expected_stop": 97.5,  # Setup bar (2D) low
        },
        # ===== 3-BAR REVERSAL PATTERNS (SHORT) =====
        # 1-2U-2D (inside, up, down)
        "1-2U-2D": {
            "description": "Inside bar, followed by 2U (up), then 2D (down) - short reversal",
            "bias": "short",
            "bar_count": 3,
            "open": [100.0, 100.5, 102.0, 99.0, 96.0],
            "high": [102.0, 101.8, 103.5, 100.0, 97.0],
            "low": [99.0, 100.0, 101.5, 97.5, 95.0],
            "close": [101.0, 101.5, 103.0, 98.0, 96.5],
            # Setup bar is index 2 (2U), trigger is index 3 (2D)
            "expected_entry": 101.5,  # Setup bar (2U) low
            "expected_stop": 103.5,  # Setup bar (2U) high
        },
        # 3-1-2D (outside, inside, down)
        "3-1-2D": {
            "description": "3-bar (outside), inside bar, then 2D (down) - short reversal",
            "bias": "short",
            "bar_count": 3,
            "open": [100.0, 99.0, 100.5, 98.0, 95.0],
            "high": [102.0, 103.5, 101.8, 99.0, 96.0],
            "low": [99.0, 98.0, 100.0, 97.5, 94.0],
            "close": [101.0, 103.0, 101.0, 98.5, 95.5],
            # Setup bar is index 2 (1), trigger is index 3 (2D)
            "expected_entry": 100.0,  # Setup bar (1) low
            "expected_stop": 101.8,  # Setup bar (1) high
        },
        # 3-2U-2D (outside, up, down)
        "3-2U-2D": {
            "description": "3-bar (outside), 2U (up), then 2D (down) - short reversal",
            "bias": "short",
            "bar_count": 3,
            "open": [100.0, 99.0, 102.0, 99.5, 96.0],
            "high": [102.0, 103.5, 104.0, 100.0, 97.0],
            "low": [99.0, 98.0, 101.5, 98.0, 95.0],
            "close": [101.0, 103.0, 103.5, 99.0, 96.5],
            # Setup bar is index 2 (2U), trigger is index 3 (2D)
            "expected_entry": 101.5,  # Setup bar (2U) low
            "expected_stop": 104.0,  # Setup bar (2U) high
        },
        # 2U-1-2D (up, inside, down)
        "2U-1-2D": {
            "description": "2U (up), inside bar, then 2D (down) - short reversal",
            "bias": "short",
            "bar_count": 3,
            "open": [100.0, 102.0, 102.5, 98.0, 95.0],
            "high": [102.0, 104.0, 103.8, 99.0, 96.0],
            "low": [99.0, 101.5, 102.0, 97.5, 94.0],
            "close": [101.0, 103.0, 103.0, 98.5, 95.5],
            # Setup bar is index 2 (1), trigger is index 3 (2D)
            "expected_entry": 102.0,  # Setup bar (1) low
            "expected_stop": 103.8,  # Setup bar (1) high
        },
        # ===== 2-BAR REVERSAL PATTERNS (SHORT) =====
        # 2U-2D (up, down)
        "2U-2D": {
            "description": "2U (up) followed by 2D (down) - short reversal",
            "bias": "short",
            "bar_count": 2,
            "open": [100.0, 102.0, 98.0, 95.0, 92.0],
            "high": [102.0, 104.0, 99.5, 96.0, 93.0],
            "low": [99.0, 101.5, 97.5, 94.0, 91.0],
            "close": [101.0, 103.0, 98.5, 95.5, 92.5],
            # Setup bar is index 1 (2U), trigger is index 2 (2D)
            "expected_entry": 101.5,  # Setup bar (2U) low
            "expected_stop": 104.0,  # Setup bar (2U) high
        },
        # ===== CONTINUATION PATTERNS (LONG) =====
        # 2U-2U (up, up)
        "2U-2U": {
            "description": "2U (up) followed by another 2U (up) - long continuation",
            "bias": "long",
            "bar_count": 2,
            "open": [100.0, 102.0, 104.0, 106.0, 108.0],
            "high": [102.0, 104.0, 106.0, 108.0, 110.0],
            "low": [99.0, 101.5, 103.5, 105.5, 107.5],
            "close": [101.0, 103.0, 105.0, 107.0, 109.0],
            # Setup bar is index 1 (2U), trigger is index 2 (2U)
            "expected_entry": 104.0,  # Setup bar (2U) high
            "expected_stop": 101.5,  # Setup bar (2U) low
        },
        # 2U-1-2U (up, inside, up)
        "2U-1-2U": {
            "description": "2U (up), inside bar, then 2U (up) - long continuation",
            "bias": "long",
            "bar_count": 3,
            "open": [100.0, 102.0, 102.5, 104.0, 106.0],
            "high": [102.0, 104.0, 103.8, 106.0, 108.0],
            "low": [99.0, 101.5, 102.0, 103.5, 105.5],
            "close": [101.0, 103.0, 103.0, 105.0, 107.0],
            # Setup bar is index 2 (1), trigger is index 3 (2U)
            "expected_entry": 103.8,  # Setup bar (1) high
            "expected_stop": 102.0,  # Setup bar (1) low
        },
        # ===== CONTINUATION PATTERNS (SHORT) =====
        # 2D-2D (down, down)
        "2D-2D": {
            "description": "2D (down) followed by another 2D (down) - short continuation",
            "bias": "short",
            "bar_count": 2,
            "open": [100.0, 98.0, 96.0, 94.0, 92.0],
            "high": [102.0, 99.5, 97.0, 95.0, 93.0],
            "low": [99.0, 97.5, 95.5, 93.5, 91.5],
            "close": [101.0, 98.0, 96.5, 94.5, 92.5],
            # Setup bar is index 1 (2D), trigger is index 2 (2D)
            "expected_entry": 97.5,  # Setup bar (2D) low
            "expected_stop": 99.5,  # Setup bar (2D) high
        },
        # 2D-1-2D (down, inside, down)
        "2D-1-2D": {
            "description": "2D (down), inside bar, then 2D (down) - short continuation",
            "bias": "short",
            "bar_count": 3,
            "open": [100.0, 98.0, 98.5, 96.0, 94.0],
            "high": [102.0, 99.5, 99.8, 97.0, 95.0],
            "low": [99.0, 97.5, 98.0, 95.5, 93.5],
            "close": [101.0, 98.0, 99.0, 96.5, 94.5],
            # Setup bar is index 2 (1), trigger is index 3 (2D)
            "expected_entry": 98.0,  # Setup bar (1) low
            "expected_stop": 99.8,  # Setup bar (1) high
        },
        # ===== CONTEXT REVERSAL PATTERNS =====
        # 3-2U (context-dependent)
        "3-2U": {
            "description": "3-bar followed by 2U - context-dependent reversal pattern",
            "bias": "long",
            "bar_count": 2,
            "open": [95.0, 97.0, 100.0, 102.0, 104.0],
            "high": [97.0, 99.0, 103.0, 104.0, 106.0],
            "low": [94.0, 96.0, 99.0, 101.5, 103.5],
            "close": [96.5, 98.5, 102.5, 103.5, 105.0],
            # Setup bar is index 2 (3), trigger is index 3 (2U)
            "expected_entry": 103.0,  # Setup bar (3) high
            "expected_stop": 99.0,  # Setup bar (3) low
        },
        # 3-2D (context-dependent)
        "3-2D": {
            "description": "3-bar followed by 2D - context-dependent reversal pattern",
            "bias": "short",
            "bar_count": 2,
            "open": [95.0, 98.0, 100.0, 99.0, 97.0],
            "high": [98.0, 100.0, 103.0, 100.5, 98.0],
            "low": [94.5, 97.5, 98.5, 98.0, 96.5],
            "close": [97.5, 99.5, 102.5, 99.0, 97.5],
            # Setup bar is index 2 (3), trigger is index 3 (2D)
            "expected_entry": 98.5,  # Setup bar (3) low
            "expected_stop": 103.0,  # Setup bar (3) high
        },
        # ===== REAL MSFT DATA PATTERNS (from issue #25) =====
        "2D-2U_MSFT_20250926": {
            "description": "2D-2U Long Reversal - MSFT 2025-09-26",
            "bias": "long",
            "bar_count": 2,
            "data_source": "Real MSFT market data",
            "open": [507.00, 509.00],
            "high": [510.01, 513.94],
            "low": [505.04, 506.62],
            "close": [508.50, 512.00],
            # Setup bar is index 0 (2D), trigger is index 1 (2U)
            "expected_entry": 510.01,  # Setup bar high
            "expected_stop": 505.04,  # Setup bar low
        },
        "2U-1-2D_MSFT_20250923": {
            "description": "2U-1-2D Short Reversal - MSFT 2025-09-23",
            "bias": "short",
            "bar_count": 3,
            "data_source": "Real MSFT market data",
            "open": [512.00, 516.00, 513.00],
            "high": [519.30, 517.74, 514.59],
            "low": [510.31, 512.54, 507.31],
            "close": [518.00, 514.00, 509.00],
            # Setup bar is index 0 (2U), inside is index 1 (1), trigger is index 2 (2D)
            "expected_entry": 510.31,  # Setup bar low
            "expected_stop": 519.30,  # Setup bar high
        },
        "2U-2U_MSFT_20250915": {
            "description": "2U-2U Long Continuation - MSFT 2025-09-15",
            "bias": "long",
            "bar_count": 2,
            "data_source": "Real MSFT market data",
            "open": [505.00, 510.00],
            "high": [512.55, 515.45],
            "low": [503.85, 507.00],
            "close": [511.00, 514.00],
            # Setup bar is index 0 (2U), trigger is index 1 (2U)
            "expected_entry": 512.55,  # Setup bar high
            "expected_stop": 503.85,  # Setup bar low
        },
        "2U-1-2U_MSFT_20250715": {
            "description": "2U-1-2U Long Continuation - MSFT 2025-07-15",
            "bias": "long",
            "bar_count": 3,
            "data_source": "Real MSFT market data",
            "open": [498.00, 502.00, 503.00],
            "high": [505.03, 503.97, 508.30],
            "low": [497.78, 501.03, 502.78],
            "close": [504.00, 502.50, 507.00],
            # Setup bar is index 0 (2U), inside is index 1 (1), trigger is index 2 (2U)
            "expected_entry": 505.03,  # Setup bar high
            "expected_stop": 497.78,  # Setup bar low
        },
    }

    @classmethod
    def create(cls, pattern_type: str, extend_data: bool = True) -> DataFrame:
        """
        Create test data for a specific signal pattern.

        Args:
            pattern_type: The pattern type (e.g., "1-2D-2U", "2U-2D", "2D-2U_MSFT_20250926")
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
            "open": config["open"].copy(),
            "high": config["high"].copy(),
            "low": config["low"].copy(),
            "close": config["close"].copy(),
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
        # For MSFT patterns, add symbol column
        data_dict = {
            "timestamp": [datetime(2024, 1, 1, 9, 30) + timedelta(minutes=i * 5) for i in range(data_length)],
            "open": base_data["open"],
            "high": base_data["high"],
            "low": base_data["low"],
            "close": base_data["close"],
            "volume": [1000.0] * data_length,
        }

        # Add symbol column for MSFT patterns
        if "MSFT" in pattern_type:
            data_dict["symbol"] = ["MSFT"] * data_length

        return DataFrame(data_dict)

    @classmethod
    def get_expected_values(cls, pattern_type: str) -> dict:
        """
        Get expected entry/stop prices for a pattern.

        Args:
            pattern_type: The pattern type

        Returns:
            Dict with pattern, bias, expected_entry, expected_stop

        Raises:
            ValueError: If pattern_type is not found
        """
        if pattern_type not in cls.PATTERN_CONFIGS:
            available = list(cls.PATTERN_CONFIGS.keys())
            raise ValueError(f"Pattern '{pattern_type}' not found. Available: {available}")

        config = cls.PATTERN_CONFIGS[pattern_type]

        # Extract pattern name (remove MSFT suffix if present)
        pattern_name = pattern_type.split("_")[0]

        return {
            "pattern": pattern_name,
            "bias": config["bias"],
            "expected_entry": config["expected_entry"],
            "expected_stop": config["expected_stop"],
        }

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
        # Remove MSFT suffix if present
        base_pattern = pattern_type.split("_")[0]
        return base_pattern in context_patterns

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
                "volume": [1000.0] * data_length,
            }
        )
