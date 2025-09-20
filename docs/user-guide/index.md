# User Guide

Welcome to the TheStrat user guide. This section provides comprehensive documentation for using the TheStrat Python module in your trading applications.

## What is #TheStrat?

#TheStrat is a technical analysis methodology that focuses on understanding market structure through the identification of specific bar patterns and their relationships across multiple timeframes.

## Guide Structure

This user guide is organized into the following sections:

### [Installation](installation.md)
How to install and set up the TheStrat module in your environment.

### [Quick Start](quickstart.md)
Get up and running quickly with basic examples and common use cases.

### [Examples](examples.md)
Detailed examples showing how to use each component and feature.

### [Asset Classes](asset-classes.md)
Understanding how different asset classes work within the framework.

## Prerequisites

Before using TheStrat, you should have:

- **Python 3.11 or higher** installed
- Basic understanding of financial markets and OHLCV data
- Familiarity with Python data structures (pandas/polars DataFrames)

## Key Concepts

### Timeframe Aggregation
TheStrat works across multiple timeframes. The aggregation component handles converting your base timeframe data (e.g., 1-minute bars) into higher timeframes (e.g., 5-minute, 15-minute, hourly).

### Inside and Outside Bars
Core to #TheStrat methodology:
- **Inside Bar**: High ≤ previous high AND Low ≥ previous low
- **Outside Bar**: High > previous high AND Low < previous low

### Swing Points and Market Structure
TheStrat uses precise swing point detection to identify market structure:

- **Swing High**: A price peak that is the highest point within its lookback/lookahead window and meets the percentage threshold compared to the previous swing high
- **Swing Low**: A price trough that is the lowest point within its lookback/lookahead window and meets the percentage threshold compared to the previous swing low
- **Higher High (HH)**: Each new swing high that is higher than the previous swing high (bullish structure)
- **Lower High (LH)**: Each new swing high that is lower than the previous swing high (bearish structure)
- **Higher Low (HL)**: Each new swing low that is higher than the previous swing low (bullish structure)
- **Lower Low (LL)**: Each new swing low that is lower than the previous swing low (bearish structure)

### Configuration Parameters
Swing point detection can be configured per timeframe:

- **Window Size**: Number of bars to look back and ahead for peak/valley confirmation (default: 5)
- **Threshold**: Minimum percentage change required to confirm a new swing point (default: 2.0%)

### Signals and Patterns
The indicators component identifies key pattern sequences and generates actionable signals based on TheStrat rules.

## Getting Help

If you need assistance:

1. Check the [Examples](examples.md) section for similar use cases
2. Review the [API Reference](../reference/index.md) for detailed method documentation
3. Contact the maintainer for private module support

Let's get started with [Installation](installation.md)!
