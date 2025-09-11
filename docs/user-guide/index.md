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

### Signals and Pivots
The indicators component identifies key market structure points and generates actionable signals based on TheStrat rules.

## Getting Help

If you need assistance:

1. Check the [Examples](examples.md) section for similar use cases
2. Review the [API Reference](../reference/index.md) for detailed method documentation
3. Contact the maintainer for private module support

Let's get started with [Installation](installation.md)!