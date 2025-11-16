# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- insertion marker -->
## [Unreleased]

## [0.0.1a32] - 2025-11-16

### Fixed
- **Monthly aggregation DST handling (Issue #51)**: Fixed monthly aggregation timestamps being 1 hour too early by implementing DST-aware shifted-index pattern. Monthly bars now correctly maintain session start time (e.g., 09:30 ET for equities) across EST/EDT transitions, matching the behavior of quarterly and yearly periods.
