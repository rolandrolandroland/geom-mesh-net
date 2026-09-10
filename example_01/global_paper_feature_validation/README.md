# Global Paper Feature Validation

This directory contains global spatial-summary features calculated from original patterns 0-9.

## Configuration

- Guest labels: `2` and `3`
- Random relabelings per pattern: `5`
- Guest G and F maximum radius: `10.0`
- Guest K maximum radius: `70.0`
- Guest-to-host G maximum radius: `8.0`
- Maximum guest points used for each K estimate: `3000`

All original points are used for G, F, and guest-to-host G. The K point cap is a random guest-point sample used to keep large-radius translation correction tractable.

## Results

- Numerical-check failures: `[]`
- Radius-range warnings: `[]`

## Main Artifacts

- `global_paper_features_10_patterns.csv`: feature table
- `validation_diagnostics.csv`: numerical and range checks
- `summary_function_anomalies_10_patterns.png`: aggregate curves
- `pattern_*_summary_functions.png`: per-pattern curves
