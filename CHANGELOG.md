# Changelog

## 0.1.0

- Refactored the original MNIST MLA notebook into an importable Python package.
- Centralized model and training configuration.
- Fixed the transformer-to-MLA call signature.
- Changed the vision encoder to bidirectional attention by default.
- Corrected the weight-absorbed MLA einsum contractions.
- Added parity for attention dropout between MLA execution paths.
- Added reproducible seeding and checkpoint metadata.
- Added evaluation and prediction-grid utilities.
- Added unit tests and GitHub Actions CI.
