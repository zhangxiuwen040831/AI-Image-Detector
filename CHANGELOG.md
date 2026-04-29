# Changelog

## v1.0.0-final-defense

### Added

- Added final release documentation for graduation-defense deployment.
- Added GitHub Release notes in `RELEASE_NOTES.md`.
- Added Git LFS tracking for `.pth` checkpoint files.
- Added final `photos_test` regression result CSV.

### Changed

- Stabilized the three-branch model release around `deploy_safe_tri_branch`.
- Kept semantic, frequency, and noise branches all active in forward inference.
- Kept full `tri_fusion / tri_classifier` code for future retraining.
- Clarified threshold-based decision logic in frontend and documentation.
- Updated README with setup, model architecture, deployment mode, and validation details.

### Fixed

- Fixed the old-checkpoint risk where randomly initialized `tri_classifier` could affect inference.
- Fixed the 41% AIGC probability vs 35% threshold explanation in the UI.
- Fixed Fusion Evidence Triangle center-point behavior by using real `evidence_weights`.
- Fixed BranchContribution final decision weights showing `N/A`.
- Fixed login page API endpoint construction so it follows the active API base URL.
- Hardened backend triplet normalization for dict/list/string numeric values while preserving `0.0`.

### Cleaned

- Removed Python cache directories.
- Removed temporary process outputs in `tmp/`.
- Removed obsolete mode-comparison JSON output.
- Removed old unreferenced frontend login/register components.
