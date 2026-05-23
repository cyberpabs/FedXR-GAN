# Changelog

## [Unreleased]

### Added — Classifier (`models/classifier.py`)

- Binary CNN classifier (NORMAL vs PNEUMONIA) with 4 convolutional blocks (32→64→128→256 filters), GlobalAveragePooling, and two Dropout(0.5) layers
- 80/20 train/val split from the train directory (`seed=42`) replacing the original 16-image val folder
- Training augmentation: random horizontal flip + random brightness on train split only
- Callbacks: `ModelCheckpoint` (best val_auc), `ReduceLROnPlateau` (factor=0.5, patience=3), `EarlyStopping` (patience=8)
- Threshold sweep at evaluation time (0.10→0.90, step 0.05) printing accuracy, precision, recall, F1 per threshold
- `--evaluate-only <path>` CLI flag to evaluate a saved model without retraining
- Timestamped model saves (`classifier_model_YYYYMMDD_HHMMSS.keras`) and training curve plots to avoid overwriting previous experiments
- `load_and_wrap_model` utility to recompile a saved model for evaluation

### Fixed

- Preprocessing mismatch: test set was being passed as raw `[0, 255]` pixels to a model trained on `[-1, 1]`; `load_test` now consistently applies `/127.5 - 1.0` normalization
- Noisy validation caused by the dataset's official 16-image `val/` folder; replaced with a proper 80/20 split
- Learning rate `1e-4` caused val loss spikes at early epochs; reduced to `3e-5`
- Dropout too low (0.4/0.3) causing train loss to collapse to ~0 while val remained high; increased to 0.5/0.5

### Environment

- Fixed GPU detection under WSL2: `pip install "tensorflow[and-cuda]==2.19.0"` required for TF's internal CUDA plugin loader to find cuDNN/cuBLAS
- Pinned TensorFlow to 2.19.0 in `requirements.txt`

### Results

- Test accuracy: 83.65% | AUC: 0.7667 | F1: 0.8768 (threshold=0.90) | Recall: 0.9308
