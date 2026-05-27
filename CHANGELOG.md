# Changelog

## [1.0.0] - 2026-05

### Added

#### GAN
- `gan/model.py`: DCGAN with configurable generator/discriminator filter banks for 64×64 and 128×128 output sizes
- `gan/main.py`: entry point for dataset download, preprocessing, and DCGAN training
- `gan/constants.py`: shared GAN hyperparameters (`NOISE_DIM=100`, `EPOCHS=7000`, checkpoint dir)
- `gan/equilibrate_normal.py`: generates synthetic NORMAL X-rays until class count matches PNEUMONIA
- `gan/calculate_fid_score.py`: FID evaluation via InceptionV3 feature extraction (score: 25.37)
- `gan/restore_model.py`: loads a saved generator checkpoint and generates a sample image

#### Data
- `data/constants.py`: dataset paths, image size (128×128), batch size, and buffer size constants
- `data/dataset_download.py`: Kaggle dataset downloader with progress bar and zip extraction
- `data/dataset_preprocessing.py`: image preprocessing pipeline (resize, normalize to [-1,1], jitter augmentation)

#### Classifier
- `models/classifier.py`: 4-block CNN (Conv2D → BatchNorm → LeakyReLU → MaxPool) with GlobalAvgPool + Dense head; includes training, evaluation, threshold sweep, and checkpoint callbacks

#### Federated Learning
- `federated/task.py`: stratified per-client data partitioning with class interleaving and stratified val split
- `federated/client_app.py`: Flower ClientApp with `@train` and `@evaluate` handlers
- `federated/server_app.py`: Flower ServerApp with FedAvg, per-round CSV metrics logging
- `federated/evaluate_saved_model.py`: standalone evaluation script for the final federated model
- `federated/plot_metrics.py`: plots federated training curves (loss + AUC) from CSV
- `federated/script.sh`: configures `num-supernodes` in `~/.flwr/config.toml` on fresh clone
- `pyproject.toml`: Flower app config with FedAvg hyperparameters (50 rounds, 4 partitions, 1 local epoch)

#### Results
- `results/federated_metrics.csv`: per-round federated training metrics (50 rounds)
- `fid_score_results.txt`: FID score output from InceptionV3 evaluation

### Environment
- Requires `pip install "tensorflow[and-cuda]==2.19.0"` for GPU support under WSL2
- Run `bash federated/setup_simulation.sh` after cloning before `flwr run . local-simulation --stream`
- Flower 1.30.0 migrates federation config from `pyproject.toml` to `~/.flwr/config.toml` on first run
