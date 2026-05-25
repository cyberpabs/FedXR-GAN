# Changelog

## [Unreleased]

### Added
- `federated/task.py`: stratified data partitioning with class interleaving and stratified val split
- `federated/client_app.py`: Flower ClientApp with @train and @evaluate handlers, GPU support
- `federated/server_app.py`: Flower ServerApp with FedAvg, per-round CSV metrics logging
- `federated/setup_simulation.sh`: helper script to configure num-supernodes on fresh clone
- `results/federated_metrics.csv`: per-round federated training metrics output

### Changed
- `pyproject.toml`: added Flower simulation config, pinned tensorflow[and-cuda]==2.19.0

### Environment
- GPU detection under WSL2 requires `pip install "tensorflow[and-cuda]==2.19.0"`
- After cloning run `bash federated/setup_simulation.sh` before `flwr run . local-simulation --stream`
- Flower 1.30.0 migrates federation config from pyproject.toml to ~/.flwr/config.toml on first run