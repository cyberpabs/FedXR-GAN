#!/bin/bash
# Sets the number of supernodes for local simulation.
# Run this once after cloning: bash federated/setup_simulation.sh
NUM_SUPERNODES=2
sed -i "s/options.num-supernodes = .*/options.num-supernodes = $NUM_SUPERNODES/" ~/.flwr/config.toml
echo "Set num-supernodes to $NUM_SUPERNODES in ~/.flwr/config.toml"