#!/bin/bash
# Sets the number of supernodes for local simulation.
# Run this once after cloning: bash federated/setup_simulation.sh

NUM_SUPERNODES=2
CONFIG="$HOME/.flwr/config.toml"

mkdir -p "$HOME/.flwr"

if [ ! -f "$CONFIG" ]; then
    echo "Creating $CONFIG..."
    cat > "$CONFIG" << EOF
[superlink]
default = "local-simulation"
[superlink.supergrid]
address = "supergrid.flower.ai"
[superlink.local]
address = ":local:"
[superlink.local-simulation]
address = ":local:"
options.num-supernodes = $NUM_SUPERNODES
EOF
else
    sed -i "s/options.num-supernodes = .*/options.num-supernodes = $NUM_SUPERNODES/" "$CONFIG"
fi

echo "Set num-supernodes to $NUM_SUPERNODES in $CONFIG"