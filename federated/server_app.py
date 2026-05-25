# server_app.py
# Defines the Flower ServerApp for federated training.
# Initializes the global model, runs FedAvg rounds, and saves the final model.
# Also logs per-round metrics to results/federated_metrics.csv for comparison
# against centralized training.
#
# References:
# https://flower.ai/docs/framework/tutorial-quickstart-tensorflow.html
# https://flower.ai/docs/framework/ref-api/flwr.serverapp.strategy.FedAvg.html
# https://flower.ai/docs/framework/ref-api/flwr.serverapp.ServerApp.html

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import csv
from flwr.serverapp import ServerApp, Grid
from flwr.serverapp.strategy import FedAvg
from flwr.app import Context, ArrayRecord
from federated.task import load_model

FINAL_MODEL_PATH = "federated/final_model.keras"
METRICS_PATH = "results/federated_metrics.csv"

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds = context.run_config.get("num-server-rounds", 5)
    fraction_train = context.run_config.get("fraction-train", 0.5)

    model = load_model()
    arrays = ArrayRecord(model.get_weights())

    print(f"Starting FedAvg: {num_rounds} rounds, fraction_train={fraction_train}")

    strategy = FedAvg(fraction_train=fraction_train)

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "train_loss", "train_auc", "eval_loss", "eval_auc"])

    for round_num in range(1, num_rounds + 1):
        print(f"Round {round_num}/{num_rounds}")

        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            num_rounds=1,
        )

        # Use aggregated weights as input for next round
        arrays = result.arrays

        # Extract metrics using the correct attribute names
        train_metrics = result.train_metrics_clientapp.get(1, {})
        eval_metrics = result.evaluate_metrics_clientapp.get(1, {})

        train_loss = train_metrics.get("train_loss", float("nan"))
        train_auc = train_metrics.get("train_auc", float("nan"))
        eval_loss = eval_metrics.get("eval_loss", float("nan"))
        eval_auc = eval_metrics.get("eval_auc", float("nan"))

        print(f"  train_loss={train_loss:.4f} train_auc={train_auc:.4f} eval_loss={eval_loss:.4f} eval_auc={eval_auc:.4f}")

        with open(METRICS_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([round_num, train_loss, train_auc, eval_loss, eval_auc])

    # Save final aggregated model
    model.set_weights(arrays.to_numpy_ndarrays())
    model.save(FINAL_MODEL_PATH)
    print(f"Final model saved to {FINAL_MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")
