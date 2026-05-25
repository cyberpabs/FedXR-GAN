"""Plot federated training curves from the metrics CSV.

Usage:
    python3 federated/plot_metrics.py
"""

import argparse
import csv
import math
import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_METRICS_PATH = "results/federated_metrics.csv"
DEFAULT_OUTPUT_DIR = "results/classifier_results"

LOSS_COLUMNS = ("round", "train_loss", "eval_loss")
METRIC_OPTIONS = (
    ("AUC", "train_auc", "eval_auc"),
    ("Accuracy", "train_acc", "eval_acc"),
)


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_metrics(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [column for column in LOSS_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(missing)}"
            )

        metric_title = None
        train_metric_column = None
        eval_metric_column = None
        for title, train_column, eval_column in METRIC_OPTIONS:
            if train_column in fieldnames and eval_column in fieldnames:
                metric_title = title
                train_metric_column = train_column
                eval_metric_column = eval_column
                break

        if train_metric_column is None:
            expected = " or ".join(
                f"{train_column}/{eval_column}"
                for _, train_column, eval_column in METRIC_OPTIONS
            )
            raise ValueError(f"{path} must contain metric columns: {expected}")

        metrics = {
            "round": [],
            "train_loss": [],
            "eval_loss": [],
            "train_metric": [],
            "eval_metric": [],
            "metric_title": metric_title,
        }
        for row in reader:
            metrics["round"].append(int(_to_float(row["round"])))
            metrics["train_loss"].append(_to_float(row["train_loss"]))
            metrics["eval_loss"].append(_to_float(row["eval_loss"]))
            metrics["train_metric"].append(_to_float(row[train_metric_column]))
            metrics["eval_metric"].append(_to_float(row[eval_metric_column]))

    if not metrics["round"]:
        raise ValueError(f"{path} does not contain any metric rows")

    return metrics


def plot_federated_history(metrics, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    rounds = metrics["round"]

    axes[0].plot(rounds, metrics["train_loss"], label="train")
    axes[0].plot(rounds, metrics["eval_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Round")
    axes[0].legend()

    axes[1].plot(rounds, metrics["train_metric"], label="train")
    axes[1].plot(rounds, metrics["eval_metric"], label="val")
    axes[1].set_title(metrics["metric_title"])
    axes[1].set_xlabel("Round")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build classifier-style training curves from federated metrics."
    )
    parser.add_argument(
        "--metrics",
        default=DEFAULT_METRICS_PATH,
        help=f"Input CSV path (default: {DEFAULT_METRICS_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the PNG output (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional PNG filename. Defaults to federated_training_curves_<timestamp>.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = load_metrics(args.metrics)

    os.makedirs(args.output_dir, exist_ok=True)
    output_name = args.output_name
    if output_name is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_name = f"federated_training_curves_{ts}.png"

    output_path = os.path.join(args.output_dir, output_name)
    plot_federated_history(metrics, output_path)
    print(f"Federated training curves saved to {output_path}")


if __name__ == "__main__":
    main()
