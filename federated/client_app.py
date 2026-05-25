# client_app.py
# Defines the Flower ClientApp for federated training.
# Each client trains locally on its partition and sends weights back to the server.
#
# References:
# https://flower.ai/docs/framework/tutorial-quickstart-tensorflow.html

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from flwr.clientapp import ClientApp
from flwr.app import Message, Context, ArrayRecord, RecordDict, MetricRecord
from federated.task import load_partition, load_model

app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    # Each client gets a unique partition_id assigned by Flower via node_config
    partition_id = context.node_config["partition-id"]
    num_partitions = context.run_config.get("num-partitions", 3)

    train_ds, val_ds = load_partition(partition_id, num_partitions)

    model = load_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-5),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.AUC(name="auc", from_logits=True)],
    )
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())

    local_epochs = context.run_config.get("local-epochs", 3)

    print(f"Client {partition_id}: training for {local_epochs} epoch(s)")
    history = model.fit(train_ds, validation_data=val_ds, epochs=local_epochs, verbose=0)

    train_loss = history.history["loss"][-1]
    train_auc = history.history["auc"][-1]
    num_examples = sum(len(b) for b, _ in train_ds)
    print(f"Client {partition_id}: loss={train_loss:.4f} auc={train_auc:.4f}")

    content = RecordDict({
        "arrays": ArrayRecord(model.get_weights()),
        "metrics": MetricRecord({"train_loss": train_loss, "train_auc": train_auc,
                                 "num-examples": float(num_examples)}),
    })
    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.run_config.get("num-partitions", 3)

    _, val_ds = load_partition(partition_id, num_partitions)

    model = load_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-5),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.AUC(name="auc", from_logits=True)],
    )
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())

    results = model.evaluate(val_ds, return_dict=True, verbose=0)
    num_examples = sum(len(b) for b, _ in val_ds)
    print(f"Client {partition_id}: val_loss={results['loss']:.4f} val_auc={results['auc']:.4f}")

    content = RecordDict({
        "metrics": MetricRecord({"eval_loss": results["loss"], "eval_auc": results["auc"],
                                 "num-examples": float(num_examples)}),
    })
    return Message(content=content, reply_to=msg)
