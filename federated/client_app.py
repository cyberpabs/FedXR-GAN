# client_app.py
# Defines the Flower ClientApp for federated training.
# Each client trains locally on its partition and sends weights back to the server.
#
# References:
# https://flower.ai/docs/framework/tutorial-quickstart-tensorflow.html

import os, sys
# Fix: Python doesn't see the root, hence this line
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
from flwr.clientapp import ClientApp
from flwr.app import Message, Context, ArrayRecord
from task import load_partition, load_model

app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    # Read which partition this client owns
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Load only this client's slice of the data
    train_ds, val_ds = load_partition(partition_id, num_partitions)

    # Build model and set weights received from the server
    model = load_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-5),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())

    local_epochs = context.run_config.get("local-epochs", 3)

    print(f"Client {partition_id}: training for {local_epochs} epoch(s)")
    history = model.fit(train_ds, validation_data=val_ds, epochs=local_epochs, verbose=0)

    train_loss = history.history["loss"][-1]
    train_acc = history.history["accuracy"][-1]
    print(f"Client {partition_id}: loss={train_loss:.4f} acc={train_acc:.4f}")

    # Pack updated weights and metrics back into a Message for the server
    content = {
        "arrays": ArrayRecord(model.get_weights()),
        "metrics": {"train_loss": train_loss, "train_acc": train_acc,
                    "num-examples": sum(1 for _ in train_ds.unbatch())},
    }
    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    # Read which partition this client owns
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    _, val_ds = load_partition(partition_id, num_partitions)

    # Build model and set weights received from the server
    model = load_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(3e-5),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                 tf.keras.metrics.AUC(name="auc")],
    )
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())

    results = model.evaluate(val_ds, return_dict=True, verbose=0)
    print(f"Client {partition_id}: val_loss={results['loss']:.4f} val_acc={results['accuracy']:.4f}")

    # Send evaluation metrics back (no model weights needed here)
    content = {
        "metrics": {"eval_loss": results["loss"], "eval_acc": results["accuracy"],
                    "num-examples": sum(1 for _ in val_ds.unbatch())},
    }
    return Message(content=content, reply_to=msg)
