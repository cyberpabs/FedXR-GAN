import tensorflow as tf
from federated.task import load_test_data

tf.config.set_visible_devices([], 'GPU')

MODEL_PATH = "federated/final_model.keras"

test_ds = load_test_data() 

print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-5),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc", from_logits=True),
        tf.keras.metrics.Recall(name="recall")
    ],
)

print("Evaluating model...")
results = model.evaluate(test_ds, return_dict=True, verbose=1)

print("\n--- Evaluation Results ---")
for metric_name, value in results.items():
    print(f"{metric_name.capitalize()}: {value:.4f}")
