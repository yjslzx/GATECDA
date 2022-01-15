import tensorflow as tf
from tensorflow.keras import layers


def get_dnn():
    model = tf.keras.Sequential(
        [
            layers.Dropout(rate=0.1),
            layers.BatchNormalization(),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(rate=0.1),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(rate=0.1),
            layers.Dense(32, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(rate=0.1),
            layers.Dense(1, activation="sigmoid")
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy','AUC'],
    )

    return model