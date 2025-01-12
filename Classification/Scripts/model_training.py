import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

def train_model(model, train_ds, val_ds, epochs, fine_tune=False):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001 if not fine_tune else 0.0001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-2,
        patience=3,
        verbose=1,
        restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping]
    )

    return history
