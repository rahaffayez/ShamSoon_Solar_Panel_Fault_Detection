#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def fine_tune_model(model, train_ds, val_ds, base_model_index=1, frozen_layers=14, learning_rate=0.0001, epochs=10):
    """
    Fine-tunes a pre-trained model by unfreezing some layers and recompiling it.

    Args:
        model: The pre-trained model to fine-tune.
        train_ds: Training dataset.
        val_ds: Validation dataset.
        base_model_index: Index of the base model in the model's layers.
        frozen_layers: Number of layers to freeze in the base model.
        learning_rate: Learning rate for fine-tuning.
        epochs: Number of epochs for fine-tuning.

    Returns:
        History object containing training/validation metrics.
    """
    # Unfreeze the base model
    base_model = model.layers[base_model_index]
    base_model.trainable = True

    # Freeze the first `frozen_layers` layers of the base model
    for layer in base_model.layers[:frozen_layers]:
        layer.trainable = False

    # Recompile the model with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Define callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-2,
        patience=3,
        verbose=1,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=2,
        verbose=1,
        min_lr=1e-6
    )

    # Use `.keras` file extension for saving the model
    checkpoint = ModelCheckpoint(
        filepath='best_model.keras',  
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # Fine-tune the model
    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )

    return fine_tune_history