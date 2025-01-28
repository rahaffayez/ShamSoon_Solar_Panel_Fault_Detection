
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def build_smaller_model(img_height, img_width, num_classes):
    # Load the MobileNetV2 base model
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(img_height, img_width, 3)
    )
    base_model.trainable = False  # Freeze the base model

    # Build the custom model on top of the base model
    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes)(x)

    # Create the final model
    model = models.Model(inputs, outputs)
    return model


