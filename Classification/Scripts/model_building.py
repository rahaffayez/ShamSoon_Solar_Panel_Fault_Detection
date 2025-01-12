import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

def build_model(img_height, img_width, num_classes):
    base_model = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(img_height, img_width, 3)
    )
    base_model.trainable = False  # Freeze the base model

    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = tf.keras.applications.vgg16.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes)(x)

    model = models.Model(inputs, outputs)
    return model