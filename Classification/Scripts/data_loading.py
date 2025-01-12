import tensorflow as tf

def load_dataset(data_dir, img_height, img_width, batch_size, validation_split=0.2):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset='training',
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset='validation',
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True
    )

    return train_ds, val_ds
