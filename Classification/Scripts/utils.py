#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()

def evaluate_model(model, val_ds):
    loss, accuracy = model.evaluate(val_ds)
    print(f"Validation Loss: {loss}")
    print(f"Validation Accuracy: {accuracy}")
    return loss, accuracy

def visualize_predictions(model, val_ds, class_names):
    plt.figure(figsize=(20, 20))
    for images, labels in val_ds.take(1):
        for i in range(16):
            ax = plt.subplot(4, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            predictions = model.predict(tf.expand_dims(images[i], 0))
            score = tf.nn.softmax(predictions[0])
            if class_names[labels[i]] == class_names[np.argmax(score)]:
                plt.title("Actual: " + class_names[labels[i]])
                plt.ylabel("Predicted: " + class_names[np.argmax(score)], fontdict={'color': 'green'})
            else:
                plt.title("Actual: " + class_names[labels[i]])
                plt.ylabel("Predicted: " + class_names[np.argmax(score)], fontdict={'color': 'red'})
            plt.gca().axes.yaxis.set_ticklabels([])
            plt.gca().axes.xaxis.set_ticklabels([])
    plt.show()

def predict_image_class(model, img_path, class_labels, target_size=(224, 224)):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=target_size)
    img_array = np.expand_dims(image.img_to_array(img), axis=0)

    # Make prediction using the model
    predicted_class_index = np.argmax(tf.nn.softmax(model.predict(img_array)), axis=-1)[0]

    # Display the image and prediction
    plt.imshow(img)
    plt.title(f"Prediction: {class_labels[predicted_class_index]}")
    plt.axis('off')
    plt.show()

    print(f"The image belongs to the class: {class_labels[predicted_class_index]}")
    return predicted_class_index