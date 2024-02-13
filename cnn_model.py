import tensorflow as tf
from tensorflow.keras import layers, models

# Define a Convolutional Neural Network (CNN) model
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# Example usage for image classification
input_shape = (28, 28, 1)  # Input shape for MNIST dataset
num_classes = 10  # Number of classes for classification

# Create CNN model
cnn_model = create_cnn_model(input_shape, num_classes)

# Print model summary
cnn_model.summary()
