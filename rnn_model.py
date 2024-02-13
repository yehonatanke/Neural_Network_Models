import tensorflow as tf
from tensorflow.keras import layers, models

# Generate synthetic sequence data
import numpy as np
data = np.random.randn(1000, 100, 20)  # 1000 sequences of length 100 with 20 features each
labels = np.random.randint(2, size=(1000,))  # Binary labels for sequence classification

# Split the data into training, validation, and test sets
train_data, val_data, test_data = data[:800], data[800:900], data[900:]
train_labels, val_labels, test_labels = labels[:800], labels[800:900], labels[900:]

# Define a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) cells model
def create_rnn_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.LSTM(128, input_shape=input_shape))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# Define input shape and number of classes
input_shape = (100, 20)  # Input shape for sequence data
num_classes = 2  # Number of classes for classification

# Create RNN model
rnn_model = create_rnn_model(input_shape, num_classes)

# Compile the model
rnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train the model
rnn_model.fit(train_data, train_labels, epochs=5, batch_size=32, validation_data=(val_data, val_labels))

# Evaluate the model on test data
test_loss, test_acc = rnn_model.evaluate(test_data, test_labels)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
