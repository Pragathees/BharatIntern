import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build the neural network model
model = Sequential()
model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1)

# Evaluate the model
accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
print('Accuracy:', accuracy)

# Test the model with new images
def test_model(image):
    image = image.reshape(1, 28, 28, 1)
    image = image.astype('float32') / 255
    prediction = model.predict(image)
    digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return digit, confidence

# Example usage:
# Load your own handwritten image
image = plt.imread('your_image.png')
# Convert to grayscale
image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
# Normalize the image
image = image.astype('float32') / 255
# Use the test_model function to get the prediction and confidence
digit, confidence = test_model(image)
print('Predicted digit:', digit)
print('Confidence:', confidence)
