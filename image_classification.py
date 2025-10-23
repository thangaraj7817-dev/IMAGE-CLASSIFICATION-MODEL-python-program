# Task 3: Image Classification using CNN

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import numpy as np

# Step 1: Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Step 2: Preprocess data
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Step 3: Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes for MNIST
])

# Step 4: Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Step 5: Train model
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Step 6: Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

# Step 7: Save trained model
model.save("cnn_image_model.h5")
print("✅ Model saved as cnn_image_model.h5")

# Step 8: Save test accuracy to a text file
folder_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(folder_path, "cnn_model_results.txt")
with open(file_path, "w") as file:
    file.write(f"Test Accuracy: {test_acc}\n")
print(f"✅ Accuracy saved to {file_path}")

# Step 9 (Optional): Visualize first 10 predictions
predictions = model.predict(X_test[:10])
for i, pred in enumerate(predictions):
    plt.imshow(X_test[i].reshape(28,28), cmap='gray')
    plt.title(f"Predicted: {np.argmax(pred)}")
    plt.show()
