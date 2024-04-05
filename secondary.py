import numpy as np
import tensorflow as tf
import keras
from keras import layers, Sequential, datasets, Input
import matplotlib.pyplot as plt


tf.keras.datasets.mnist.load_data(path="mnist.npz")


(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten the images (from 28x28 to 784)
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

model = Sequential(
    [
        layers.Dense(256, activation="relu", input_shape=(784,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

model.build()

model.summary()

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["accuracy"],
)

# Train the model
history = model.fit(
    x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test)
)


# Plot accuracy
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.title("Accuracy Plot")
plt.show()

# Show misclassified samples
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

misclassified_indices = np.where(predicted_labels != y_test)[0]
misclassified_images = x_test[misclassified_indices]
misclassified_labels = y_test[misclassified_indices]
predicted_misclassified_labels = predicted_labels[misclassified_indices]

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(misclassified_images[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(
        f"True: {misclassified_labels[i]} Predicted: {predicted_misclassified_labels[i]}"
    )
plt.suptitle("Misclassified Samples")
plt.show()
