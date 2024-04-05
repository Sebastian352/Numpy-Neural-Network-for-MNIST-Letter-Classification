import numpy as np


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


def CrossEntropyLoss(a, y):
    return np.mean(-np.log(a[y]))


# Example usage:
predictions = np.array(
    [[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.5, 0.3, 0.2]]
)  # Raw scores for each class
a = softmax(predictions)  # Predicted probabilities
targets = np.array([0, 1, 2])  # True labels for each example
loss = CrossEntropyLoss(a, targets)
print("Cross-entropy loss:", loss)
