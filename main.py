import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.parameters = {}
        self.cache = {}
        self.grads = {}
        self.layers_dims = [784, 256, 128, 64, 32, 10]
        self.activation_functions = {
            "relu": lambda x: np.maximum(0, x),
            "softmax": lambda x: np.exp(x - np.max(x, axis=1, keepdims=True))
            / np.sum(
                np.exp(x - np.max(x, axis=1, keepdims=True)), axis=1, keepdims=True
            ),
        }
        self.initialize_parameters()

    def initialize_parameters(self):
        np.random.seed(42)
        for l in range(1, len(self.layers_dims)):
            self.parameters["W" + str(l)] = (
                np.random.randn(self.layers_dims[l], self.layers_dims[l - 1]) * 0.01
            )
            self.parameters["b" + str(l)] = np.zeros((self.layers_dims[l], 1))

    def forward_propagation(self, X):
        A = X.T
        for l in range(1, len(self.layers_dims)):
            Z = np.dot(self.parameters["W" + str(l)], A) + self.parameters["b" + str(l)]
            if l == len(self.layers_dims) - 1:
                A = self.activation_functions["softmax"](Z)
            else:
                A = self.activation_functions["relu"](Z)
            self.cache["Z" + str(l)] = Z
            self.cache["A" + str(l)] = A
        return A

    def compute_cost(self, A, Y):
        m = Y.shape[0]
        cost = (
            -1 / m * np.sum(Y * np.log(A + 1e-10))
        )  # Add small epsilon to prevent log(0)
        return cost

    def backward_propagation(self, X, Y):
        m = Y.shape[0]
        for l in range(len(self.layers_dims) - 1, 0, -1):
            if l == len(self.layers_dims) - 1:
                dZ = self.cache["A" + str(l)] - Y.T
            else:
                dA = np.dot(
                    self.parameters["W" + str(l + 1)].T, self.grads["dZ" + str(l + 1)]
                )
                dZ = dA * (self.cache["Z" + str(l)] > 0)
            self.grads["dW" + str(l)] = (
                1 / m * np.dot(dZ, self.cache["A" + str(l - 1)].T)
            )
            self.grads["db" + str(l)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            self.grads["dZ" + str(l)] = dZ

    def update_parameters(self, learning_rate):
        for l in range(1, len(self.layers_dims)):
            self.parameters["W" + str(l)] -= learning_rate * self.grads["dW" + str(l)]
            self.parameters["b" + str(l)] -= learning_rate * self.grads["db" + str(l)]

    def train(self, X, Y, learning_rate=0.01, num_epochs=100, batch_size=64):
        m = X.shape[0]
        costs = []
        for epoch in range(num_epochs):
            shuffled_indices = np.random.permutation(m)
            X_shuffled = X[shuffled_indices]
            Y_shuffled = Y[shuffled_indices]
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                Y_batch = Y_shuffled[i : i + batch_size]
                A = self.forward_propagation(X_batch)
                cost = self.compute_cost(A, Y_batch)
                self.backward_propagation(X_batch, Y_batch)
                self.update_parameters(learning_rate)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.4f}")
            costs.append(cost)
        return costs


# Preparing the data
(x_train, y_train), (x_test, y_test) = np.load("mnist.npz", allow_pickle=True)["arr_0"]
x_train = x_train.reshape((x_train.shape[0], -1)).astype("float32") / 255.0
x_test = x_test.reshape((x_test.shape[0], -1)).astype("float32") / 255.0
y_train = np.eye(10)[y_train]

# Creating the neural network
model = NeuralNetwork()

# Training the neural network
num_epochs = 100
learning_rate = 0.01
costs = model.train(x_train, y_train, learning_rate, num_epochs)

# Plot the
