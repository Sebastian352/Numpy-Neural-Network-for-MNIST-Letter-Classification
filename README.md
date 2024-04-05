# MNIST Digit Classification

This repository contains two implementations for classifying the MNIST dataset: one using TensorFlow and Keras (`secondary.py`) and another using numpy (`main.py`).

## TensorFlow Implementation

### File: `secondary.py`

This script builds and trains a neural network using TensorFlow and Keras to classify handwritten digits from the MNIST dataset.

#### Requirements
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib

#### Usage
1. Ensure all dependencies are installed.
2. Run the script `secondary.py`.
3. The script will train the neural network and display accuracy plots and misclassified samples.

## NumPy Implementation

### File: `main.py`

This script implements a neural network from scratch using NumPy to classify handwritten digits from the MNIST dataset.

#### Requirements
- Python 3.x
- NumPy
- Matplotlib

#### Usage
1. Ensure all dependencies are installed.
2. Run the script `main.py`.
3. The script will train the neural network and display cost over epochs.

## MNIST Dataset

The MNIST dataset consists of 60,000 training images and 10,000 testing images of handwritten digits. Each image is 28x28 pixels, and the task is to classify each image into one of 10 classes (0 through 9).

### Data Preprocessing
- Images are normalized to have pixel values between 0 and 1.
- Training and testing images are reshaped to flatten them into vectors of length 784.

### Model Architecture
- **TensorFlow Implementation**: A fully connected neural network with four hidden layers (256, 128, 64, and 32 neurons) and a 10-neuron output layer using softmax activation.
- **NumPy Implementation**: A fully connected neural network with the same architecture as the TensorFlow implementation.

### Performance Comparison
- Both implementations are trained on the same dataset and aim to achieve high accuracy in classifying digits.

Feel free to explore and compare the implementations!

