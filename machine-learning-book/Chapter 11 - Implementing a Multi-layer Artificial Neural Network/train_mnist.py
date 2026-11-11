"""Train a from-scratch NumPy neural network to recognize handwritten MNIST digits.

Usage:
    python train_mnist.py                      # train, evaluate, save model
    python train_mnist.py --epochs 50          # train longer
    python train_mnist.py --image mnist/7/00015.png   # also classify one PNG
    python train_mnist.py --load mnist_mlp.npz --image some_digit.png  # skip training
"""
import argparse
import gzip
import os
import struct
import urllib.request

import numpy as np

BASE_URL = "https://raw.githubusercontent.com/fgnt/mnist/master"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


##########################
### MODEL
##########################

def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # for numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def int_to_onehot(y, num_labels):
    onehot = np.zeros((y.shape[0], num_labels))
    onehot[np.arange(y.shape[0]), y] = 1
    return onehot


class NeuralNetMLP:
    """784 inputs -> sigmoid hidden layer -> softmax output (10 digits)."""

    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        self.num_classes = num_classes
        rng = np.random.RandomState(random_seed)
        self.weight_h = rng.normal(0.0, 0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)
        self.weight_out = rng.normal(0.0, 0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)

    def forward(self, x):
        a_h = sigmoid(x @ self.weight_h.T + self.bias_h)          # [n, hidden]
        a_out = softmax(a_h @ self.weight_out.T + self.bias_out)  # [n, classes]
        return a_h, a_out

    def backward(self, x, a_h, a_out, y):
        # Cross-entropy loss with softmax: dLoss/dz_out = (a_out - y_onehot) / n
        delta_out = (a_out - int_to_onehot(y, self.num_classes)) / y.shape[0]
        d_w_out = delta_out.T @ a_h
        d_b_out = delta_out.sum(axis=0)

        # Backpropagate into the hidden layer (sigmoid derivative = a * (1 - a))
        delta_h = (delta_out @ self.weight_out) * a_h * (1. - a_h)
        d_w_h = delta_h.T @ x
        d_b_h = delta_h.sum(axis=0)
        return d_w_out, d_b_out, d_w_h, d_b_h


##########################
### DATA
##########################

def download(filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(f"{BASE_URL}/{filename}", filename)


def read_images(filename):
    with gzip.open(filename, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        pixels = np.frombuffer(f.read(), dtype=np.uint8)
    return pixels.reshape(num, rows * cols)


def read_labels(filename):
    with gzip.open(filename, "rb") as f:
        struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)


def scale(images):
    # Same scaling as the book: pixels from [0, 255] to [-1, 1]
    return ((images / 255.0) - 0.5) * 2


def load_mnist():
    for filename in FILES.values():
        download(filename)

    X_train = scale(read_images(FILES["train_images"]))
    y_train = read_labels(FILES["train_labels"])
    X_test = scale(read_images(FILES["test_images"]))
    y_test = read_labels(FILES["test_labels"])

    # Hold out the last 5,000 training images for validation
    X_valid, y_valid = X_train[-5000:], y_train[-5000:]
    X_train, y_train = X_train[:-5000], y_train[:-5000]
    return X_train, y_train, X_valid, y_valid, X_test, y_test


##########################
### TRAINING / EVALUATION
##########################

def compute_loss_and_acc(model, X, y, batch_size=1000):
    loss, correct = 0.0, 0
    for start in range(0, X.shape[0], batch_size):
        features = X[start:start + batch_size]
        targets = y[start:start + batch_size]
        _, probas = model.forward(features)
        loss -= np.sum(np.log(probas[np.arange(len(targets)), targets] + 1e-12))
        correct += np.sum(np.argmax(probas, axis=1) == targets)
    return loss / X.shape[0], correct / X.shape[0]


def train(model, X_train, y_train, X_valid, y_valid,
          epochs, batch_size, learning_rate, seed=123):
    rng = np.random.RandomState(seed)

    for epoch in range(epochs):
        order = rng.permutation(X_train.shape[0])

        for start in range(0, X_train.shape[0], batch_size):
            idx = order[start:start + batch_size]
            features, labels = X_train[idx], y_train[idx]

            a_h, a_out = model.forward(features)
            d_w_out, d_b_out, d_w_h, d_b_h = model.backward(features, a_h, a_out, labels)

            model.weight_out -= learning_rate * d_w_out
            model.bias_out -= learning_rate * d_b_out
            model.weight_h -= learning_rate * d_w_h
            model.bias_h -= learning_rate * d_b_h

        train_loss, train_acc = compute_loss_and_acc(model, X_train, y_train)
        _, valid_acc = compute_loss_and_acc(model, X_valid, y_valid)
        print(f"Epoch {epoch + 1:03d}/{epochs:03d} "
              f"| Train Loss: {train_loss:.4f} "
              f"| Train Acc: {train_acc * 100:.2f}% "
              f"| Valid Acc: {valid_acc * 100:.2f}%")


##########################
### SAVE / LOAD / PREDICT
##########################

def save_model(model, path):
    np.savez(path, weight_h=model.weight_h, bias_h=model.bias_h,
             weight_out=model.weight_out, bias_out=model.bias_out)
    print(f"Model saved to {path}")


def load_model(path):
    params = np.load(path)
    num_hidden, num_features = params["weight_h"].shape
    model = NeuralNetMLP(num_features, num_hidden, params["weight_out"].shape[0])
    for name in ("weight_h", "bias_h", "weight_out", "bias_out"):
        setattr(model, name, params[name])
    return model


def predict_image(model, image_path):
    from PIL import Image
    image = Image.open(image_path).convert("L").resize((28, 28))
    features = scale(np.asarray(image, dtype=np.float64).reshape(1, -1))
    _, probas = model.forward(features)
    return int(np.argmax(probas, axis=1)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=50, help="Number of hidden units")
    parser.add_argument("--save", default="mnist_mlp.npz", help="Where to save trained weights")
    parser.add_argument("--load", help="Load saved weights instead of training")
    parser.add_argument("--image", help="Optional 28x28 grayscale PNG to classify")
    args = parser.parse_args()

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_mnist()
    print(f"Train: {X_train.shape[0]} | Valid: {X_valid.shape[0]} | Test: {X_test.shape[0]}")

    if args.load:
        model = load_model(args.load)
    else:
        model = NeuralNetMLP(num_features=784, num_hidden=args.hidden, num_classes=10)
        train(model, X_train, y_train, X_valid, y_valid,
              args.epochs, args.batch_size, args.learning_rate)
        save_model(model, args.save)

    _, test_acc = compute_loss_and_acc(model, X_test, y_test)
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    if args.image:
        print(f"Prediction for {args.image}: {predict_image(model, args.image)}")


if __name__ == "__main__":
    main()
