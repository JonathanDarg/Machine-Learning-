import argparse
import os

import numpy as np
from PIL import Image

from neuralnet import NeuralNetMLP


def load_images(data_dir):
    images = []
    labels = []

    for label in range(10):
        label_dir = os.path.join(data_dir, str(label))
        for filename in os.listdir(label_dir):
            if filename.lower().endswith(".png"):
                image_path = os.path.join(label_dir, filename)
                image = Image.open(image_path).convert("L")
                images.append(np.asarray(image, dtype=np.float32).reshape(-1) / 255.0)
                labels.append(label)

    return np.asarray(images), np.asarray(labels, dtype=np.int64)


def train(model, features, labels, epochs, batch_size, learning_rate, seed):
    rng = np.random.RandomState(seed)

    for epoch in range(epochs):
        order = rng.permutation(features.shape[0])
        features = features[order]
        labels = labels[order]

        for start in range(0, features.shape[0], batch_size):
            batch_features = features[start:start + batch_size]
            batch_labels = labels[start:start + batch_size]
            hidden, output = model.forward(batch_features)
            gradients = model.backward(batch_features, hidden, output, batch_labels)

            model.weight_out -= learning_rate * gradients[0]
            model.bias_out -= learning_rate * gradients[1]
            model.weight_h -= learning_rate * gradients[2]
            model.bias_h -= learning_rate * gradients[3]

        _, output = model.forward(features)
        predictions = np.argmax(output, axis=1)
        accuracy = np.mean(predictions == labels)
        print(f"Epoch {epoch + 1}/{epochs} - training accuracy: {accuracy:.4f}")


def predict(model, image_path):
    image = Image.open(image_path).convert("L")
    features = np.asarray(image, dtype=np.float32).reshape(1, -1) / 255.0
    _, output = model.forward(features)
    return int(np.argmax(output, axis=1)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="mnist", help="Folder containing folders 0 through 9")
    parser.add_argument("--image", help="Optional 28x28 PNG to classify after training")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    args = parser.parse_args()

    features, labels = load_images(args.data)
    model = NeuralNetMLP(num_features=features.shape[1], num_hidden=50, num_classes=10)
    print(f"Loaded {features.shape[0]} images with {features.shape[1]} pixels each.")
    train(model, features, labels, args.epochs, args.batch_size, args.learning_rate, seed=123)

    if args.image:
        print(f"Prediction for {args.image}: {predict(model, args.image)}")


if __name__ == "__main__":
    main()