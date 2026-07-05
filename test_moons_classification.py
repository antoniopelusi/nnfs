"""
Test pipeline #2 - Binary classification on the "two moons" dataset.

Demonstrates the binary classification setup:
    Dense -> ReLU -> Dense -> Sigmoid
    Loss:      Binary Cross-Entropy
    Optimizer: SGD with momentum and learning rate decay
    Accuracy:  Categorical (binary mode)

The two-moons dataset is a classic non-linearly-separable toy problem
(two interleaving half-circles), useful to check that the network can
learn a non-trivial decision boundary.

Run from the project root with:
    python test_moons_classification.py
"""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nn_lib import (
    Accuracy_Categorical,
    Activation_ReLU,
    Activation_Sigmoid,
    Layer_Dense,
    Loss_BinaryCrossentropy,
    Model,
    Optimizer_SGD,
    init,
)


def main():
    init(0)

    # Generate a synthetic two-moons dataset: 2 features, 2 classes,
    # not linearly separable. `noise` adds a bit of Gaussian jitter to
    # make the task slightly harder and more realistic.
    X, y = make_moons(n_samples=2000, noise=0.2, random_state=0)

    # Binary cross-entropy expects targets shaped (n_samples, 1) rather
    # than a flat (n_samples,) vector.
    y = y.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Model()
    model.add(Layer_Dense(X_train.shape[1], 64))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(64, 1))
    model.add(Activation_Sigmoid())

    model.set(
        loss=Loss_BinaryCrossentropy(),
        optimizer=Optimizer_SGD(learning_rate=1.0, decay=1e-3, momentum=0.9),
        accuracy=Accuracy_Categorical(binary=True),
    )

    model.finalize()

    model.train(
        X_train,
        y_train,
        epochs=2000,
        print_every=200,
        validation_data=(X_test, y_test),
    )


if __name__ == "__main__":
    main()
