"""
Test pipeline #4 - Multi-class classification with regularization and
dropout, on a synthetic "blobs" dataset with added label noise.

Demonstrates a more elaborate multi-class classification setup, closer to
what is needed on a noisier, higher-dimensional problem:
    Dense (L2 regularized) -> ReLU -> Dropout -> Dense -> Softmax
    Loss:      Categorical Cross-Entropy
    Optimizer: Adam with learning rate decay
    Accuracy:  Categorical

This mirrors the architecture/technique choices (L2 regularization +
dropout) useful to fight overfitting on harder, noisier datasets.

Run from the project root with:
    python test_blobs_classification.py
"""

import os
import sys

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from nn_lib import (
    Accuracy_Categorical,
    Activation_ReLU,
    Activation_Softmax,
    Layer_Dense,
    Layer_Dropout,
    Loss_CategoricalCrossentropy,
    Model,
    Optimizer_Adam,
    init,
)


def add_label_noise(y, fraction, n_classes, seed=0):
    """
    Randomly flip a fraction of the labels to a different class, to make
    the classification task noisier and less trivially separable.

    Parameters:
        y (ndarray): original integer class labels.
        fraction (float): fraction of labels to flip (0-1).
        n_classes (int): total number of classes.
        seed (int): random seed for reproducibility.

    Returns:
        ndarray: labels with noise applied.
    """
    rng = np.random.RandomState(seed)
    y_noisy = y.copy()
    n_flip = int(len(y) * fraction)
    flip_idx = rng.choice(len(y), size=n_flip, replace=False)
    y_noisy[flip_idx] = rng.randint(0, n_classes, size=n_flip)
    return y_noisy


def main():
    init(0)

    n_classes = 4

    # Generate a synthetic multi-class dataset with 8 informative
    # features and some cluster spread, plus injected label noise, so the
    # model actually benefits from regularization and dropout instead of
    # trivially memorizing the training set.
    X, y = make_blobs(
        n_samples=3000,
        centers=n_classes,
        n_features=8,
        cluster_std=3.0,
        random_state=0,
    )
    y = add_label_noise(y, fraction=0.05, n_classes=n_classes, seed=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Model()
    model.add(
        Layer_Dense(
            X_train.shape[1],
            128,
            weight_regularizer_l2=5e-4,
            bias_regularizer_l2=5e-4,
        )
    )
    model.add(Activation_ReLU())
    model.add(Layer_Dropout(0.2))
    model.add(Layer_Dense(128, n_classes))
    model.add(Activation_Softmax())

    model.set(
        loss=Loss_CategoricalCrossentropy(),
        optimizer=Optimizer_Adam(learning_rate=0.02, decay=5e-5),
        accuracy=Accuracy_Categorical(),
    )

    model.finalize()

    model.train(
        X_train,
        y_train,
        epochs=1000,
        print_every=100,
        validation_data=(X_test, y_test),
    )


if __name__ == "__main__":
    main()
