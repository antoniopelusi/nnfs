"""
Test pipeline #4 (DP variant) - Multi-class classification with
regularization and dropout, on a synthetic "blobs" dataset with added
label noise, using a differentially-private optimizer.

Same architecture as test_blobs_classification.py:
    Dense (L2 regularized) -> ReLU -> Dropout -> Dense -> Softmax
    Loss:      Categorical Cross-Entropy
    Optimizer: Adam with learning rate decay, wrapped in DPWrapper
               (gradient clipping + noise)
    Accuracy:  Categorical

See DPWrapper's docstring (optimizers.py) for the caveats of this
simplified DP-SGD approximation: clipping and noise are applied to the
already-aggregated batch gradient, not per individual sample, so this
does NOT provide formal (epsilon, delta) privacy guarantees. It is meant
to illustrate the effect of gradient clipping and noise injection on
training, on top of the L2 regularization and dropout already used to
fight overfitting on this noisier, higher-dimensional problem.

Run from the project root with:
    python test_blobs_classification_dp.py
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nn_lib import (
    Accuracy_Categorical,
    Activation_ReLU,
    Activation_Softmax,
    DPWrapper,
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

    # Wrap the usual Adam optimizer with DPWrapper: gradients are clipped
    # to a maximum L2 norm and perturbed with Gaussian noise before Adam
    # applies its update rule. The clip_norm here is somewhat larger than
    # in the simpler classification tests, since this deeper/wider layer
    # (128 neurons, L2-regularized) tends to produce larger gradients.
    model.set(
        loss=Loss_CategoricalCrossentropy(),
        optimizer=DPWrapper(
            Optimizer_Adam(learning_rate=0.02, decay=5e-5),
            clip_norm=2.0,
            noise_multiplier=0.3,
        ),
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
