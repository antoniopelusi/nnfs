"""
Test pipeline #3 (DP variant) - Regression on a synthetic sine wave, with
a differentially-private optimizer.

Same architecture as test_regression_sine.py:
    Dense -> ReLU -> Dense -> ReLU -> Dense -> Linear
    Loss:      Mean Squared Error
    Optimizer: Adam, wrapped in DPWrapper (gradient clipping + noise)
    Accuracy:  Regression (tolerance-based)

See DPWrapper's docstring (optimizers.py) for the caveats of this
simplified DP-SGD approximation: clipping and noise are applied to the
already-aggregated batch gradient, not per individual sample, so this
does NOT provide formal (epsilon, delta) privacy guarantees. It is meant
to illustrate the effect of gradient clipping and noise injection on
training.

The dataset is a single noisy sine wave: 1 input feature (x), 1 output
target (y = sin(x) + noise). It is generated locally with NumPy, so no
external dataset or network access is needed.

Run from the project root with:
    python test_regression_sine_dp.py
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nn_lib import (
    Accuracy_Regression,
    Activation_Linear,
    Activation_ReLU,
    DPWrapper,
    Layer_Dense,
    Loss_MeanSquaredError,
    Model,
    Optimizer_Adam,
    init,
)


def make_sine_dataset(n_samples=1000, noise=0.05, seed=0):
    """
    Generate a synthetic noisy sine-wave regression dataset.

    Parameters:
        n_samples (int): number of (x, y) points to generate.
        noise (float): standard deviation of the Gaussian noise added to
            the target values.
        seed (int): random seed for reproducibility.

    Returns:
        (ndarray, ndarray): X of shape (n_samples, 1), y of shape
        (n_samples, 1).
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2 * np.pi, 2 * np.pi, size=(n_samples, 1))
    y = np.sin(X) + rng.normal(0, noise, size=X.shape)
    return X, y


def main():
    init(0)

    X, y = make_sine_dataset(n_samples=1500, noise=0.05, seed=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Standardizing the single input feature helps the network converge
    # faster; the target (already in the small range [-1, 1]) is left
    # untouched.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # A slightly deeper network with two hidden layers, since fitting a
    # sine wave requires more representational capacity than a linearly
    # separable problem.
    model = Model()
    model.add(Layer_Dense(X_train.shape[1], 64))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(64, 64))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(64, 1))
    model.add(Activation_Linear())

    # Wrap the usual Adam optimizer with DPWrapper: gradients are clipped
    # to a maximum L2 norm and perturbed with Gaussian noise before Adam
    # applies its update rule. The clip_norm here is somewhat larger than
    # in the classification tests, since regression gradients on this
    # deeper network tend to have a larger natural scale.
    model.set(
        loss=Loss_MeanSquaredError(),
        optimizer=DPWrapper(
            Optimizer_Adam(learning_rate=0.01, decay=1e-3),
            clip_norm=2.0,
            noise_multiplier=0.3,
        ),
        accuracy=Accuracy_Regression(),
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
