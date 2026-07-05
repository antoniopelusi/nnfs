"""
Test pipeline #2 (DP variant) - Binary classification on the "two moons"
dataset, with a differentially-private optimizer.

Same architecture as test_moons_classification.py:
    Dense -> ReLU -> Dense -> Sigmoid
    Loss:      Binary Cross-Entropy
    Optimizer: SGD with momentum and learning rate decay, wrapped in
               DPWrapper (gradient clipping + noise)
    Accuracy:  Categorical (binary mode)

See DPWrapper's docstring (optimizers.py) for the caveats of this
simplified DP-SGD approximation: clipping and noise are applied to the
already-aggregated batch gradient, not per individual sample, so this
does NOT provide formal (epsilon, delta) privacy guarantees. It is meant
to illustrate the effect of gradient clipping and noise injection on
training.

The two-moons dataset is a classic non-linearly-separable toy problem
(two interleaving half-circles), useful to check that the network can
still learn a non-trivial decision boundary despite the added noise.

Run from the project root with:
    python test_moons_classification_dp.py
"""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nn_lib import (
    Accuracy_Categorical,
    Activation_ReLU,
    Activation_Sigmoid,
    DPWrapper,
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

    # Wrap the usual momentum-SGD optimizer with DPWrapper: gradients are
    # clipped to a maximum L2 norm and perturbed with Gaussian noise
    # before SGD applies its (momentum-based) update rule.
    model.set(
        loss=Loss_BinaryCrossentropy(),
        optimizer=DPWrapper(
            Optimizer_SGD(learning_rate=1.0, decay=1e-3, momentum=0.9),
            clip_norm=1.0,
            noise_multiplier=0.3,
        ),
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
