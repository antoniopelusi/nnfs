"""
Test pipeline #1 (DP variant) - Multi-class classification on the Iris
dataset, with a differentially-private optimizer.

Same architecture as test_iris_classification.py:
    Dense -> ReLU -> Dense -> Softmax
    Loss:      Categorical Cross-Entropy
    Optimizer: Adam, wrapped in DPWrapper (gradient clipping + noise)
    Accuracy:  Categorical

See DPWrapper's docstring (optimizers.py) for the caveats of this
simplified DP-SGD approximation: clipping and noise are applied to the
already-aggregated batch gradient, not per individual sample, so this
does NOT provide formal (epsilon, delta) privacy guarantees. It is meant
to illustrate the effect of gradient clipping and noise injection on
training.

Run from the project root with:
    python test_iris_classification_dp.py
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nn_lib import (
    Accuracy_Categorical,
    Activation_ReLU,
    Activation_Softmax,
    DPWrapper,
    Layer_Dense,
    Loss_CategoricalCrossentropy,
    Model,
    Optimizer_Adam,
    init,
)


def main():
    # Fix the random seed for reproducible results.
    init(0)

    # Load the classic Iris dataset: 150 samples, 4 numeric features,
    # 3 balanced classes (setosa, versicolor, virginica). It ships with
    # scikit-learn, so no download/network access is required.
    data = load_iris()
    X, y = data.data, data.target

    # Split into training and test sets, stratifying on the label so both
    # splits keep the same class proportions.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    # Standardize features (zero mean, unit variance). Neural networks
    # trained with gradient descent converge much faster and more
    # reliably on scaled inputs.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build the model: a small network is enough for a dataset this
    # simple (4 input features, 3 output classes).
    model = Model()
    model.add(Layer_Dense(X_train.shape[1], 32))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(32, 3))
    model.add(Activation_Softmax())

    # Wrap the usual Adam optimizer with DPWrapper: gradients are clipped
    # to a maximum L2 norm and perturbed with Gaussian noise before Adam
    # applies its update rule.
    model.set(
        loss=Loss_CategoricalCrossentropy(),
        optimizer=DPWrapper(
            Optimizer_Adam(learning_rate=0.01, decay=1e-4),
            clip_norm=1.0,
            noise_multiplier=0.3,
        ),
        accuracy=Accuracy_Categorical(),
    )

    model.finalize()

    model.train(
        X_train,
        y_train,
        epochs=500,
        print_every=50,
        validation_data=(X_test, y_test),
    )


if __name__ == "__main__":
    main()
