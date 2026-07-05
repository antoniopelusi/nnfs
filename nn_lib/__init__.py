"""
nn_lib - A from-scratch (NumPy-only) neural network library.

PIPELINE STAGES (in logical order, from low-level building block to the
training loop):

1. LAYERS        -> Layer_Dense, Layer_Dropout, Layer_Input
                    Transform the data flowing through the network
                    (forward) and compute gradients for backpropagation
                    (backward).

2. ACTIVATIONS   -> Activation_ReLU, Activation_Softmax, Activation_Sigmoid,
                    Activation_Linear
                    Introduce non-linearity and/or turn the network's raw
                    output into an interpretable form (probabilities, a
                    continuous value, etc).

3. OPTIMIZERS    -> Optimizer_SGD, Optimizer_Adagrad, Optimizer_RMSprop,
                    Optimizer_Adam, DPWrapper
                    Decide HOW to update every layer's weights/biases using
                    the gradients computed during the backward pass.
                    DPWrapper wraps any of the other optimizers with
                    gradient clipping and noise injection, as an
                    educational approximation of DP-SGD -- it does NOT
                    provide a formal differential-privacy guarantee (see
                    its docstring in optimizers.py for the caveats).

4. LOSSES        -> Loss, Loss_CategoricalCrossentropy,
                    Activation_Softmax_Loss_CategoricalCrossentropy,
                    Loss_BinaryCrossentropy, Loss_MeanSquaredError,
                    Loss_MeanAbsoluteError
                    Measure how far the model's predictions are from the
                    ground truth and provide the starting point for
                    backpropagation (dinputs).

5. ACCURACY      -> Accuracy, Accuracy_Categorical, Accuracy_Regression
                    Measure prediction quality in a human-readable way
                    (e.g. % correct). NOT used for training, only for
                    reporting/monitoring.

6. MODEL         -> Model
                    Ties every stage above together: builds the layer
                    chain, runs the forward/backward passes, and drives
                    the training loop (epochs, optimization, metric
                    printing).

This package is dataset-agnostic: it contains no reference to any specific
dataset. Bring your own data (X, y) as NumPy arrays and build a pipeline
with the classes exported here. See the `tests/` scripts at the project
root for complete, runnable examples on different datasets.
"""

from .accuracy import (
    Accuracy,
    Accuracy_Categorical,
    Accuracy_Regression,
)
from .activations import (
    Activation_Linear,
    Activation_ReLU,
    Activation_Sigmoid,
    Activation_Softmax,
)
from .layers import (
    Layer_Dense,
    Layer_Dropout,
    Layer_Input,
)
from .losses import (
    Activation_Softmax_Loss_CategoricalCrossentropy,
    Loss,
    Loss_BinaryCrossentropy,
    Loss_CategoricalCrossentropy,
    Loss_MeanAbsoluteError,
    Loss_MeanSquaredError,
)
from .model import Model
from .optimizers import (
    DPWrapper,
    Optimizer_Adagrad,
    Optimizer_Adam,
    Optimizer_RMSprop,
    Optimizer_SGD,
)
from .utils import init

__all__ = [
    "init",
    "Layer_Dense",
    "Layer_Dropout",
    "Layer_Input",
    "Activation_ReLU",
    "Activation_Softmax",
    "Activation_Sigmoid",
    "Activation_Linear",
    "Optimizer_SGD",
    "Optimizer_Adagrad",
    "Optimizer_RMSprop",
    "Optimizer_Adam",
    "DPWrapper",
    "Loss",
    "Loss_CategoricalCrossentropy",
    "Activation_Softmax_Loss_CategoricalCrossentropy",
    "Loss_BinaryCrossentropy",
    "Loss_MeanSquaredError",
    "Loss_MeanAbsoluteError",
    "Accuracy",
    "Accuracy_Categorical",
    "Accuracy_Regression",
    "Model",
]
