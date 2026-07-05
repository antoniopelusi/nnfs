"""
STAGE 1: LAYERS

Building blocks that transform the data flowing through the network
(forward pass) and compute gradients for backpropagation (backward pass).
"""

import numpy as np


class Layer_Dense:
    """
    Fully-connected ("dense") layer.

    STAGE: Layer (the network's basic building block).

    Represents a linear transformation of the incoming data:
        output = input @ weights + biases

    This is the fundamental unit the network is made of: stacking several
    Layer_Dense instances (with non-linear activations in between) forms
    the "trainable" part of the model, i.e. the part whose parameters
    (weights and biases) get updated during training.

    Also supports L1/L2 regularization on weights and biases, a mechanism
    to limit overfitting by penalizing overly large weights.
    """

    def __init__(
        self,
        n_inputs,
        n_neurons,
        weight_regularizer_l1=0,
        weight_regularizer_l2=0,
        bias_regularizer_l1=0,
        bias_regularizer_l2=0,
    ):
        """
        Initialize the layer's weights, biases and regularization
        coefficients.

        STAGE: model construction (called once per layer, before training,
        typically while defining the architecture).

        Parameters:
            n_inputs (int): number of incoming features (dimensionality of
                the input vector for each sample).
            n_neurons (int): number of neurons in this layer, i.e. the
                dimensionality of the output produced for each sample.
            weight_regularizer_l1/l2 (float): strength of L1/L2
                regularization applied to the weights.
            bias_regularizer_l1/l2 (float): strength of L1/L2
                regularization applied to the biases.
        """
        # Initialize weights and biases
        # Weights are initialized with small random Gaussian values
        # (scaled by 0.01) to break symmetry between neurons: if they all
        # started identical, they would all learn the same thing.
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # Biases start at zero: there is no need to break symmetry here,
        # since the weights already provide a source of asymmetry.
        self.biases = np.zeros((1, n_neurons))
        # Set regularization strength
        # Store the L1/L2 regularization coefficients for weights and
        # biases; these are used both when computing the regularization
        # loss (Loss class) and in this layer's own backward pass.
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs, training):
        """
        Perform the layer's linear transformation (forward pass).

        STAGE: forward pass of the network (called on every epoch/batch,
        both during training and inference).

        Parameters:
            inputs (ndarray): matrix of shape (n_samples, n_inputs) holding
                the incoming data (the previous layer's output).
            training (bool): whether we are in training or inference mode;
                not used directly here, but required for a consistent
                interface with other layers (e.g. Dropout).

        Effect:
            Sets self.output to the result of the linear transformation.
        """
        # Remember input values
        # Inputs are stored because they are needed in the backward pass
        # to compute the gradient w.r.t. the weights
        # (dweights = inputs.T @ dvalues).
        self.inputs = inputs
        # Calculate output values from inputs, weights and biases
        # Matrix product between inputs and weights, plus the bias
        # (automatically broadcast across every row/sample in the batch).
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        """
        Compute the gradients of the loss w.r.t. this layer's weights,
        biases and inputs (backward pass / backpropagation).

        STAGE: backward pass (called once per epoch/batch, after the
        forward pass and the loss computation, proceeding from the output
        layer back towards the input layer).

        Parameters:
            dvalues (ndarray): gradient of the loss w.r.t. this layer's
                output, computed by the next layer/function in the chain
                (chain rule).

        Effect:
            Sets self.dweights, self.dbiases (gradients to be consumed by
            the optimizer) and self.dinputs (gradient to propagate to the
            previous layer).
        """
        # Gradients on parameters
        # Derivative of the loss w.r.t. the weights: product of the
        # transposed inputs and the incoming gradient (chain rule).
        self.dweights = np.dot(self.inputs.T, dvalues)
        # Derivative of the loss w.r.t. the biases: sum of the gradients
        # across all samples in the batch (the bias is shared by every
        # sample).
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        # If L1 regularization on weights is enabled, add its contribution
        # to the gradient: the derivative of L1 is +1/-1 depending on the
        # sign of the weight.
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        # The derivative of L2 (sum of squares) w.r.t. the weights is
        # 2*weight, so this term is added to the already-computed gradient.
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        # L1 on biases
        # Same reasoning as L1 on weights, applied to the biases.
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        # Same reasoning as L2 on weights, applied to the biases.
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # Gradient on values
        # Gradient to propagate to the previous layer: product of the
        # incoming gradient and the transposed weights (chain rule).
        self.dinputs = np.dot(dvalues, self.weights.T)


class Layer_Dropout:
    """
    Dropout layer: a regularization technique that randomly disables a
    fraction of neurons during training.

    STAGE: Layer (an intermediate block, typically placed right after an
    activation, between one Layer_Dense and the next).

    Purpose: prevents the network from relying too heavily on specific
    neurons (co-adaptation), improving generalization and reducing
    overfitting. During inference (training=False) dropout is disabled and
    data passes through unchanged.
    """

    def __init__(self, rate):
        """
        Initialize the dropout layer.

        STAGE: model construction.

        Parameters:
            rate (float): probability of "dropping" (zeroing out) a
                neuron, between 0 and 1 (e.g. 0.1 = drop 10% of neurons).
        """
        # Store rate, we invert it as for example for dropout
        # of 0.1 we need success rate of 0.9
        # We store the "success rate" (probability that a neuron STAYS
        # active), since that's what the binomial distribution used in
        # the forward pass needs.
        self.rate = 1 - rate

    def forward(self, inputs, training):
        """
        Apply the dropout mask during training, or pass data through
        unchanged during inference.

        STAGE: forward pass of the network.

        Parameters:
            inputs (ndarray): incoming data.
            training (bool): if False, dropout is disabled (used during
                validation/inference, where the full, deterministic
                network is desired).
        """
        # Save input values
        # Stored for consistency with the other layers, even though it is
        # not reused directly in the backward pass here.
        self.inputs = inputs

        # If not in the training mode - return values
        # During inference, dropout must not alter the output: we simply
        # copy the inputs.
        if not training:
            self.output = inputs.copy()
            return

        # Generate and save scaled mask
        # Generate a binary (0/1) mask with probability self.rate of being
        # 1 for each element, then scale it by dividing by self.rate
        # ("inverted dropout"): this automatically compensates for the
        # reduced average activation, so no change is needed at inference
        # time.
        self.binary_mask = (
            np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        )
        # Apply mask to output values
        # Apply the mask element-wise: "dropped" neurons become 0, the
        # others are scaled to compensate.
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        """
        Propagate the gradient through the same mask used in the forward
        pass.

        STAGE: backward pass.

        Parameters:
            dvalues (ndarray): gradient of the loss w.r.t. this layer's
                output.
        """
        # Gradient on values
        # The derivative of dropout w.r.t. the input is the mask itself
        # (neurons dropped in the forward pass have zero gradient, the
        # others are scaled the same way as in the forward pass).
        self.dinputs = dvalues * self.binary_mask


class Layer_Input:
    """
    "Dummy" input layer, used only to give the layer chain a uniform
    interface.

    STAGE: Layer (the initial node of the chain, created automatically by
    Model.finalize(); it is not added manually by the user).

    Purpose: represents the network's entry point, so that the first real
    layer (e.g. Layer_Dense) can read its "output" (i.e. the original data
    X) through the same interface (layer.prev.output) used for every other
    layer in the chain.
    """

    def forward(self, inputs, training):
        """
        Expose the input data as this node's own output, unchanged.

        STAGE: forward pass (first step of the chain).

        Parameters:
            inputs (ndarray): the raw dataset features (X).
            training (bool): unused, present for interface consistency.
        """
        # Simply "forward" the incoming data as this node's output, so the
        # next layer can read it from layer.prev.output.
        self.output = inputs
