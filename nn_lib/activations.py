"""
STAGE 2: ACTIVATIONS

Functions that introduce non-linearity into the network and/or turn the
final output into an interpretable form (probabilities, a continuous
value, etc).
"""

import numpy as np


class Activation_ReLU:
    """
    ReLU (Rectified Linear Unit) activation: f(x) = max(0, x).

    STAGE: Activation (typically applied right after a Layer_Dense, in the
    network's hidden layers).

    Purpose: introduces non-linearity into the network (without it,
    stacking several Layer_Dense instances would be equivalent to a single
    linear layer) and is computationally very cheap. It is the standard
    activation for hidden layers in modern networks.
    """

    def forward(self, inputs, training):
        """
        Apply the ReLU function element-wise.

        STAGE: forward pass.

        Parameters:
            inputs (ndarray): output of the previous layer (e.g.
                Layer_Dense).
            training (bool): unused, present for interface consistency.
        """
        # Remember input values
        # Inputs are stored: they are needed in the backward pass to know
        # where the gradient must be zeroed out (where the input was <= 0).
        self.inputs = inputs
        # Calculate output values from inputs
        # Apply max(0, x) element-wise: negative values become 0, positive
        # values are left unchanged.
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        """
        Compute the gradient of ReLU w.r.t. its inputs.

        STAGE: backward pass.

        Parameters:
            dvalues (ndarray): gradient of the loss w.r.t. this
                activation's output.
        """
        # Since we need to modify original variable,
        # let's make a copy of values first
        # Copy the incoming gradient so it can be modified without
        # altering the object passed in by the next layer.
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        # The derivative of ReLU is 0 where the input was <= 0, so the
        # gradient must not propagate through those points (set to zero).
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        """
        Return predictions from this activation's outputs.

        STAGE: evaluation/inference (used by Model to compute accuracy;
        ReLU is normally a hidden-layer activation, so this method exists
        mainly for interface consistency).

        Parameters:
            outputs (ndarray): this activation's output.

        Returns:
            ndarray: the same values passed in, since ReLU does not
            produce a specific "interpretable" shape like classes or
            probabilities.
        """
        return outputs


class Activation_Softmax:
    """
    Softmax activation: turns a vector of raw values (logits) into a
    probability distribution that sums to 1.

    STAGE: Activation (typically the network's final activation, used for
    multi-class classification problems).

    Purpose: makes the network's output interpretable as "how confident
    the model is that the sample belongs to each class".
    """

    def forward(self, inputs, training):
        """
        Compute Softmax probabilities from the incoming logits.

        STAGE: forward pass (last step before the loss computation in a
        multi-class classification problem).

        Parameters:
            inputs (ndarray): logits produced by the previous layer (one
                row per sample, one column per class).
            training (bool): unused, present for interface consistency.
        """
        # Remember input values
        # Stored for interface consistency (not reused directly in the
        # backward pass, which relies on self.output).
        self.inputs = inputs

        # Get unnormalized probabilities
        # Subtract each row's maximum before exponentiating: a numerical
        # trick to avoid overflow (exponentials of large numbers) without
        # changing the final softmax result.
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        # Divide each exponentiated value by its row's sum, so that every
        # row (sample) sums to exactly 1: this is the definition of
        # softmax.
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, dvalues):
        """
        Compute the gradient of Softmax w.r.t. its inputs, using the
        Jacobian matrix for each individual sample.

        STAGE: backward pass. Note: when Softmax is paired with
        Categorical Cross-Entropy, the
        Activation_Softmax_Loss_CategoricalCrossentropy class is used
        instead for a much more efficient computation, skipping this
        method entirely.

        Parameters:
            dvalues (ndarray): gradient of the loss w.r.t. the softmax
                output.
        """

        # Create uninitialized array
        # Allocate the output array without initializing it (faster); it
        # will be fully filled in by the loop below.
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        # Unlike other layers, softmax's derivative is not element-wise
        # but requires a Jacobian matrix per sample (because every output
        # depends on every input), so we iterate sample by sample.
        for index, (single_output, single_dvalues) in enumerate(
            zip(self.output, dvalues)
        ):
            # Flatten output array
            # Turn this sample's output vector into a column vector,
            # needed for the matrix operations below.
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            # Build the Softmax Jacobian matrix for this sample:
            # diag(output) - output @ output.T (the standard formula for
            # the softmax derivative).
            jacobian_matrix = np.diagflat(single_output) - np.dot(
                single_output, single_output.T
            )
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            # Apply the chain rule by multiplying the Jacobian by the
            # incoming gradient for this sample, and store the result in
            # the corresponding position of the final array.
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

    def predictions(self, outputs):
        """
        Convert Softmax probabilities into the predicted class.

        STAGE: evaluation/inference.

        Parameters:
            outputs (ndarray): probabilities produced by softmax (one row
                per sample, one column per class).

        Returns:
            ndarray: index of the highest-probability class, for each
            sample (the discrete label predicted by the model).
        """
        return np.argmax(outputs, axis=1)


class Activation_Sigmoid:
    """
    Sigmoid activation: f(x) = 1 / (1 + e^-x).

    STAGE: Activation (typically the network's final activation for binary
    classification problems, or multi-label problems).

    Purpose: squashes any real value into the (0, 1) range, interpretable
    as the probability of belonging to the positive class.
    """

    def forward(self, inputs, training):
        """
        Compute the sigmoid output from the inputs.

        STAGE: forward pass.

        Parameters:
            inputs (ndarray): logits produced by the previous layer.
            training (bool): unused, present for interface consistency.
        """
        # Save input and calculate/save output
        # of the sigmoid function
        # Store the input (for consistency) and directly compute the
        # sigmoid function element-wise.
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        """
        Compute the gradient of Sigmoid w.r.t. its inputs.

        STAGE: backward pass.

        Parameters:
            dvalues (ndarray): gradient of the loss w.r.t. the sigmoid
                output.
        """
        # Derivative - calculates from output of the sigmoid function
        # The sigmoid derivative conveniently expresses itself in terms of
        # its own output: sigmoid'(x) = output * (1 - output). The chain
        # rule is applied by multiplying by the incoming gradient.
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        """
        Convert sigmoid probabilities into binary predictions (0/1).

        STAGE: evaluation/inference.

        Parameters:
            outputs (ndarray): probabilities produced by sigmoid.

        Returns:
            ndarray: 1 where the probability exceeds 0.5, 0 otherwise
            (the standard decision threshold for binary classification).
        """
        return (outputs > 0.5) * 1


class Activation_Linear:
    """
    Linear (identity) activation: f(x) = x.

    STAGE: Activation (typically the network's final activation for
    regression problems, where the output must be able to take any real
    value, not limited to a specific range).

    Purpose: applies no transformation at all, letting the value produced
    by the previous layer become the model's final output directly.
    """

    def forward(self, inputs, training):
        """
        Pass the inputs through unchanged as output (identity function).

        STAGE: forward pass.

        Parameters:
            inputs (ndarray): output of the previous layer.
            training (bool): unused, present for interface consistency.
        """
        # Just remember values
        # Store both input and output (identical) for interface
        # consistency with the other layers/activations.
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        """
        Compute the gradient of the linear activation (which is 1, so the
        gradient passes through unchanged).

        STAGE: backward pass.

        Parameters:
            dvalues (ndarray): gradient of the loss w.r.t. the linear
                activation's output.
        """
        # derivative is 1, 1 * dvalues = dvalues - the chain rule
        # Since the derivative of f(x)=x is 1, the gradient propagates
        # unchanged (it is only copied to avoid mutating the original
        # array).
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        """
        Return predictions for a regression problem.

        STAGE: evaluation/inference.

        Parameters:
            outputs (ndarray): output of the linear activation.

        Returns:
            ndarray: the same values, since in regression the raw output
            IS already the final prediction.
        """
        return outputs
