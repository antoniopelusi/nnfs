"""
STAGE 4: LOSSES

Measure how far the model's predictions are from the ground truth and
provide the starting point for backpropagation (dinputs).
"""

import numpy as np


class Loss:
    """
    Base class for all loss functions.

    STAGE: Loss (provides functionality shared by every loss subclass:
    computing the mean batch loss and computing the regularization loss).

    Subclasses (e.g. Loss_CategoricalCrossentropy) must implement their own
    forward() method (per-sample loss computation) and backward() method
    (computation of the initial gradient for backpropagation).
    """

    def regularization_loss(self):
        """
        Compute the loss contribution coming from L1/L2 regularization
        across every trainable layer of the model.

        STAGE: loss computation (called by calculate() when
        include_regularization=True, typically during training, not
        during validation).

        Returns:
            float: total regularization loss summed over all trainable
            layers.
        """

        # 0 by default
        # Starting value: if no layer has regularization enabled, the
        # regularization loss stays at 0.
        regularization_loss = 0

        # Calculate regularization loss
        # iterate all trainable layers
        # Iterate over every layer with trainable parameters (set by
        # Model.finalize() via remember_trainable_layers).
        for layer in self.trainable_layers:
            # L1 regularization - weights
            # calculate only when factor greater than 0
            # L1 penalty: sum of the absolute value of the weights,
            # weighted by the regularization coefficient. Only computed if
            # the coefficient is greater than zero (optimization).
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(
                    np.abs(layer.weights)
                )

            # L2 regularization - weights
            # L2 penalty: sum of the squared weights, weighted by the
            # regularization coefficient.
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(
                    layer.weights * layer.weights
                )

            # L1 regularization - biases
            # calculate only when factor greater than 0
            # Same reasoning as L1 on weights, applied to the biases.
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(
                    np.abs(layer.biases)
                )

            # L2 regularization - biases
            # Same reasoning as L2 on weights, applied to the biases.
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(
                    layer.biases * layer.biases
                )

        return regularization_loss

    def remember_trainable_layers(self, trainable_layers):
        """
        Store a reference to the model's trainable layers, needed to
        compute the regularization loss.

        STAGE: model construction/finalization (called by Model.finalize(),
        once).

        Parameters:
            trainable_layers (list): the layers that own trainable
                parameters (weights/biases), typically Layer_Dense.
        """
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization=False):
        """
        Compute the mean batch loss (data loss) and, optionally, the
        regularization loss.

        STAGE: loss computation (called every epoch, both during training
        and validation, right after the model's forward pass).

        Parameters:
            output (ndarray): the model's predictions (output of the last
                activation).
            y (ndarray): ground truth values, either as sparse or one-hot
                labels depending on the task.
            include_regularization (bool): if True, also returns the
                regularization loss (used during training, not
                validation).

        Returns:
            float or (float, float): the data loss alone, or the tuple
            (data_loss, regularization_loss) if requested.
        """

        # Calculate sample losses
        # Delegate to the specific subclass (e.g. CategoricalCrossentropy)
        # the computation of the loss for each sample in the batch.
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        # The batch's "official" loss is the mean of the individual sample
        # losses.
        data_loss = np.mean(sample_losses)

        # If just data loss - return it
        # If regularization is not needed (e.g. during validation), only
        # the data loss is returned.
        if not include_regularization:
            return data_loss

        # Return the data and regularization losses
        # During training, both components are returned, to be summed
        # externally into the total loss.
        return data_loss, self.regularization_loss()


class Loss_CategoricalCrossentropy(Loss):
    """
    Categorical Cross-Entropy loss, used for multi-class (mutually
    exclusive) classification problems.

    STAGE: Loss (typically paired with the Softmax activation as the
    network's final activation).

    Purpose: measures how far the probability distribution predicted by
    the model is from the true class, heavily penalizing confident but
    wrong predictions.
    """

    def forward(self, y_pred, y_true):
        """
        Compute the cross-entropy for each sample in the batch.

        STAGE: loss computation (called by Loss.calculate()).

        Parameters:
            y_pred (ndarray): probabilities predicted by the model (output
                of Softmax), shape (n_samples, n_classes).
            y_true (ndarray): ground truth, either as sparse labels (shape
                (n_samples,)) or one-hot (shape (n_samples, n_classes)).

        Returns:
            ndarray: a vector with the loss for each sample.
        """

        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        # Clip the predicted probabilities into a range that excludes
        # exact 0 and 1, to avoid log(0) = -infinity; the clipping is
        # symmetric (also on the upper side) so it does not introduce a
        # systematic bias in the final mean.
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        # If labels are "sparse" (a single integer per sample indicating
        # the correct class), directly extract the predicted probability
        # for each sample's correct class.
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # Mask values - only for one-hot encoded labels
        # If labels are one-hot (a vector with a 1 on the correct class
        # and 0 elsewhere), multiply element-wise and sum per row: this is
        # equivalent to extracting the correct class's probability, but
        # also works with soft labels.
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        # The cross-entropy for a single sample is the negative log of the
        # probability assigned to the correct class: the closer that
        # probability is to 1, the closer the loss is to 0.
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        """
        Compute the gradient of cross-entropy w.r.t. the predicted
        probabilities.

        STAGE: backward pass (the first gradient computed in the
        backpropagation chain, the starting point for the output layer).
        Note: if the output activation is Softmax, the combined
        Activation_Softmax_Loss_CategoricalCrossentropy class is preferred
        for a more efficient and numerically stable computation.

        Parameters:
            dvalues (ndarray): the model's output (predicted
                probabilities), shape (n_samples, n_classes).
            y_true: ground truth (sparse or one-hot).
        """

        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        # If labels are sparse (integers), convert them to one-hot format
        # so the gradient formula can be applied in a vectorized way
        # across every class.
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        # Derivative of cross-entropy w.r.t. the predicted probabilities:
        # -y_true / y_pred (the standard log-loss derivative formula).
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        # Divide by the number of samples to get an average gradient,
        # consistent with using the mean loss in the forward pass; this
        # way the gradient does not "explode" with larger batches.
        self.dinputs = self.dinputs / samples


class Activation_Softmax_Loss_CategoricalCrossentropy:
    """
    Optimized combination of Softmax + Categorical Cross-Entropy for the
    backward pass.

    STAGE: bridge between Activation (stage 2) and Loss (stage 4), used
    exclusively to speed up and numerically stabilize the gradient
    computation when the final activation is Softmax and the loss is
    Categorical Cross-Entropy (the most common case in multi-class
    classification).

    The forward pass of Softmax and of the loss remain the ones from the
    original classes (Model calls them separately); this class only
    intervenes in the backward pass, exploiting a mathematical
    simplification (the combined derivative of softmax+cross-entropy
    reduces to "prediction - target", far simpler and more efficient than
    Softmax's full Jacobian).
    """

    def backward(self, dvalues, y_true):
        """
        Compute the combined Softmax + Cross-Entropy gradient in a
        simplified, efficient way.

        STAGE: backward pass (replaces the separate calls to
        Activation_Softmax.backward() and
        Loss_CategoricalCrossentropy.backward(), used by Model.backward()
        when applicable).

        Parameters:
            dvalues (ndarray): the model's output (probabilities predicted
                by Softmax), shape (n_samples, n_classes).
            y_true (ndarray): ground truth, sparse or one-hot.
        """

        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        # If labels are one-hot, convert them to sparse labels (class
        # index), since the simplified formula works per-index.
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        # Copy the predicted probabilities so they can be modified without
        # altering the original array (which might be used elsewhere).
        self.dinputs = dvalues.copy()
        # Calculate gradient
        # Simplified formula for the combined derivative: subtract 1 from
        # the predicted probability of each sample's correct class (the
        # mathematical equivalent of the separate Softmax+CCE computation,
        # but far more efficient).
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        # Normalize by the number of samples, consistent with the mean
        # loss used in the forward pass.
        self.dinputs = self.dinputs / samples


class Loss_BinaryCrossentropy(Loss):
    """
    Binary Cross-Entropy loss, used for binary or multi-label
    classification problems (non mutually-exclusive classes).

    STAGE: Loss (typically paired with the Sigmoid activation as the
    network's final activation).

    Purpose: measures the discrepancy between the predicted probability
    and the true binary label (0 or 1), for one or more independent
    outputs per sample.
    """

    def forward(self, y_pred, y_true):
        """
        Compute the binary cross-entropy for each sample in the batch.

        STAGE: loss computation.

        Parameters:
            y_pred (ndarray): predicted probabilities (output of Sigmoid).
            y_true (ndarray): true binary labels (0 or 1), same shape as
                y_pred.

        Returns:
            ndarray: the mean (across outputs) loss for each sample.
        """

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        # As in categorical cross-entropy, clip the predicted probability
        # to avoid log(0), symmetrically on both sides.
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise loss
        # Binary cross-entropy formula: sum of the two terms (for the
        # positive and negative class), applied element-wise to each
        # output of the sample.
        sample_losses = -(
            y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        # Average over outputs (useful when there are several independent
        # binary labels per sample, e.g. multi-label).
        sample_losses = np.mean(sample_losses, axis=-1)

        # Return losses
        return sample_losses

    def backward(self, dvalues, y_true):
        """
        Compute the gradient of binary cross-entropy w.r.t. the predicted
        probabilities.

        STAGE: backward pass.

        Parameters:
            dvalues (ndarray): probabilities predicted by the model.
            y_true (ndarray): true binary labels.
        """

        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        # Same clipping as in the forward pass, needed because the
        # derivative contains divisions by y_pred and (1 - y_pred).
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Calculate gradient
        # Derivative of binary cross-entropy w.r.t. the predicted
        # probability, divided by the number of outputs (consistent with
        # the mean computed in the forward pass).
        self.dinputs = (
            -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        )
        # Normalize gradient
        # Further normalize by the number of samples in the batch.
        self.dinputs = self.dinputs / samples


class Loss_MeanSquaredError(Loss):  # L2 loss
    """
    Mean Squared Error (MSE) loss, also known as "L2 loss".

    STAGE: Loss (typically used for regression problems, paired with the
    linear activation as the network's final activation).

    Purpose: quadratically penalizes the distance between the predicted
    and true value, giving larger errors proportionally more weight.
    """

    def forward(self, y_pred, y_true):
        """
        Compute the mean squared error for each sample in the batch.

        STAGE: loss computation.

        Parameters:
            y_pred (ndarray): values predicted by the model.
            y_true (ndarray): true values, same shape as y_pred.

        Returns:
            ndarray: the mean (across outputs) loss for each sample.
        """

        # Calculate loss
        # Mean of the squared difference between prediction and true
        # value, computed across each sample's outputs (useful for
        # multi-output regression).
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=-1)

        # Return losses
        return sample_losses

    def backward(self, dvalues, y_true):
        """
        Compute the gradient of MSE w.r.t. the predicted values.

        STAGE: backward pass.

        Parameters:
            dvalues (ndarray): values predicted by the model.
            y_true (ndarray): true values.
        """

        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Gradient on values
        # Derivative of MSE w.r.t. the prediction: -2*(y_true - y_pred),
        # divided by the number of outputs, consistent with the mean in
        # the forward pass.
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalize gradient
        # Further normalize by the number of samples in the batch.
        self.dinputs = self.dinputs / samples


class Loss_MeanAbsoluteError(Loss):  # L1 loss
    """
    Mean Absolute Error (MAE) loss, also known as "L1 loss".

    STAGE: Loss (an alternative to MSE for regression problems, less
    sensitive to outliers because it penalizes linearly instead of
    quadratically).
    """

    def forward(self, y_pred, y_true):
        """
        Compute the mean absolute error for each sample in the batch.

        STAGE: loss computation.

        Parameters:
            y_pred (ndarray): values predicted by the model.
            y_true (ndarray): true values, same shape as y_pred.

        Returns:
            ndarray: the mean (across outputs) loss for each sample.
        """

        # Calculate loss
        # Mean of the absolute difference between prediction and true
        # value, across each sample's outputs.
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

        # Return losses
        return sample_losses

    def backward(self, dvalues, y_true):
        """
        Compute the gradient of MAE w.r.t. the predicted values.

        STAGE: backward pass.

        Parameters:
            dvalues (ndarray): values predicted by the model.
            y_true (ndarray): true values.
        """

        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Calculate gradient
        # The derivative of the absolute value is the sign function: +1
        # if the prediction is below the true value, -1 if above. It is
        # normalized by the number of outputs.
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize gradient
        # Further normalize by the number of samples in the batch.
        self.dinputs = self.dinputs / samples
