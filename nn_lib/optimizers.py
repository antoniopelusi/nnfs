"""
STAGE 3: OPTIMIZERS

Algorithms that decide HOW to update every trainable layer's
weights/biases, using the gradients computed during the backward pass.
"""

import numpy as np


class Optimizer_SGD:
    """
    Stochastic Gradient Descent optimizer (with optional momentum and
    learning rate decay).

    STAGE: Optimizer (used during training, after the backward pass, to
    update every trainable layer's weights/biases).

    Purpose: the simplest optimization algorithm: moves parameters in the
    direction opposite to the gradient, scaled by the learning rate.
    Momentum adds "inertia" to the updates, helping to escape shallow
    local minima and speeding up convergence.
    """

    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        """
        Initialize the SGD optimizer.

        STAGE: training pipeline construction (called once, before
        starting training).

        Parameters:
            learning_rate (float): step size used to update parameters.
            decay (float): learning rate decay rate over time (0 = no
                decay).
            momentum (float): fraction of the previous update kept in the
                new update (0 = no momentum, equivalent to "vanilla" SGD).
        """
        self.learning_rate = learning_rate
        # "Current" learning rate, which may be modified by decay at every
        # epoch; starts equal to the initial learning rate.
        self.current_learning_rate = learning_rate
        self.decay = decay
        # Counter of completed training iterations (epochs), used to
        # compute learning rate decay.
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        """
        Update the current learning rate based on decay, before applying
        parameter updates for the current epoch.

        STAGE: training loop (called once per epoch, before iterating over
        the trainable layers).
        """
        if self.decay:
            # Decay formula: the learning rate progressively decreases as
            # iterations increase, allowing finer adjustments towards the
            # end of training.
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer):
        """
        Update a trainable layer's weights and biases using the gradient
        computed in the backward pass (with or without momentum).

        STAGE: training loop (called once for every trainable layer, at
        every epoch, after the backward pass).

        Parameters:
            layer: a layer object (e.g. Layer_Dense) exposing dweights,
                dbiases (gradients) and weights, biases (parameters to
                update).
        """

        # If we use momentum
        if self.momentum:
            # If layer does not contain momentum arrays, create them
            # filled with zeros
            # If this is the first time this layer is updated with
            # momentum, create the "velocity" (momentum) arrays with the
            # same shape as the weights and biases, initialized to zero.
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights
                # The array doesn't exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.biases)
            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            # The update is a combination of the "direction" of the
            # previous step (scaled by momentum) and the current gradient
            # (scaled by the learning rate): this gives inertia to the
            # movement through parameter space.
            weight_updates = (
                self.momentum * layer.weight_momentums
                - self.current_learning_rate * layer.dweights
            )
            layer.weight_momentums = weight_updates

            # Build bias updates
            # Same momentum logic as for weights, applied to biases.
            bias_updates = (
                self.momentum * layer.bias_momentums
                - self.current_learning_rate * layer.dbiases
            )
            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (as before momentum update)
        else:
            # Without momentum, the update is simply the gradient scaled
            # by (minus) the learning rate: we step in the direction
            # opposite to the gradient.
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        # Update weights and biases using either
        # vanilla or momentum updates
        # Actually apply the computed update to the layer's parameters.
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        """
        Increment the iteration counter after updating every trainable
        layer for the current epoch.

        STAGE: training loop (called once per epoch, after updating all
        layers).
        """
        self.iterations += 1


class Optimizer_Adagrad:
    """
    Adagrad (Adaptive Gradient) optimizer.

    STAGE: Optimizer.

    Purpose: adapts the learning rate for every individual parameter based
    on its historical gradients: parameters that received large updates in
    the past are slowed down, those with small updates are sped up. Useful
    with sparse data, but tends to slow learning down too much in the long
    run because the gradient cache grows monotonically.
    """

    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7):
        """
        Initialize the Adagrad optimizer.

        STAGE: training pipeline construction.

        Parameters:
            learning_rate (float): base update step size.
            decay (float): learning rate decay rate over time.
            epsilon (float): small constant to avoid division by zero when
                normalizing the gradient.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        """
        Update the current learning rate based on decay.

        STAGE: training loop (called once per epoch, before parameter
        updates).
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer):
        """
        Update a trainable layer's weights and biases, normalizing the
        gradient by the historical cache of squared gradients.

        STAGE: training loop (called for every trainable layer, at every
        epoch, after the backward pass).

        Parameters:
            layer: a layer object exposing dweights, dbiases, weights,
                biases.
        """

        # If layer does not contain cache arrays,
        # create them filled with zeros
        # If this is the first update for this layer, create the "cache"
        # arrays (accumulation of squared gradients), initialized to zero.
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        # The cache accumulates the square of the current gradients: it
        # grows monotonically with every update, "remembering" how much a
        # parameter has been updated in the past.
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        # The standard update (gradient * learning rate) is divided by the
        # square root of the cache: parameters with a large cache
        # (historically updated a lot) receive smaller updates, and vice
        # versa. Epsilon avoids division by zero.
        layer.weights += (
            -self.current_learning_rate
            * layer.dweights
            / (np.sqrt(layer.weight_cache) + self.epsilon)
        )
        layer.biases += (
            -self.current_learning_rate
            * layer.dbiases
            / (np.sqrt(layer.bias_cache) + self.epsilon)
        )

    def post_update_params(self):
        """
        Increment the iteration counter.

        STAGE: training loop (called once per epoch).
        """
        self.iterations += 1


class Optimizer_RMSprop:
    """
    RMSprop (Root Mean Square Propagation) optimizer.

    STAGE: Optimizer.

    Purpose: like Adagrad, adapts the learning rate for every parameter
    based on the history of its gradients, but uses an exponential moving
    average (instead of a cumulative sum) to prevent the cache from
    growing unbounded, fixing Adagrad's excessive slowdown problem.
    """

    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, rho=0.9):
        """
        Initialize the RMSprop optimizer.

        STAGE: training pipeline construction.

        Parameters:
            learning_rate (float): base update step size.
            decay (float): learning rate decay rate over time.
            epsilon (float): constant to avoid division by zero.
            rho (float): "memory" factor for the moving average of the
                cache (values close to 1 = longer memory).
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        """
        Update the current learning rate based on decay.

        STAGE: training loop (called once per epoch).
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer):
        """
        Update a trainable layer's weights and biases using an
        exponential moving average of squared gradients to normalize the
        update.

        STAGE: training loop (called for every trainable layer, at every
        epoch, after the backward pass).

        Parameters:
            layer: a layer object exposing dweights, dbiases, weights,
                biases.
        """

        # If layer does not contain cache arrays,
        # create them filled with zeros
        # Initialize the cache to zero on first use, as in Adagrad.
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        # Unlike Adagrad, the cache here is an exponential moving average:
        # a fraction rho of the previous cache is kept, and (1-rho) of the
        # current squared gradient is added. This prevents the cache from
        # growing unbounded over time.
        layer.weight_cache = (
            self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        )
        layer.bias_cache = (
            self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2
        )

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        # Same normalization principle as Adagrad, but based on the
        # "limited memory" cache just computed.
        layer.weights += (
            -self.current_learning_rate
            * layer.dweights
            / (np.sqrt(layer.weight_cache) + self.epsilon)
        )
        layer.biases += (
            -self.current_learning_rate
            * layer.dbiases
            / (np.sqrt(layer.bias_cache) + self.epsilon)
        )

    def post_update_params(self):
        """
        Increment the iteration counter.

        STAGE: training loop (called once per epoch).
        """
        self.iterations += 1


class Optimizer_Adam:
    """
    Adam (Adaptive Moment Estimation) optimizer.

    STAGE: Optimizer (the most commonly used optimizer in practice for
    neural networks, often the default choice).

    Purpose: combines the benefits of momentum (moving average of the
    gradient, "first moment") and RMSprop (moving average of the squared
    gradient, "second moment"), and additionally applies a bias correction
    that compensates for the fact that, early in training, the moving
    averages are biased towards zero (since they are initialized at zero).
    """

    def __init__(
        self, learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999
    ):
        """
        Initialize the Adam optimizer.

        STAGE: training pipeline construction.

        Parameters:
            learning_rate (float): base update step size.
            decay (float): learning rate decay rate over time.
            epsilon (float): constant to avoid division by zero.
            beta_1 (float): memory factor for the moving average of the
                gradient (first moment, momentum-like).
            beta_2 (float): memory factor for the moving average of the
                squared gradient (second moment, RMSprop-like).
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        """
        Update the current learning rate based on decay.

        STAGE: training loop (called once per epoch).
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer):
        """
        Update a trainable layer's weights and biases by combining
        momentum (first moment) and adaptive normalization (second
        moment), with initial bias correction.

        STAGE: training loop (called for every trainable layer, at every
        epoch, after the backward pass).

        Parameters:
            layer: a layer object exposing dweights, dbiases, weights,
                biases.
        """

        # If layer does not contain cache arrays,
        # create them filled with zeros
        # On first use, initialize both the momentums (first moment) and
        # the cache (second moment) to zero, for weights and biases.
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum  with current gradients
        # Exponential moving average of the current gradient (first
        # moment): similar to "classic" momentum as seen in SGD.
        layer.weight_momentums = (
            self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        )
        layer.bias_momentums = (
            self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        )
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        # Bias correction: in the first iterations the moving average is
        # biased towards zero (since it started at zero); this correction
        # compensates for that effect, dividing by a factor that
        # approaches 1 as iterations increase.
        weight_momentums_corrected = layer.weight_momentums / (
            1 - self.beta_1 ** (self.iterations + 1)
        )
        bias_momentums_corrected = layer.bias_momentums / (
            1 - self.beta_1 ** (self.iterations + 1)
        )
        # Update cache with squared current gradients
        # Exponential moving average of the squared gradient (second
        # moment): as in RMSprop, estimates the recent "scale" of the
        # gradients.
        layer.weight_cache = (
            self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        )
        layer.bias_cache = (
            self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        )
        # Get corrected cache
        # Same bias correction applied to the second moment.
        weight_cache_corrected = layer.weight_cache / (
            1 - self.beta_2 ** (self.iterations + 1)
        )
        bias_cache_corrected = layer.bias_cache / (
            1 - self.beta_2 ** (self.iterations + 1)
        )

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        # The final update uses the corrected momentum (a "smooth"
        # direction for the gradient) normalized by the square root of the
        # corrected cache (a per-parameter adaptive scale), combining the
        # benefits of momentum and RMSprop.
        layer.weights += (
            -self.current_learning_rate
            * weight_momentums_corrected
            / (np.sqrt(weight_cache_corrected) + self.epsilon)
        )
        layer.biases += (
            -self.current_learning_rate
            * bias_momentums_corrected
            / (np.sqrt(bias_cache_corrected) + self.epsilon)
        )

    def post_update_params(self):
        """
        Increment the iteration counter.

        STAGE: training loop (called once per epoch).
        """
        self.iterations += 1
