"""
STAGE 6: MODEL (ORCHESTRATOR)

The highest-level stage of the pipeline: ties Layers, Activations, Losses,
Optimizers and Accuracy together, driving the model's entire lifecycle
(construction, forward pass, backward pass, training).
"""

from .activations import Activation_Softmax
from .layers import Layer_Input
from .losses import (
    Activation_Softmax_Loss_CategoricalCrossentropy,
    Loss_CategoricalCrossentropy,
)


class Model:
    """
    Orchestrator class representing an entire neural network end-to-end.

    STAGE: Model (the pipeline's highest level: ties Layers, Activations,
    Loss, Optimizer and Accuracy together, driving the model's full
    lifecycle: construction, forward pass, backward pass, training).

    Typical usage:
        1. Instantiate a Model.
        2. Add layers/activations in order with add().
        3. Configure loss, optimizer and accuracy with set().
        4. Finalize the architecture with finalize() (links the layers
           together into a chain and identifies the trainable layers).
        5. Train the model with train().

    This class is completely dataset-agnostic: it only expects NumPy
    arrays (X, y) with the shapes each layer/loss/accuracy documents.
    """

    def __init__(self):
        """
        Initialize an empty model.

        STAGE: model construction (the first step, before adding any
        layer).
        """
        # Create a list of network objects
        # Ordered list of layers/activations that make up the network,
        # populated by the user via the add() method.
        self.layers = []
        # Softmax classifier's output object
        # Optional reference to the optimized Softmax+CrossEntropy
        # combination, set automatically by finalize() when applicable;
        # stays None otherwise.
        self.softmax_classifier_output = None

    def add(self, layer):
        """
        Add a layer or activation to the model's chain.

        STAGE: architecture construction (called multiple times by the
        user, once for every component of the network, in the order data
        must flow through them).

        Parameters:
            layer: an instance of a layer (e.g. Layer_Dense,
                Layer_Dropout) or of an activation (e.g. Activation_ReLU,
                Activation_Softmax).
        """
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        """
        Configure the loss function, the optimizer and the accuracy metric
        to use during training.

        STAGE: pipeline construction (called once, after adding every
        layer with add() and before finalize()).

        Parameters (keyword-only):
            loss: an instance of a Loss subclass.
            optimizer: an instance of an optimizer (e.g. Optimizer_Adam).
            accuracy: an instance of an Accuracy subclass.
        """
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):
        """
        Link every added layer into a navigable chain (each one knows its
        "prev" and "next"), identify the trainable layers, and enable any
        applicable optimization (combined Softmax+CrossEntropy).

        STAGE: pipeline construction (called once, after add() and set(),
        before train()). This step is required because forward() and
        backward() rely on the prev/next references built here.
        """

        # Create and set the input layer
        # Create the "dummy" starting node of the chain, which will expose
        # the original X data through the same interface as every other
        # layer.
        self.input_layer = Layer_Input()

        # Count all the objects
        layer_count = len(self.layers)

        # Initialize a list containing trainable layers:
        # A list that will hold only the layers with trainable parameters
        # (those with a "weights" attribute, typically Layer_Dense
        # instances).
        self.trainable_layers = []

        # Iterate the objects
        # Walk through the list of layers to build the bidirectional links
        # (prev/next) that let forward()/backward() traverse the chain in
        # both directions.
        for i in range(layer_count):
            # If it's the first layer,
            # the previous layer object is the input layer
            # The chain's first layer has the just-created dummy input
            # node as its "previous" layer.
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            # All layers except for the first and the last
            # Intermediate layers simply connect to the previous and next
            # layer in the list.
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            # The last layer - the next object is the loss
            # Also let's save aside the reference to the last object
            # whose output is the model's output
            # The last layer has the loss object as its "next" (for
            # symmetry with the chain), and is saved separately as
            # "output_layer_activation": it is the one that produces the
            # final predictions via its own predictions() method.
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # If layer contains an attribute called "weights",
            # it's a trainable layer -
            # add it to the list of trainable layers
            # We don't need to check for biases -
            # checking for weights is enough
            # Recognize "trainable" layers by the presence of a weights
            # attribute (only Layer_Dense has it among this library's
            # components); these are the layers the optimizer will need to
            # update.
            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])

        # Update loss object with trainable layers
        # Tell the loss object which layers are trainable, needed to
        # compute the L1/L2 regularization loss.
        self.loss.remember_trainable_layers(self.trainable_layers)

        # If output activation is Softmax and
        # loss function is Categorical Cross-Entropy
        # create an object of combined activation
        # and loss function containing
        # faster gradient calculation
        # Automatic optimization: if the model is set up for standard
        # multi-class classification (final activation Softmax + loss
        # Categorical Cross-Entropy), prepare the combined class that
        # speeds up and stabilizes the gradient computation in the
        # backward pass.
        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(
            self.loss, Loss_CategoricalCrossentropy
        ):
            # Create an object of combined activation
            # and loss functions
            self.softmax_classifier_output = (
                Activation_Softmax_Loss_CategoricalCrossentropy()
            )

    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):
        """
        Run the complete training loop: forward pass, loss computation,
        backward pass and parameter updates, repeated for the given number
        of epochs.

        STAGE: training loop (the main entry point for training the model,
        called by the user after add(), set() and finalize()).

        Parameters (keyword-only besides X, y):
            X (ndarray): training set input data.
            y (ndarray): training set ground truth values (labels).
            epochs (int): number of training epochs (one epoch = one full
                forward+backward+update pass over the entire training set,
                since there is no mini-batching here).
            print_every (int): how many epochs between each printed
                metrics summary.
            validation_data (optional tuple): a pair (X_val, y_val) to
                evaluate the model on, without updating parameters, at the
                end of training.
        """

        # Initialize accuracy object
        # Initialize the accuracy object (computes the precision threshold
        # for regression, does nothing for classification).
        self.accuracy.init(y)

        # Main training loop
        # Main loop: every epoch runs a full forward -> loss -> backward
        # -> parameter update cycle.
        for epoch in range(1, epochs + 1):
            # Perform the forward pass
            # Compute the model's output over the entire training set, in
            # training mode (so dropout is active, if present).
            output = self.forward(X, training=True)

            # Calculate loss
            # Compute both the "data" loss and the regularization loss,
            # and sum them to get the epoch's total loss.
            data_loss, regularization_loss = self.loss.calculate(
                output, y, include_regularization=True
            )
            loss = data_loss + regularization_loss

            # Get predictions and calculate an accuracy
            # Convert the model's raw output into interpretable
            # predictions (e.g. class index) via the last activation's
            # predictions() method, then compute the accuracy.
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            # Perform backward pass
            # Compute every gradient in the network, from the loss down to
            # the first layer, via backpropagation.
            self.backward(output, y)

            # Optimize (update parameters)
            # Apply the parameter (weights/biases) update to every
            # trainable layer, using the configured optimizer:
            # pre_update_params() updates the learning rate (decay),
            # update_params() updates each layer's parameters,
            # post_update_params() increments the iteration counter.
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            # Print a summary
            # Periodically print a summary of the training metrics, useful
            # to monitor how training is progressing.
            if not epoch % print_every:
                # For regression, the tolerance-based accuracy above is a
                # heuristic; also compute R^2, an objective, dataset-
                # independent goodness-of-fit metric (see
                # Accuracy_Regression.r2_score for details). Detected via
                # hasattr so classification accuracies (which don't
                # implement r2_score) are left untouched.
                r2_suffix = ""
                if hasattr(self.accuracy, "r2_score"):
                    r2 = self.accuracy.r2_score(predictions, y)
                    r2_suffix = f", r2: {r2:.3f}"

                print(
                    f"epoch: {epoch}, "
                    + f"acc: {accuracy:.3f}, "
                    + f"loss: {loss:.3f} ("
                    + f"data_loss: {data_loss:.3f}, "
                    + f"reg_loss: {regularization_loss:.3f}), "
                    + f"lr: {self.optimizer.current_learning_rate}"
                    + r2_suffix
                )

        # If there is the validation data
        # If validation data was provided, evaluate the model on it at the
        # end of training (not during, in this implementation).
        if validation_data is not None:
            # For better readability
            X_val, y_val = validation_data

            # Perform the forward pass
            # Forward pass in inference mode (training=False): dropout is
            # disabled to evaluate the "complete" network.
            output = self.forward(X_val, training=False)

            # Calculate the loss
            # Only the data loss is computed during validation
            # (regularization is a training-time term, it makes no sense
            # to evaluate it on validation data).
            loss = self.loss.calculate(output, y_val)

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)

            # Print a summary
            # Print the validation metrics, to compare the model's
            # performance on data never seen during training. As above,
            # also report R^2 for regression models: it is the number to
            # trust for an objective read of validation performance,
            # since the tolerance-based accuracy's threshold is only a
            # heuristic.
            r2_suffix = ""
            if hasattr(self.accuracy, "r2_score"):
                r2 = self.accuracy.r2_score(predictions, y_val)
                r2_suffix = f", r2: {r2:.3f}"

            print(
                f"validation, "
                + f"acc: {accuracy:.3f}, "
                + f"loss: {loss:.3f}"
                + r2_suffix
            )

    def forward(self, X, training):
        """
        Run the forward pass of the entire network, propagating the data
        through every layer in the chain, in the order they were added.

        STAGE: forward pass (called both during training and during
        validation/inference).

        Parameters:
            X (ndarray): input data.
            training (bool): whether we are in training or inference mode
                (relevant for layers like Dropout).

        Returns:
            ndarray: the model's final output (from the last layer/
            activation in the chain).
        """

        # Call forward method on the input layer
        # this will set the output property that
        # the first layer in "prev" object is expecting
        # Kick off the chain by setting the dummy input node's output to
        # the raw X data.
        self.input_layer.forward(X, training)

        # Call forward method of every object in a chain
        # Pass output of the previous object as a parameter
        # Every layer reads its own "prev" output (set in the previous
        # step or in the previous iteration of this loop) and computes its
        # own output: this way data flows in cascade through the entire
        # network.
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # "layer" is now the last object from the list,
        # return its output
        # After the loop, "layer" points at the last element of the chain:
        # its output is the model's final prediction.
        return layer.output

    def backward(self, output, y):
        """
        Run the backward pass of the entire network (backpropagation),
        computing the gradients for every layer starting from the loss.

        STAGE: backward pass (called once per epoch, during training,
        right after the loss computation).

        Parameters:
            output (ndarray): the output produced by the forward pass (the
                model's predictions).
            y (ndarray): ground truth values.
        """

        # If softmax classifier
        # If the combined Softmax+CrossEntropy optimization was enabled
        # (set up in finalize()), use a more efficient backward-pass path.
        if self.softmax_classifier_output is not None:
            # First call backward method
            # on the combined activation/loss
            # this will set dinputs property
            # Directly compute the combined gradient (prediction -
            # target), skipping the separate and more expensive
            # computation of Softmax's Jacobian.
            self.softmax_classifier_output.backward(output, y)

            # Since we'll not call backward method of the last layer
            # which is Softmax activation
            # as we used combined activation/loss
            # object, let's set dinputs in this object
            # Since the "normal" backward of the last layer (Softmax) will
            # not be called, manually copy the already-computed gradient
            # into its dinputs attribute, so the previous layer (in the
            # loop below) can still read it from layer.next.dinputs as if
            # the flow had been regular.
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            # Call backward method going through
            # all the objects but last
            # in reversed order passing dinputs as a parameter
            # Walk backwards through every layer EXCEPT the last (already
            # handled above), propagating the gradient from one layer to
            # the previous one.
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        # First call backward method on the loss
        # this will set dinputs property that the last
        # layer will try to access shortly
        # "Standard" path (used when the Softmax+CrossEntropy optimization
        # does not apply): first compute the gradient of the loss w.r.t.
        # the model's predictions.
        self.loss.backward(output, y)

        # Call backward method going through all the objects
        # in reversed order passing dinputs as a parameter
        # Then walk the entire layer chain in reverse order (from last to
        # first), applying the chain rule to each: every layer receives
        # the gradient computed by the next one (layer.next.dinputs) and
        # produces its own (layer.dinputs), to be passed to the layer
        # still further back.
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
