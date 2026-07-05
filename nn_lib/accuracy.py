"""
STAGE 5: ACCURACY (EVALUATION METRICS)

Measure prediction quality in a human-readable way (e.g. % of correct
classifications). These metrics are NOT used for training: they only help
monitor the model's performance, during both training and validation.
"""

import numpy as np


class Accuracy:
    """
    Base class for computing the model's accuracy.

    STAGE: Accuracy (a reporting metric, NOT used for training: it only
    serves to monitor the model's performance in a way the user can
    interpret, both during training and validation).

    Subclasses must implement compare(), which defines what a "correct
    prediction" means for the specific type of task (classification or
    regression).
    """

    def calculate(self, predictions, y):
        """
        Compute the mean accuracy by comparing predictions and ground
        truth values.

        STAGE: evaluation (called every epoch by the Model's training
        loop, both on training and validation data, after obtaining
        predictions from the output activation).

        Parameters:
            predictions (ndarray): the model's predictions, already
                converted into a form comparable with y (e.g. a class
                index).
            y (ndarray): ground truth values.

        Returns:
            float: the fraction (0-1) of correct predictions.
        """

        # Get comparison results
        # Delegate to the specific subclass the element-wise comparison
        # between predictions and ground truth values.
        comparisons = self.compare(predictions, y)

        # Calculate an accuracy
        # The mean of a boolean array (True=1, False=0) is exactly the
        # fraction of correct predictions.
        accuracy = np.mean(comparisons)

        # Return accuracy
        return accuracy


class Accuracy_Categorical(Accuracy):
    """
    Accuracy computation for classification models (multi-class or
    binary).

    STAGE: Accuracy (used when the output activation is Softmax or
    Sigmoid).
    """

    def __init__(self, *, binary=False):
        """
        Initialize the accuracy object for classification.

        STAGE: pipeline construction (set up in Model.set()).

        Parameters:
            binary (bool): True if the task is binary classification
                (labels already in 0/1 form), False for multi-class
                classification with labels possibly one-hot encoded.
        """
        # Binary mode?
        self.binary = binary

    def init(self, y):
        """
        No initialization needed for classification (unlike regression,
        where a precision threshold based on the data must be computed).

        STAGE: training preparation (called once at the start of
        Model.train(), for interface consistency with
        Accuracy_Regression).

        Parameters:
            y (ndarray): ground truth values of the training set (unused
                here).
        """
        pass

    def compare(self, predictions, y):
        """
        Compare discrete predictions to the ground truth values.

        STAGE: evaluation (called by Accuracy.calculate()).

        Parameters:
            predictions (ndarray): class indices predicted by the model.
            y (ndarray): ground truth values, either sparse labels or
                one-hot.

        Returns:
            ndarray: a boolean array, True where the prediction is
            correct.
        """
        # If we are not in binary mode and labels are one-hot (2D array),
        # convert them to sparse labels (class index) so they can be
        # compared directly with the predictions.
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y


class Accuracy_Regression(Accuracy):
    """
    Accuracy computation for regression models.

    STAGE: Accuracy (used when the output activation is linear).

    Unlike classification, regression has no natural notion of an "exact"
    prediction: a tolerance threshold (precision) is therefore defined,
    within which a prediction is considered "correct".
    """

    def __init__(self):
        """
        Initialize the accuracy object for regression.

        STAGE: pipeline construction (set up in Model.set()).
        """
        # Create precision property
        # The tolerance threshold (precision) is computed later, once the
        # ground truth data is known (see init()).
        self.precision = None

    def init(self, y, reinit=False):
        """
        Compute the tolerance threshold (precision) based on the
        variability of the training set's ground truth values.

        STAGE: training preparation (called once at the start of
        Model.train(), before the epoch loop).

        Parameters:
            y (ndarray): ground truth values of the training set, used to
                estimate the data's variability.
            reinit (bool): if True, force recomputation of the threshold
                even if already set (useful when reusing the same model on
                different data).
        """
        if self.precision is None or reinit:
            # The threshold is proportional to the standard deviation of
            # the ground truth values: datasets with highly variable
            # values will get a larger tolerance, more "compact" datasets
            # a smaller one. The factor of 250 is a heuristic to obtain a
            # reasonably strict threshold.
            self.precision = np.std(y) / 250

    def compare(self, predictions, y):
        """
        Compare continuous predictions to the ground truth values, within
        the tolerance threshold.

        STAGE: evaluation (called by Accuracy.calculate()).

        Parameters:
            predictions (ndarray): values predicted by the model.
            y (ndarray): ground truth values.

        Returns:
            ndarray: a boolean array, True where the absolute difference
            between prediction and ground truth is below the precision
            threshold.
        """
        return np.absolute(predictions - y) < self.precision

    def r2_score(self, predictions, y):
        """
        Compute the coefficient of determination (R^2), a scale-free,
        dataset-independent metric of regression fit quality.

        STAGE: evaluation (called by Model.train(), in addition to
        compare()/calculate(), whenever the accuracy object is an
        Accuracy_Regression instance).

        Unlike the tolerance-based `compare()` above -- whose threshold is
        an arbitrary heuristic and mostly useful to track relative
        progress across epochs on the SAME dataset -- R^2 has a fixed,
        well-known interpretation that holds across any dataset:
            R^2 = 1  -> perfect predictions
            R^2 = 0  -> the model is no better than always predicting the
                        mean of y
            R^2 < 0  -> the model is worse than that constant baseline
        This makes it possible to judge a regression run on its own,
        without needing to compare it against other runs first.

        Parameters:
            predictions (ndarray): values predicted by the model.
            y (ndarray): ground truth values.

        Returns:
            float: the R^2 score. Returns 0.0 in the degenerate case where
            y has zero variance (ss_tot == 0), to avoid a division by
            zero.
        """
        # Residual sum of squares: how much error the model actually
        # makes.
        ss_res = np.sum((y - predictions) ** 2)
        # Total sum of squares: how much error a trivial "always predict
        # the mean" baseline would make.
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        # Guard against the degenerate case of a constant target (no
        # variance to explain), which would otherwise divide by zero.
        if ss_tot == 0:
            return 0.0

        # R^2 measures the fraction of the target's variance explained by
        # the model, relative to that trivial baseline.
        return 1 - ss_res / ss_tot
