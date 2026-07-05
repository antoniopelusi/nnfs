# NN Lib — A From-Scratch Neural Network Library (NumPy)

A small, dependency-light library for building, training and evaluating
feed-forward neural networks using only NumPy, with no framework like
PyTorch or TensorFlow. It is meant to make every mathematical step that
happens inside a neural network explicit and inspectable: forward pass,
backward pass (backpropagation), optimization, regularization, loss
computation and metrics.

The library is completely **dataset-agnostic**: it has no built-in
dataset and no dependency on any dataset-loading package. You bring your
own `(X, y)` NumPy arrays and build a training pipeline out of the classes
it exposes. See the `test_*.py` scripts at the project root for complete,
runnable examples on different datasets and tasks.

## Project layout

```
.
├── src/
│   └── nn_lib/
│       ├── __init__.py      # public API (import everything from here)
│       ├── utils.py         # init() - reproducibility helper
│       ├── layers.py        # Layer_Dense, Layer_Dropout, Layer_Input
│       ├── activations.py   # ReLU, Softmax, Sigmoid, Linear
│       ├── optimizers.py    # SGD, Adagrad, RMSprop, Adam
│       ├── losses.py        # Categorical/Binary Cross-Entropy, MSE, MAE
│       ├── accuracy.py      # Accuracy_Categorical, Accuracy_Regression
│       └── model.py         # Model - the orchestrator class
├── test_iris_classification.py     # multi-class classification example
├── test_moons_classification.py    # binary classification example
├── test_regression_sine.py         # regression example
├── test_blobs_classification.py    # classification + dropout/L2 example
└── README.md
```

`src/nn_lib` is the library itself; everything else at the project root
is example/test code showing how to use it. To use the library from your
own script, add `src` to your Python path and import from `nn_lib`:

```python
import sys
sys.path.insert(0, "src")

from nn_lib import (
    init, Layer_Dense, Activation_ReLU, Activation_Softmax,
    Loss_CategoricalCrossentropy, Optimizer_Adam,
    Accuracy_Categorical, Model,
)
```

(Alternatively, install the project in editable mode with a
`pyproject.toml`/`setup.py` pointing at `src/`, if you want to `import
nn_lib` from anywhere without manipulating `sys.path`.)

## The pipeline, stage by stage

```
Data (X, y)
   │
   ▼
[1] LAYERS  ──►  [2] ACTIVATIONS  ──►  [4] LOSS  ──►  [3] OPTIMIZER
   ▲                                      │
   │                                      ▼
   └──────────── backward pass ◄──── [5] ACCURACY (reporting only)
                                          │
                                     [6] MODEL orchestrates everything
```

### Stage 1 — Layers (`nn_lib.layers`)

The "building blocks" that transform data as it flows through the
network. In this library, "layer" is used loosely to also include the
dummy input node -- the table below covers all three classes.

| Class | Role | Trainable? |
|---|---|---|
| `Layer_Dense` | Fully-connected ("dense") layer. | Yes (weights + biases) |
| `Layer_Dropout` | Regularization layer, randomly zeroes activations. | No |
| `Layer_Input` | Dummy entry point for the raw data. | No |

#### `Layer_Dense` — the only layer with learnable parameters

**What it does, technically:** computes `output = input @ weights +
biases`, a linear (affine) transformation.

**What it does, functionally:** this is where the network actually
"learns" -- weights and biases are the parameters the optimizer updates
every epoch. Everything else in the pipeline (activations, dropout,
loss) either shapes how gradients flow through this layer or measures
how well it is doing; `Layer_Dense` is the one component that accumulates
knowledge about the data.

**Constructor:** `Layer_Dense(n_inputs, n_neurons, weight_regularizer_l1=0,
weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0)`
- `n_inputs`: must match the number of features coming in -- either the
  dataset's feature count (for the first `Layer_Dense` in the network) or
  the `n_neurons` of the previous `Layer_Dense` (for every subsequent
  one). Getting this wrong is the most common setup mistake; there is no
  automatic shape inference.
- `n_neurons`: how many outputs this layer produces per sample; this
  becomes the next layer's `n_inputs`, or -- for the last `Layer_Dense`
  in the network -- must equal the number of classes (classification) or
  output values (regression) you are predicting.
- `weight_regularizer_l1/l2`, `bias_regularizer_l1/l2`: optional
  penalties added to the loss to discourage large weights/biases (see
  Stage 4). Leave at `0` (default) unless you observe overfitting (large
  gap between training and validation accuracy/loss).

**Where it goes in the pipeline:** every hidden block and the output
block of the network start with a `Layer_Dense`. It is always immediately
followed by an activation (Stage 2) -- never stack two `Layer_Dense`
directly, since without a non-linear activation in between they collapse
mathematically into a single linear layer, wasting capacity.

#### `Layer_Dropout` — overfitting control, used only between hidden blocks

**What it does, technically:** during training, multiplies the input by
a randomly generated binary mask (each unit kept with probability
`1 - rate`), then rescales the surviving activations by `1 / (1 - rate)`
("inverted dropout") so their expected magnitude is unchanged. During
inference (`training=False`) it is a no-op: input passes through
untouched.

**What it does, functionally:** forces the network to not rely on any
single neuron too heavily, since that neuron might be "dropped" on a
given training step. This spreads the learned representation across more
neurons and tends to improve generalization to unseen data.

**Constructor:** `Layer_Dropout(rate)`, where `rate` is the fraction of
neurons to drop (e.g. `0.1` = drop 10%, keep 90%). Typical values range
from `0.1` to `0.5`; higher rates regularize more aggressively but can
slow down or hurt training if the network is small or the dataset is
already easy.

**Where it goes in the pipeline:** always *after* a hidden activation
(`Dense -> Activation -> Dropout`), never right after a `Dense` (dropping
raw pre-activation values makes little sense) and never as the very last
component of the network (you don't want to randomly zero out final
predictions). Add it only when you have evidence of overfitting, or
preemptively on larger networks / smaller datasets where overfitting is
likely.

#### `Layer_Input` — internal only, never add manually

**What it does:** exposes the raw `X` array as `self.output`, so the
first real layer can read `layer.prev.output` through the exact same
interface used everywhere else in the chain.

**Where it goes:** nowhere you control -- `Model.finalize()` creates and
inserts it automatically as the head of the chain. It exists purely to
keep `Model.forward()`'s loop uniform (every layer, including the first,
reads from `layer.prev.output`).

#### Shared interface

Every layer exposes two symmetric methods:
- **`forward(inputs, training)`**: computes the output from the input.
  `training` is passed to every layer for interface consistency, even
  though only `Layer_Dropout` actually branches on it.
- **`backward(dvalues)`**: computes the gradient of the loss with respect
  to its own parameters (if any, stored as `dweights`/`dbiases`) and with
  respect to its input (`dinputs`), to propagate to the previous layer.

### Stage 2 — Activations (`nn_lib.activations`)

Functions (with no trainable parameters) that introduce non-linearity or
turn the final output into an interpretable form. Every `Layer_Dense` in
the network is followed by exactly one of these.

| Class | Formula | Placement | Output shape/range |
|---|---|---|---|
| `Activation_ReLU` | `f(x) = max(0, x)` | Hidden layers only | Same shape as input, values ≥ 0 |
| `Activation_Softmax` | `f(x)_i = exp(x_i) / sum(exp(x))` per row | Final layer, multi-class classification | `(n_samples, n_classes)`, each row sums to 1 |
| `Activation_Sigmoid` | `f(x) = 1 / (1 + e^-x)` | Final layer, binary/multi-label classification | Same shape as input, each value in (0, 1) |
| `Activation_Linear` | `f(x) = x` | Final layer, regression | Same shape as input, unconstrained |

#### `Activation_ReLU` — hidden-layer default

**Technical:** zeroes out negative values, passes positive values
through unchanged. Cheap to compute, cheap to differentiate (derivative
is either 0 or 1).

**Functional role:** without a non-linear activation between
`Layer_Dense` instances, stacking them would be mathematically equivalent
to a single linear layer, no matter how many you stack. ReLU is what lets
the network approximate non-linear functions (curved decision
boundaries, curves like a sine wave, etc). Use it after every hidden
`Layer_Dense`; it is essentially always the right default choice for
hidden layers in this kind of network.

#### `Activation_Softmax` — multi-class classification output only

**Technical:** exponentiates each row's values (after subtracting the
row max, for numerical stability) and normalizes so the row sums to 1.

**Functional role:** turns arbitrary real-valued logits into a proper
probability distribution over mutually-exclusive classes -- "the model
is 80% sure this is class 2, 15% class 0, 5% class 1". Use it **only**
as the very last activation, and only when classes are mutually
exclusive (each sample belongs to exactly one class). Its
`n_neurons` in the preceding `Layer_Dense` must equal the number of
classes.

#### `Activation_Sigmoid` — binary / multi-label output only

**Technical:** squashes each value independently into `(0, 1)`.

**Functional role:** interpretable as "the probability this specific
output is true/positive", computed independently per output unit. Use it
as the last activation for: (a) binary classification (one output unit,
threshold at 0.5), or (b) multi-label classification, where a sample can
belong to several non-exclusive classes at once (one output unit per
label, each independently 0/1). Do not use it for mutually-exclusive
multi-class problems -- use Softmax instead, since Sigmoid's outputs
don't sum to 1 and don't compete with each other.

#### `Activation_Linear` — regression output only

**Technical:** the identity function; gradient is always 1.

**Functional role:** the only activation that does not compress or
reshape the value range, which is exactly what a regression target
needs (a house price, a temperature, a coordinate -- any unconstrained
real number). Never use ReLU/Sigmoid/Softmax as the final activation for
regression: they would incorrectly restrict your predictions to
non-negative values, `(0,1)`, or a probability simplex, respectively.

#### Shared interface

Every activation exposes:
- **`forward(inputs, training)`** / **`backward(dvalues)`**: same
  contract as layers.
- **`predictions(outputs)`**: converts the raw activation output into an
  interpretable prediction -- e.g. `argmax` per row for Softmax
  (returns a class index), a `>0.5` threshold for Sigmoid (returns 0/1),
  or the value itself for Linear. `Model` calls this automatically on the
  output layer's activation; you never need to call it yourself.

### Stage 3 — Optimizers (`nn_lib.optimizers`)

Algorithms that, given the gradient computed in the backward pass, decide
**how** to update every trainable layer's weights and biases. They act
only on layers with parameters (`Layer_Dense`); activations and dropout
have nothing for an optimizer to update.

| Class | Update rule (conceptual) | Per-parameter adaptive? |
|---|---|---|
| `Optimizer_SGD` | `param += -lr * gradient` (+ optional momentum term) | No |
| `Optimizer_Adagrad` | `param += -lr * gradient / sqrt(cumulative_sum(gradient^2))` | Yes |
| `Optimizer_RMSprop` | `param += -lr * gradient / sqrt(moving_avg(gradient^2))` | Yes |
| `Optimizer_Adam` | RMSprop's adaptive scaling + a momentum-smoothed gradient, with bias correction | Yes |

#### `Optimizer_SGD` — the baseline

**Technical:** without momentum, each update is simply
`param -= learning_rate * gradient`. With `momentum > 0`, a fraction of
the previous update is carried over (`velocity = momentum * velocity -
lr * gradient`, then `param += velocity`), giving the descent "inertia".

**Functional role:** the simplest possible way to use the gradient --
straightforward to reason about, but sensitive to the choice of learning
rate: too high and training diverges/oscillates, too low and it crawls.
Momentum helps it push through small local bumps and flat regions instead
of stalling.

**Constructor:** `Optimizer_SGD(learning_rate=1., decay=0., momentum=0.)`.
Use a plain `Optimizer_SGD(learning_rate=...)` (no momentum) mainly for
teaching/debugging purposes, where you want the most literal, undamped
gradient-descent behavior. Add `momentum` (commonly `0.9`) when plain SGD
converges too slowly or gets stuck.

#### `Optimizer_Adagrad` — adaptive, but only good early on

**Technical:** accumulates the sum of squared gradients for each
parameter since the start of training, then divides the update by its
square root. Parameters that have historically received large gradients
get progressively smaller effective learning rates.

**Functional role:** useful when different parameters need very
different learning rates (e.g. sparse input features that some
parameters see rarely). Its major weakness is that the accumulated sum
only ever grows, so the effective learning rate keeps shrinking and can
stall training long before convergence on longer runs.

**Constructor:** `Optimizer_Adagrad(learning_rate=1., decay=0.,
epsilon=1e-7)`. Rarely the best first choice in this library; included
mainly for completeness and comparison -- prefer `Optimizer_RMSprop` or
`Optimizer_Adam` for most practical training runs.

#### `Optimizer_RMSprop` — Adagrad's fix

**Technical:** same idea as Adagrad, but the accumulator is an
exponential moving average (`cache = rho * cache + (1-rho) *
gradient^2`) instead of an ever-growing sum, so old gradients are
gradually "forgotten".

**Functional role:** keeps Adagrad's per-parameter adaptivity without the
runaway learning-rate decay, making it viable for longer training runs.
A solid choice when you want adaptive learning rates but are not sure
whether Adam's extra momentum term is helping or hurting on your data.

**Constructor:** `Optimizer_RMSprop(learning_rate=0.001, decay=0.,
epsilon=1e-7, rho=0.9)`. `rho` closer to 1 means longer memory of past
gradients (smoother, slower-adapting cache).

#### `Optimizer_Adam` — the default choice

**Technical:** maintains two moving averages per parameter -- the first
moment (mean of recent gradients, momentum-style) and the second moment
(mean of recent squared gradients, RMSprop-style) -- both corrected for
their initial bias toward zero, then combines them into the update.

**Functional role:** in practice, the combination of "smoothed gradient
direction" (momentum) and "per-parameter adaptive scale" (RMSprop-style)
makes Adam converge reliably across a very wide range of problems without
much tuning. It is why every `test_*.py` script in this repo uses it (or
SGD with momentum, chosen deliberately for `test_moons_classification.py`
to also demonstrate that path).

**Constructor:** `Optimizer_Adam(learning_rate=0.001, decay=0.,
epsilon=1e-7, beta_1=0.9, beta_2=0.999)`. Start here for any new
pipeline; only switch to another optimizer if you have a specific reason
to (e.g. teaching plain gradient descent, or comparing convergence
behavior).

#### Learning rate decay (`decay` parameter, shared by all four)

**Technical:** every epoch, `current_learning_rate = learning_rate / (1 +
decay * iterations)`, applied by `pre_update_params()` before that
epoch's updates.

**Functional role:** takes progressively smaller steps as training
advances, which helps settle into a minimum instead of oscillating around
it once the model is already close to a good solution. Every `test_*.py`
script sets a small non-zero `decay` (e.g. `1e-4` to `5e-5`) for this
reason; a `decay` of `0` (the default) keeps the learning rate constant
for the entire run.

#### Where this fits in the pipeline

Optimizers are configured once, in `model.set(optimizer=...)`, and never
appear in the `model.add(...)` chain -- they operate on the finished
network as a whole, not as a layer within it. `Model.train()` calls each
optimizer's three methods in this fixed order, every epoch:
1. `pre_update_params()` — applies learning rate decay.
2. `update_params(layer)` — updates a single layer's parameters (called
   once per trainable layer, i.e. once per `Layer_Dense`).
3. `post_update_params()` — increments the iteration counter used by both
   decay and Adam's bias correction.

### Stage 4 — Losses (`nn_lib.losses`)

Measure how far the model's predictions are from the ground truth, and
provide the starting gradient for backpropagation. This is the single
number `Optimizer` instances are, indirectly, trying to minimize.

| Class | `y_true` shape | Paired output activation |
|---|---|---|
| `Loss_CategoricalCrossentropy` | `(n_samples,)` sparse or `(n_samples, n_classes)` one-hot | `Activation_Softmax` |
| `Activation_Softmax_Loss_CategoricalCrossentropy` | Same as above | `Activation_Softmax` (internal optimization, see below) |
| `Loss_BinaryCrossentropy` | `(n_samples, n_outputs)`, values in `{0, 1}` | `Activation_Sigmoid` |
| `Loss_MeanSquaredError` | `(n_samples, n_outputs)`, continuous | `Activation_Linear` |
| `Loss_MeanAbsoluteError` | `(n_samples, n_outputs)`, continuous | `Activation_Linear` |

#### `Loss_CategoricalCrossentropy` — multi-class classification

**Technical:** for each sample, takes the negative log of the predicted
probability assigned to the correct class: `loss = -log(y_pred[correct_class])`.
Predictions are clipped away from exactly 0 or 1 first, to avoid
`log(0)`.

**Functional role:** heavily penalizes confident-but-wrong predictions
(predicting 1% probability for the actual correct class costs far more
than predicting 40%), which is exactly the behavior you want when
training a classifier to be both correct and well-calibrated. Always
pair with `Activation_Softmax` as the final activation -- the loss
assumes its input is already a valid probability distribution per row.

#### `Activation_Softmax_Loss_CategoricalCrossentropy` — internal fast path

**Technical:** mathematically, the combined derivative of Softmax
followed by Categorical Cross-Entropy simplifies to just
`predicted_probabilities - true_labels`, far cheaper than computing
Softmax's full Jacobian matrix and then chaining it with the loss's own
gradient.

**Functional role:** you never instantiate or call this class yourself.
`Model.finalize()` detects, automatically, when the last layer is
`Activation_Softmax` and the configured loss is
`Loss_CategoricalCrossentropy`, and swaps in this class internally for
the backward pass only (forward pass and reported loss value are
unaffected). It exists purely as a performance/stability optimization for
the single most common classification setup.

#### `Loss_BinaryCrossentropy` — binary / multi-label classification

**Technical:** for each output unit independently, `loss =
-(y_true*log(y_pred) + (1-y_true)*log(1-y_pred))`, then averaged across
the output units of each sample.

**Functional role:** the binary analogue of categorical cross-entropy,
applied independently per output -- which is precisely why it also
covers multi-label problems for free (several independent yes/no
decisions per sample) and not just single-output binary classification.
Always pair with `Activation_Sigmoid`.

#### `Loss_MeanSquaredError` — regression, penalizes big misses hard

**Technical:** `loss = mean((y_true - y_pred)^2)` across each sample's
outputs.

**Functional role:** squaring the error means a prediction that's off by
10 contributes 100x more loss than one off by 1 -- large errors are
punished disproportionately. This is the standard default for regression
and is what every regression `test_*.py` example uses, but it also makes
training somewhat sensitive to outliers in the target values.

#### `Loss_MeanAbsoluteError` — regression, robust alternative

**Technical:** `loss = mean(|y_true - y_pred|)` across each sample's
outputs.

**Functional role:** penalizes errors linearly instead of quadratically,
so a handful of outlier targets won't dominate the loss and skew
training the way they would with MSE. Reach for this instead of
`Loss_MeanSquaredError` when your regression targets are known to
contain outliers or heavy-tailed noise you don't want the model to chase.

#### Shared base behavior (`Loss`)

All loss classes inherit from `Loss`, which handles the parts that don't
depend on the specific formula:
- **`calculate(output, y, include_regularization=False)`**: calls the
  subclass's `forward()` to get a per-sample loss vector, then averages
  it into the single scalar `data_loss` reported in training logs.
- **`regularization_loss()`**: sums the L1/L2 penalty terms configured on
  every `Layer_Dense` (see Stage 1), added to `data_loss` to form the
  total training loss. Only computed during training
  (`include_regularization=True`), never during validation -- it is not
  a measure of fit quality, only a training-time penalty.

#### Where this fits in the pipeline

Configured once, in `model.set(loss=...)`, alongside the optimizer and
accuracy -- never part of the `model.add(...)` chain. The choice is
dictated entirely by the final activation you picked in Stage 2: Softmax
pairs with `Loss_CategoricalCrossentropy`, Sigmoid with
`Loss_BinaryCrossentropy`, Linear with either MSE or MAE.

### Stage 5 — Accuracy (`nn_lib.accuracy`)

Measure prediction quality in a way the user can read directly (e.g. %
correct classifications). **These metrics do not affect training**: no
gradient flows from them, they exist purely for reporting, during both
training and validation. `Model.train()` calls `accuracy.init(y)` once
before the epoch loop, then `accuracy.calculate(predictions, y)` every
epoch (and again on validation data).

| Class | Task | Setup needed before use |
|---|---|---|
| `Accuracy_Categorical` | Classification (binary or multi-class) | None |
| `Accuracy_Regression` | Regression | Automatic, derives a tolerance from `y`'s spread |

#### `Accuracy_Categorical` — classification

**Technical:** compares the class index predicted by the output
activation's `predictions()` method against `y`, element-wise; if `y` is
one-hot encoded (2D) and `binary=False`, it is first converted to sparse
class indices via `argmax` so the comparison is apples-to-apples. The
mean of the resulting boolean array is the reported accuracy.

**Functional role:** for classification, "correct" has an unambiguous,
binary meaning (predicted class either matches the true class or it
doesn't), so this is a direct, exact measure of quality -- no heuristic
threshold is involved, unlike the regression case below.

**Constructor:** `Accuracy_Categorical(binary=False)`. Set `binary=True`
whenever you paired the model with `Activation_Sigmoid` (binary or
multi-label classification, Stage 2); leave the default `False` for
`Activation_Softmax`-based multi-class classification. This flag exists
because sigmoid-based labels are already 0/1 and should never be
`argmax`-collapsed the way one-hot Softmax labels are.

#### `Accuracy_Regression` — regression

**Technical:** `compare()` uses a tolerance threshold,
`precision = std(y_train) / 250`, computed once from the training
targets: a prediction counts as "correct" if
`abs(prediction - true_value) < precision`. Additionally,
`r2_score(predictions, y)` computes the standard coefficient of
determination, `1 - sum((y-pred)^2) / sum((y-mean(y))^2)`.

**Functional role and how to read it:** regression has no natural notion
of an exact match, so `compare()`'s tolerance-based accuracy is a
heuristic, scaled to the specific dataset's variance -- useful mainly to
watch relative progress *within* one training run, not to judge a result
in isolation or compare across different datasets/runs. For that,
`r2_score` is what you want:
- `R² = 1` → perfect predictions
- `R² = 0` → the model is no better than always predicting the mean of `y`
- `R² < 0` → the model is worse than that trivial baseline

`Model.train()` detects `r2_score` automatically (via `hasattr`) and
appends `r2: ...` to both the periodic training summary and the final
validation summary whenever the configured accuracy object is an
`Accuracy_Regression` instance -- no extra setup required.

**Constructor:** `Accuracy_Regression()`, no arguments. **When evaluating
a regression pipeline, treat R² (and the loss value) as the metric to
trust; the tolerance-based accuracy is a secondary, epoch-to-epoch
progress indicator.**

#### Where this fits in the pipeline

Configured once, in `model.set(accuracy=...)`, alongside the loss and
optimizer -- like them, it is never part of the `model.add(...)` chain.
The choice follows directly from the task/final-activation pairing:
`Accuracy_Categorical()` for Softmax, `Accuracy_Categorical(binary=True)`
for Sigmoid, `Accuracy_Regression()` for Linear.

### Stage 6 — Model (`nn_lib.model`)

The `Model` class ties every stage above together and drives the entire
training lifecycle.

#### Usage flow

```python
model = Model()

# 1. Build the architecture: add layers/activations in the order data
#    must flow through them.
model.add(Layer_Dense(n_input_features, 64))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))          # optional
model.add(Layer_Dense(64, n_output))
model.add(Activation_Softmax())         # or Sigmoid / Linear, depending on the task

# 2. Configure the training pipeline
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(learning_rate=0.01, decay=1e-4),
    accuracy=Accuracy_Categorical()
)

# 3. Finalize: links the layers into a chain and identifies the trainable
#    parameters. Call this once, after add()/set() and before train().
model.finalize()

# 4. Train
model.train(
    X_train, y_train,
    epochs=1000,
    print_every=100,
    validation_data=(X_val, y_val)   # optional
)
```

#### `finalize()` — why it's a separate, required step

**Technical:** `add()` only appends objects to a plain Python list
(`self.layers`); nothing about how they connect exists yet. `finalize()`
walks that list once and gives every layer two new attributes, `.prev`
and `.next`, pointing at its neighbors (with a dummy `Layer_Input` inserted
before the first layer, and the `loss` object referenced as the `.next`
of the last one). It also scans for every layer exposing a `weights`
attribute and collects them into `self.trainable_layers`, and checks
whether the last activation/loss pair is Softmax+CategoricalCrossentropy
to enable the fast combined backward path described in Stage 4.

**Functional role:** this is what lets `forward()` and `backward()` be
written as simple, generic loops (`for layer in self.layers: ...`)
instead of code that special-cases the first/last layer or hard-codes
which layers are trainable. It also means the trainable-layers list and
the fast-path detection are computed once, rather than being
recalculated on every epoch.

**Practical implication:** always call it exactly once, after every
`add()` and after `set()`, and before `train()`. Adding a layer *after*
calling `finalize()` will leave the new layer disconnected from the
chain (no `.prev`/`.next`, not tracked as trainable) -- if you need a
different architecture, build a new `Model` instead of mutating one
that's already been finalized.

#### What `forward()` and `backward()` actually do

- **`forward(X, training)`**: seeds the chain by handing `X` to the
  dummy `Layer_Input`, then walks `self.layers` in order, each layer
  reading `layer.prev.output` and producing its own `.output` -- so data
  cascades through the whole network in a single pass. Returns the last
  layer's `.output`, i.e. the model's raw predictions.
- **`backward(output, y)`**: mirrors `forward()` in reverse. If the fast
  Softmax+CrossEntropy path applies (Stage 4), it computes that combined
  gradient directly and walks every layer *except* the last one in
  reverse; otherwise it starts from `loss.backward(output, y)` and walks
  every layer in reverse, each one reading `layer.next.dinputs` and
  producing its own `.dinputs` -- the chain rule, applied one layer at a
  time, from output back to input.

#### What `train()` does, epoch by epoch

`train(X, y, epochs=1, print_every=1, validation_data=None)` first calls
`accuracy.init(y)` once (a no-op for classification, computes the
tolerance threshold for regression -- see Stage 5), then repeats, once
per epoch:
1. **forward pass** on the training data, with `training=True` (so
   `Layer_Dropout` is active, if present);
2. **loss computation**: `loss.calculate(..., include_regularization=True)`
   returns both the data loss and the L1/L2 regularization loss (Stage
   4), summed into the epoch's total loss;
3. **accuracy computation**: the output activation's `predictions()`
   method (Stage 2) converts raw output into interpretable predictions,
   then `accuracy.calculate()` (Stage 5) scores them;
4. **backward pass**: `self.backward(output, y)`, computing every
   gradient in the network;
5. **parameter update**: `optimizer.pre_update_params()`, then
   `optimizer.update_params(layer)` for every layer in
   `self.trainable_layers`, then `optimizer.post_update_params()` (Stage
   3);
6. **logging**: every `print_every` epochs, prints accuracy, total/data/
   regularization loss, current learning rate, and (for regression) R².

If `validation_data=(X_val, y_val)` is given, after the epoch loop ends
the model runs one more forward pass with `training=False` (disabling
dropout, so the *complete* network is evaluated), computes loss and
accuracy on it, and prints a `validation, ...` summary -- with no
backward pass and no parameter updates, since validation must never
influence training.

## Composing a pipeline, layer by layer

This section is the practical complement to Stages 1-2 above: concrete
rules and worked examples for assembling `model.add(...)` calls for a
given task.

### General ordering rules

1. **Alternate `Layer_Dense` and an activation, always.** Never add two
   `Layer_Dense` back to back, and never leave a `Layer_Dense` without a
   following activation -- even the output layer needs one (Softmax /
   Sigmoid / Linear).
2. **Every hidden activation is `Activation_ReLU`.** The only activation
   choice you make per-architecture is which one to put at the very end
   (Stage 2 table above); everything before that is ReLU.
3. **Shapes must chain exactly.** The `n_neurons` of one `Layer_Dense`
   must equal the `n_inputs` of the next one. The first `Layer_Dense`'s
   `n_inputs` equals your feature count (`X.shape[1]`); the last
   `Layer_Dense`'s `n_neurons` equals your number of output values
   (classes, or regression targets).
4. **`Layer_Dropout` is optional and only goes after a hidden
   activation**, i.e. `Dense -> ReLU -> Dropout -> Dense -> ...`. Add it
   when you see overfitting (training accuracy/loss much better than
   validation); skip it on small/simple networks where it would just slow
   down learning.
5. **Regularization (`weight_regularizer_l2`, etc.) is set on
   `Layer_Dense` itself**, at construction time, not as a separate
   pipeline component. It is another overfitting countermeasure, usually
   applied to the first (largest) `Layer_Dense`.

### Recipe: multi-class classification

Use when each sample belongs to exactly one of several classes (e.g.
species of flower, digit 0-9).

```python
model.add(Layer_Dense(n_features, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, n_classes))
model.add(Activation_Softmax())

model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(learning_rate=0.01, decay=1e-4),
    accuracy=Accuracy_Categorical(),
)
```

`y` can be either sparse integer labels (shape `(n_samples,)`, values
`0..n_classes-1`) or one-hot encoded (shape `(n_samples, n_classes)`) --
both `Loss_CategoricalCrossentropy` and `Accuracy_Categorical` handle
either format automatically. See `test_iris_classification.py`.

### Recipe: binary classification

Use when each sample is one of exactly two classes (e.g. spam/not spam).

```python
model.add(Layer_Dense(n_features, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 1))            # 1 output neuron, not 2
model.add(Activation_Sigmoid())

model.set(
    loss=Loss_BinaryCrossentropy(),
    optimizer=Optimizer_SGD(learning_rate=1.0, decay=1e-3, momentum=0.9),
    accuracy=Accuracy_Categorical(binary=True),
)
```

`y` must be shaped `(n_samples, 1)` with values in `{0, 1}` (reshape a
flat label vector with `y.reshape(-1, 1)` if needed). See
`test_moons_classification.py`.

### Recipe: multi-label classification

Use when each sample can belong to several non-exclusive classes at once
(e.g. tagging an image with multiple attributes). Structurally identical
to binary classification, just with more output neurons:

```python
model.add(Layer_Dense(n_features, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, n_labels))     # one neuron per independent label
model.add(Activation_Sigmoid())

model.set(
    loss=Loss_BinaryCrossentropy(),
    optimizer=Optimizer_Adam(learning_rate=0.001),
    accuracy=Accuracy_Categorical(binary=True),
)
```

`y` must be shaped `(n_samples, n_labels)`, each column independently
`{0, 1}`.

### Recipe: regression

Use when the target is a continuous value (e.g. a price, a temperature,
a coordinate).

```python
model.add(Layer_Dense(n_features, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, n_outputs))    # n_outputs = 1 for a single target
model.add(Activation_Linear())

model.set(
    loss=Loss_MeanSquaredError(),        # or Loss_MeanAbsoluteError()
    optimizer=Optimizer_Adam(learning_rate=0.01, decay=1e-3),
    accuracy=Accuracy_Regression(),
)
```

`y` must be shaped `(n_samples, n_outputs)` (reshape with
`y.reshape(-1, 1)` for a single target). Regression targets that are
non-linear functions of the input (curves, waves) usually need more
hidden capacity (two hidden layers, as above) than a comparably-sized
classification problem. Judge the result using `r2_score` (see Stage 5),
not the tolerance-based accuracy alone. See `test_regression_sine.py`.

### Recipe: classification with overfitting control

Same shape as the multi-class recipe, with L2 regularization on the
first `Layer_Dense` and a `Layer_Dropout` added between the hidden
activation and the output block -- use this pattern once you observe the
training metrics pulling noticeably ahead of the validation metrics:

```python
model.add(Layer_Dense(
    n_features, 128,
    weight_regularizer_l2=5e-4,
    bias_regularizer_l2=5e-4,
))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.2))
model.add(Layer_Dense(128, n_classes))
model.add(Activation_Softmax())
```

See `test_blobs_classification.py`, which combines this architecture
with an intentionally noisy dataset to demonstrate the effect.

## Choosing components for your task

| Problem type | Final activation | Recommended loss | Accuracy |
|---|---|---|---|
| Multi-class classification (mutually exclusive classes) | `Activation_Softmax` | `Loss_CategoricalCrossentropy` | `Accuracy_Categorical()` |
| Binary / multi-label classification | `Activation_Sigmoid` | `Loss_BinaryCrossentropy` | `Accuracy_Categorical(binary=True)` |
| Regression | `Activation_Linear` | `Loss_MeanSquaredError` or `Loss_MeanAbsoluteError` | `Accuracy_Regression()` |

For the optimizer, `Optimizer_Adam` is generally a good default choice;
`Optimizer_SGD` with momentum is simpler to reason about and useful for
learning purposes; `Optimizer_Adagrad`/`Optimizer_RMSprop` are
intermediate alternatives.

## Test pipelines (`test_*.py`)

Four ready-to-run scripts at the project root, each exercising a
different task/dataset combination:

| Script | Dataset | Task | Architecture highlights |
|---|---|---|---|
| `test_iris_classification.py` | Iris (scikit-learn, bundled, no download) | Multi-class classification | Simple Dense -> ReLU -> Dense -> Softmax |
| `test_moons_classification.py` | Two moons (scikit-learn, synthetic) | Binary classification | Sigmoid output, SGD with momentum |
| `test_regression_sine.py` | Synthetic noisy sine wave (NumPy) | Regression | Two hidden layers, MSE loss |
| `test_blobs_classification.py` | Blobs with injected label noise (scikit-learn, synthetic) | Multi-class classification | L2 regularization + Dropout |

Run any of them directly from the project root, e.g.:

```bash
python test_iris_classification.py
```

Each script prints a periodic training summary (accuracy, total loss,
data loss, regularization loss, current learning rate) and a final
validation summary.

## Known limitations

- No mini-batching: `train()` runs the forward/backward pass over the
  entire dataset on every epoch (fine for small/educational datasets, does
  not scale to large ones).
- No model saving/loading (parameter serialization).
- No convolutional, recurrent, or attention layers: the library only
  covers fully-connected feed-forward networks.
- The optimized Softmax+CrossEntropy combination is only enabled when the
  final activation is exactly `Activation_Softmax` and the loss is
  exactly `Loss_CategoricalCrossentropy`; other combinations use the
  "generic" (slower, but always correct) backward-pass path.

## Dependencies

- `numpy` — required by the library itself.
- `scikit-learn` — used only by the `test_*.py` example scripts, to load
  or generate example datasets (`load_iris`, `make_moons`, `make_blobs`,
  `train_test_split`, `StandardScaler`). Not required to use `nn_lib`
  itself.
