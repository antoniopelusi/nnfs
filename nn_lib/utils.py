"""
Small standalone utilities for nn_lib.

STAGE: setup (called once, before building or training any model), used to
make experiments reproducible without depending on any third-party
dataset/init package.
"""

import numpy as np


def init(seed=0):
    """
    Initialize the library's random state for reproducible experiments.

    STAGE: setup (call this once, at the very start of your script, before
    creating any layer -- weight initialization relies on NumPy's global
    random state).

    This replaces the old `nnfs.init()` call: it only fixes the NumPy
    random seed, so results are reproducible across runs. Unlike `nnfs`,
    it does not force a global float32 dtype -- the library works fine
    with NumPy's default float64.

    Parameters:
        seed (int): seed passed to `numpy.random.seed`.
    """
    np.random.seed(seed)
