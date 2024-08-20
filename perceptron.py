import numpy as np
"""Perceptron classifier

Parameters:
------------
eta: float
    Learning rate (between 0 and 1)
n_iter: int
    Number of passes over the training dataset.
random_state: int
    Random number for weight initialization

Attributes:
------------
w_ : 1d-array
    Weights after fitting
b_ : Scalar
    Bias unit after fitting (threshold)

errors_ : list
    Number of misclassifications (updates) in each epoch.

"""

