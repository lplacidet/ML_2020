# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss_mse(y, tx, w):
    """Calculate the loss using mse.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return 1/2 * np.mean(e**2)

def compute_loss_mae(y, tx, w):
    e = y - tx.dot(w)
    return np.mean(np.abs(e))