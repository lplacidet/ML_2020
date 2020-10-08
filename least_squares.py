# -*- coding: utf-8 -*-
"""Least Squares using normal equations"""

def least_squares(y, tx):
    """calculate the least squares solution."""
    
    opt_weights = np.linalg.solve(tx.T@tx,tx.T@y)
    
    mse = compute_loss_mse(y, tx, opt_weights)
    
    return mse, opt_weights