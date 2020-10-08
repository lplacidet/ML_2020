# -*- coding: utf-8 -*-
"""Ridge Regression"""

def compute_loss_mse(y, tx, w):
    e = y - tx.dot(w)
    return 1/2 * np.mean(e**2)

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""

    lambdaI = 2* tx.shape[0] * lambda_ * np.eye(tx.shape[1])
    
    w_ridge = np.linalg.solve(tx.T.dot(tx)+lambdaI,tx.T.dot(y))
    
    rmse = np.sqrt(2*compute_loss_mse(y, tx, w_ridge))
        
    return rmse, w_ridge