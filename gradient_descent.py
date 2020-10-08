# -*- coding: utf-8 -*-
"""Gradient Descent"""

def compute_gradient(y, tx, w):
    """Compute a gradient."""
    e = y - tx.dot(w)
    gradient = - (1/len(e)) * tx.T.dot(e)
    
    return gradient,e


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):

        gradient, e = compute_gradient(y,tx,w)
        loss = compute_loss(y, tx, w)
        
        w = w - gamma * gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)

    return losses, ws