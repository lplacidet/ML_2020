# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    e = y - tx.dot(w)
    gradient = - (1/len(e)) * tx.T.dot(e)
    return gradient, e


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y,tx,batch_size = batch_size, num_batches = 1):
            gradient , e = compute_stoch_gradient(y_batch,tx_batch, w)
            w = w - gamma*gradient
            loss = compute_loss(y,tx,w)
            
            ws.append(w)
            losses.append(loss)

    return losses, ws