# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""
import numpy as np
from costs import*


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    fw = tx@w.T
    e = y - fw
    gradient = -(tx.T@e)/len(y)
    return gradient


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size, num_batches=1):
            grad = compute_stoch_gradient(y_batch, tx_batch, w, max_iters)
            w = w - gamma*grad
            loss = compute_MSE_loss(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)
            print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws
