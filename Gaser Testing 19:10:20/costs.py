# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np

def compute_MSE_loss(y, tx, w):
    fw = tx@w.T
    e = y - fw
    cost = e**2
    loss = np.mean(cost)/2
    return loss


def compute_MAE_loss(y, tx, w):
    f = tx@w.T
    cost = np.abs(y-f)
    loss = np.mean(cost)/2
    return loss
