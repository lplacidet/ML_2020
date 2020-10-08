# -*- coding: utf-8 -*-
"""Splitting Data"""


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)

    split_index = int(len(x)*ratio)
    
    indices = np.random.permutation(len(x))
    indices_train = indices[:split_index]
    indices_test = indices[split_index:]
    
    train_x = x[indices_train]
    train_y = y[indices_train]
    test_x = x[indices_test]
    test_y = y[indices_test]
    
    return train_x, train_y, test_x, test_y