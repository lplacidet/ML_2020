import numpy as np
import matplotlib.pyplot as plt
from costs import*
from gradient_descent import*
from plots import gradient_descent_visualization
from logistic_regression import*
from helpers import*


#Loading Dataset
data_path = "E:\EPFL-Masters\Machine Learning\Data\Train.csv"
data = np.genfromtxt(data_path, delimiter=',', dtype='str', skip_header=1, usecols=[])
x = data[:, 2:].astype(np.float)
y = data[:, 1]
y_bin = np.zeros((y.shape[0], 1))
id = data[:, 0].astype(np.float)
y = np.where(y =='s', 0, y)
y = np.where(y =='b', 1, y).astype(np.float)

#Cleaning Data from outliers
# row_indicies = np.where(x == -999.0)[0]
# x_cleaned = np.delete(x, row_indicies, 0)
# y_cleaned = np.delete(y, row_indicies, 0)
# y_cleaned = np.reshape(y_cleaned, (y_cleaned.shape[0], 1))
# id_cleaned = np.delete(id, row_indicies, 0).T
# id_cleaned = np.reshape(id_cleaned, (id_cleaned.shape[0], 1))

#Replace each -999 to mean value
mean_cols = np.mean(x, axis=0)
indicies = np.where(x == -999.0)
x[indicies] = np.take(mean_cols, indicies[1])
x, mean, std = standardize(x)
y = np.reshape(y, (y.shape[0], 1))
id = np.reshape(id, (id.shape[0], 1)).T

#Adding Offset
tx = np.c_[np.ones(x.shape[0]), x]

#Apply Logistic Regression
##Define the parameters of the algorithm
max_iter = 100
threshold = 0.001
gamma = 0.01
lambda_ = 0.1
losses = []
w = np.zeros((tx.shape[1], 1))

##Start the logistic regression
for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_newton_method(y, tx, w, lambda_)
        # log info
        if iter % 1 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    # visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_newton_method",True)
print("loss={l}".format(l=calculate_loss(y, tx, w)))

# visualization
#visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_gradient_descent", True)
# print("loss={l}".format(l=calculate_loss(y, tx, w)))