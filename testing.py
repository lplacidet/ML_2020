import numpy as np
import matplotlib.pyplot as plt
from costs import*
from gradient_descent import*


#Loading Dataset
data_path = "E:\EPFL-Masters\Machine Learning\Data\Train.csv"
data = np.genfromtxt(data_path, delimiter=',', dtype='str', skip_header=1, usecols=[])
x = data[:, 2:].astype(np.float)
y = data[:, 1]
id = data[:, 0].astype(np.float)

#Adding Offset
tx = np.c_[np.ones(x.shape[0]), x]

#Apply Gradient descent
##Define the parameters of the algorithm.
max_iters = 200
gamma = 0.7
##Initialization
w_initial = np.array([0, 0])


