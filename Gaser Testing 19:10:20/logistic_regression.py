import numpy as np
import matplotlib.pyplot as plt
#from helpers import de_standardize, standardize


def sigmoid(t):
    """apply the sigmoid function on t."""
    return 1/(1+np.exp(-t))


def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    t1 = y*(tx@w)
    t2 = np.log(1+np.exp(tx@w))
    return np.sum(t2 - t1)
    # a = sigmoid(tx.dot(w))
    # loss = y.T.dot(np.log(a)) + (1 - y).T.dot(1 - np.log(a))
    # return -loss


def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T@(sigmoid(tx@w)-y)


def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function."""
    S = np.diag(np.diag(np.floor(sigmoid(tx@w)@(1-sigmoid(tx@w)).T)))
    return tx.T@S@tx


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    w = w - gamma*gradient
    return loss, w


def logistic_regression(y, tx, w):
    """return the loss, gradient, and Hessian."""
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    hesh = calculate_hessian(y, tx, w)
    return loss, gradient, hesh


def learning_by_newton_method(y, tx, w, lambda_):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    loss, gradient, hesh = logistic_regression(y, tx, w)
    w = w - lambda_*(np.linalg.solve(hesh, gradient))
    return loss, w


def visualization(y, x, mean_x, std_x, w, save_name, is_LR=False):
    """visualize the raw data as well as the classification result."""
    fig = plt.figure()
    # plot raw data
    x = de_standardize(x, mean_x, std_x)
    ax1 = fig.add_subplot(1, 2, 1)
    males = np.where(y == 1)
    females = np.where(y == 0)
    ax1.scatter(
        x[females, 0], x[females, 1],
        marker='*', color=[1, 0.06, 0.06], s=20, label="female sample")
    ax1.scatter(
        x[males, 0], x[males, 1],
        marker='.', color=[0.06, 0.06, 1], s=20, label="male sample")
    ax1.set_xlabel("Height")
    ax1.set_ylabel("Weight")
    ax1.legend()
    ax1.grid()
    # plot raw data with decision boundary
    ax2 = fig.add_subplot(1, 2, 2)
    height = np.arange(
        np.min(x[:, 0]), np.max(x[:, 0]) + 0.01, step=0.01)
    weight = np.arange(
        np.min(x[:, 1]), np.max(x[:, 1]) + 1, step=1)
    hx, hy = np.meshgrid(height, weight)
    hxy = (np.c_[hx.reshape(-1), hy.reshape(-1)] - mean_x) / std_x
    x_temp = np.c_[np.ones((hxy.shape[0], 1)), hxy]
    # The threshold should be different for least squares and logistic regression when label is {0,1}.
    # least square: decision boundary t >< 0.5
    # logistic regression:  decision boundary sigmoid(t) >< 0.5  <==> t >< 0
    if is_LR:
        prediction = x_temp.dot(w) > 0.0
    else:
        prediction = x_temp.dot(w) > 0.5
    prediction = prediction.reshape((weight.shape[0], height.shape[0]))    
    cs = ax2.contourf(hx, hy, prediction, 1)
    proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) 
    for pc in cs.collections]
    ax2.legend(proxy, ["prediction female", "prediction male"])
    ax2.scatter(
        x[males, 0], x[males, 1],
        marker='.', color=[0.06, 0.06, 1], s=20)
    ax2.scatter(
        x[females, 0], x[females, 1],
        marker='*', color=[1, 0.06, 0.06], s=20)
    ax2.set_xlabel("Height")
    ax2.set_ylabel("Weight")
    ax2.set_xlim([min(x[:, 0]), max(x[:, 0])])
    ax2.set_ylim([min(x[:, 1]), max(x[:, 1])])
    plt.tight_layout()
    plt.savefig(save_name+".pdf")

