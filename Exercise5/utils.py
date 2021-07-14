import sys
import numpy as np
from scipy import optimize
from matplotlib import pyplot

sys.path.append('..')
from submission import SubmissionBase


def trainLinearReg(linearRegCostFunction, X, y, lambda_=0.0, maxiter=200):
    """
    Trains linear regression using scipy's optimize.minimize.

    Parameters
    ----------
    X : array_like
        The dataset with shape (m x n+1). The bias term is assumed to be concatenated.

    y : array_like
        Function values at each datapoint. A vector of shape (m,).

    lambda_ : float, optional
        The regularization parameter.

    maxiter : int, optional
        Maximum number of iteration for the optimization algorithm.

    Returns
    -------
    theta : array_like
        The parameters for linear regression. This is a vector of shape (n+1,).
    """
    # Initialize Theta
    initial_theta = np.zeros(X.shape[1])

    # Create "short hand" for the cost function to be minimized
    costFunction = lambda t: linearRegCostFunction(X, y, t, lambda_)

    # Now, costFunction is a function that takes in only one argument
    options = {'maxiter': maxiter}

    # Minimize using scipy
    res = optimize.minimize(costFunction, initial_theta, jac=True, method='TNC', options=options)
    return res.x


def featureNormalize(X):
    """
    Normalizes the features in X returns a normalized version of X where the mean value of each
    feature is 0 and the standard deviation is 1. This is often a good preprocessing step to do when
    working with learning algorithms.

    Parameters
    ----------
    X : array_like
        An dataset which is a (m x n) matrix, where m is the number of examples,
        and n is the number of dimensions for each example.

    Returns
    -------
    X_norm : array_like
        The normalized input dataset.

    mu : array_like
        A vector of size n corresponding to the mean for each dimension across all examples.

    sigma : array_like
        A vector of size n corresponding to the standard deviations for each dimension across
        all examples.
    """
    mu = np.mean(X, axis=0)
    X_norm = X - mu

    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm /= sigma
    return X_norm, mu, sigma


def plotFit(polyFeatures, min_x, max_x, mu, sigma, theta, p):
    """
    Plots a learned polynomial regression fit over an existing figure.
    Also works with linear regression.
    Plots the learned polynomial fit with power p and feature normalization (mu, sigma).

    Parameters
    ----------
    polyFeatures : func
        A function which generators polynomial features from a single feature.

    min_x : float
        The minimum value for the feature.

    max_x : float
        The maximum value for the feature.

    mu : float
        The mean feature value over the training dataset.

    sigma : float
        The feature standard deviation of the training dataset.

    theta : array_like
        The parameters for the trained polynomial linear regression.

    p : int
        The polynomial order.
    """
    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = np.arange(min_x - 15, max_x + 25, 0.05).reshape(-1, 1)

    # Map the X values
    X_poly = polyFeatures(x, p)
    X_poly -= mu
    X_poly /= sigma

    # Add ones
    X_poly = np.concatenate([np.ones((x.shape[0], 1)), X_poly], axis=1)

    # Plot
    pyplot.plot(x, np.dot(X_poly, theta), '--', lw=2)


class Grader(SubmissionBase):
    # Random test cases
    X = np.vstack([np.ones(10),
                   np.sin(np.arange(1, 15, 1.5)),
                   np.cos(np.arange(1, 15, 1.5))]).T
    y = np.sin(np.arange(1, 31, 3))
    Xval = np.vstack([np.ones(10),
                      np.sin(np.arange(0, 14, 1.5)),
                      np.cos(np.arange(0, 14, 1.5))]).T
    yval = np.sin(np.arange(1, 11))

    def __init__(self):
        part_names = ['Regularized Linear Regression Cost Function',
                      'Regularized Linear Regression Gradient',
                      'Learning Curve',
                      'Polynomial Feature Mapping',
                      'Validation Curve']
        part_names_key = ['a6bvf', 'x4FhA', 'n3zWY', 'lLaa4', 'gyJbG']
        assignment_key = '-wEfetVmQgG3j-mtasztYg'
        super().__init__('regularized-linear-regression-and-bias-variance', assignment_key, part_names, part_names_key)

    def __iter__(self):
        for part_id in range(1, 6):
            try:
                func = self.functions[part_id]
                # Each part has different expected arguments/different function
                if part_id == 1:
                    res = func(self.X, self.y, np.array([0.1, 0.2, 0.3]), 0.5)
                elif part_id == 2:
                    theta = np.array([0.1, 0.2, 0.3])
                    res = func(self.X, self.y, theta, 0.5)[1]
                elif part_id == 3:
                    res = np.hstack(func(self.X, self.y, self.Xval, self.yval, 1)).tolist()
                elif part_id == 4:
                    res = func(self.X[1, :].reshape(-1, 1), 8)
                elif part_id == 5:
                    res = np.hstack(func(self.X, self.y, self.Xval, self.yval)).tolist()
                else:
                    raise KeyError
            except KeyError:
                yield part_id, 0
            yield part_id, res

