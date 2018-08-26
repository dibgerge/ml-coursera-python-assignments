import numpy as np
import sys
from os.path import join
from matplotlib import pyplot

sys.path.append('..')
from submission import SubmissionBase


def normalizeRatings(Y, R):
    """
    Preprocess data by subtracting mean rating for every movie (every row).

    Parameters
    ----------
    Y : array_like
        The user ratings for all movies. A matrix of shape (num_movies x num_users).

    R : array_like
        Indicator matrix for movies rated by users. A matrix of shape (num_movies x num_users).

    Returns
    -------
    Ynorm : array_like
        A matrix of same shape as Y, after mean normalization.

    Ymean : array_like
        A vector of shape (num_movies, ) containing the mean rating for each movie.
    """
    m, n = Y.shape
    Ymean = np.zeros(m)
    Ynorm = np.zeros(Y.shape)

    for i in range(m):
        idx = R[i, :] == 1
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]

    return Ynorm, Ymean


def loadMovieList():
    """
    Reads the fixed movie list in movie_ids.txt and returns a list of movie names.

    Returns
    -------
    movieNames : list
        A list of strings, representing all movie names.
    """
    # Read the fixed movieulary list
    with open(join('Data', 'movie_ids.txt'),  encoding='ISO-8859-1') as fid:
        movies = fid.readlines()

    movieNames = []
    for movie in movies:
        parts = movie.split()
        movieNames.append(' '.join(parts[1:]).strip())
    return movieNames


def computeNumericalGradient(J, theta, e=1e-4):
    """
    Computes the gradient using "finite differences" and gives us a numerical estimate of the
    gradient.

    Parameters
    ----------
    J : func
        The cost function which will be used to estimate its numerical gradient.

    theta : array_like
        The one dimensional unrolled network parameters. The numerical gradient is computed at
         those given parameters.

    e : float (optional)
        The value to use for epsilon for computing the finite difference.

    Returns
    -------
    numgrad : array_like
        The numerical gradient with respect to theta. Has same shape as theta.

    Notes
    -----
    The following code implements numerical gradient checking, and
    returns the numerical gradient. It sets `numgrad[i]` to (a numerical
    approximation of) the partial derivative of J with respect to the
    i-th input argument, evaluated at theta. (i.e., `numgrad[i]` should
    be the (approximately) the partial derivative of J with respect
    to theta[i].)
    """
    numgrad = np.zeros(theta.shape)
    perturb = np.diag(e * np.ones(theta.shape))
    for i in range(theta.size):
        loss1, _ = J(theta - perturb[:, i])
        loss2, _ = J(theta + perturb[:, i])
        numgrad[i] = (loss2 - loss1)/(2*e)
    return numgrad


def checkCostFunction(cofiCostFunc, lambda_=0.):
    """
    Creates a collaborative filtering problem to check your cost function and gradients.
    It will output the  analytical gradients produced by your code and the numerical gradients
    (computed using computeNumericalGradient). These two gradient computations should result
    in very similar values.

    Parameters
    ----------
    cofiCostFunc: func
        Implementation of the cost function.

    lambda_ : float, optional
        The regularization parameter.
    """
    # Create small problem
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)

    # Zap out most entries
    Y = np.dot(X_t, Theta_t.T)
    Y[np.random.rand(*Y.shape) > 0.5] = 0
    R = np.zeros(Y.shape)
    R[Y != 0] = 1

    # Run Gradient Checking
    X = np.random.randn(*X_t.shape)
    Theta = np.random.randn(*Theta_t.shape)
    num_movies, num_users = Y.shape
    num_features = Theta_t.shape[1]

    params = np.concatenate([X.ravel(), Theta.ravel()])
    numgrad = computeNumericalGradient(
        lambda x: cofiCostFunc(x, Y, R, num_users, num_movies, num_features, lambda_), params)

    cost, grad = cofiCostFunc(params, Y, R, num_users,num_movies, num_features, lambda_)

    print(np.stack([numgrad, grad], axis=1))
    print('\nThe above two columns you get should be very similar.'
          '(Left-Your Numerical Gradient, Right-Analytical Gradient)')

    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print('If your cost function implementation is correct, then '
          'the relative difference will be small (less than 1e-9).')
    print('\nRelative Difference: %g' % diff)


def multivariateGaussian(X, mu, Sigma2):
    """
    Computes the probability density function of the multivariate gaussian distribution.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n). Where there are m examples of n-dimensions.

    mu : array_like
        A vector of shape (n,) contains the means for each dimension (feature).

    Sigma2 : array_like
        Either a vector of shape (n,) containing the variances of independent features
        (i.e. it is the diagonal of the correlation matrix), or the full
        correlation matrix of shape (n x n) which can represent dependent features.

    Returns
    ------
    p : array_like
        A vector of shape (m,) which contains the computed probabilities at each of the
        provided examples.
    """
    k = mu.size

    # if sigma is given as a diagonal, compute the matrix
    if Sigma2.ndim == 1:
        Sigma2 = np.diag(Sigma2)

    X = X - mu
    p = (2 * np.pi) ** (- k / 2) * np.linalg.det(Sigma2) ** (-0.5)\
        * np.exp(-0.5 * np.sum(np.dot(X, np.linalg.pinv(Sigma2)) * X, axis=1))
    return p


def visualizeFit(X, mu, sigma2):
    """
    Visualize the dataset and its estimated distribution.
    This visualization shows you the  probability density function of the Gaussian distribution.
    Each example has a location (x1, x2) that depends on its feature values.

    Parameters
    ----------
    X : array_like
        The dataset of shape (m x 2). Where there are m examples of 2-dimensions. We need at most
        2-D features to be able to visualize the distribution.

    mu : array_like
        A vector of shape (n,) contains the means for each dimension (feature).

    sigma2 : array_like
        Either a vector of shape (n,) containing the variances of independent features
        (i.e. it is the diagonal of the correlation matrix), or the full
        correlation matrix of shape (n x n) which can represent dependent features.
    """

    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
    Z = multivariateGaussian(np.stack([X1.ravel(), X2.ravel()], axis=1), mu, sigma2)
    Z = Z.reshape(X1.shape)

    pyplot.plot(X[:, 0], X[:, 1], 'bx', mec='b', mew=2, ms=8)

    if np.all(abs(Z) != np.inf):
        pyplot.contour(X1, X2, Z, levels=10**(np.arange(-20., 1, 3)), zorder=100)


class Grader(SubmissionBase):
    # Random Test Cases
    n_u = 3
    n_m = 4
    n = 5
    X = np.sin(np.arange(1, 1 + n_m * n)).reshape(n_m, n, order='F')
    Theta = np.cos(np.arange(1, 1 + n_u * n)).reshape(n_u, n, order='F')
    Y = np.sin(np.arange(1, 1 + 2 * n_m * n_u, 2)).reshape(n_m, n_u, order='F')
    R = Y > 0.5
    pval = np.concatenate([abs(Y.ravel('F')),  [0.001],  [1]])
    Y = Y * R  # set 'Y' values to 0 for movies not reviewed

    yval = np.concatenate([R.ravel('F'), [1], [0]])
    #
    params = np.concatenate([X.ravel(), Theta.ravel()])

    def __init__(self):
        part_names = ['Estimate Gaussian Parameters',
                      'Select Threshold',
                      'Collaborative Filtering Cost',
                      'Collaborative Filtering Gradient',
                      'Regularized Cost',
                      'Regularized Gradient']
        super().__init__('anomaly-detection-and-recommender-systems', part_names)

    def __iter__(self):
        for part_id in range(1, 7):
            try:
                func = self.functions[part_id]

                # Each part has different expected arguments/different function
                if part_id == 1:
                    res = np.hstack(func(self.X)).tolist()
                elif part_id == 2:
                    res = np.hstack(func(self.yval, self.pval)).tolist()
                elif part_id == 3:
                    J, grad = func(self.params, self.Y, self.R, self.n_u, self.n_m, self.n)
                    res = J
                elif part_id == 4:
                    J, grad = func(self.params, self.Y, self.R, self.n_u, self.n_m, self.n, 0)
                    xgrad = grad[:self.n_m*self.n].reshape(self.n_m, self.n)
                    thetagrad = grad[self.n_m*self.n:].reshape(self.n_u, self.n)
                    res = np.hstack([xgrad.ravel('F'), thetagrad.ravel('F')]).tolist()
                elif part_id == 5:
                    res, _ = func(self.params, self.Y, self.R, self.n_u, self.n_m, self.n, 1.5)
                elif part_id == 6:
                    J, grad = func(self.params, self.Y, self.R, self.n_u, self.n_m, self.n, 1.5)
                    xgrad = grad[:self.n_m*self.n].reshape(self.n_m, self.n)
                    thetagrad = grad[self.n_m*self.n:].reshape(self.n_u, self.n)
                    res = np.hstack([xgrad.ravel('F'), thetagrad.ravel('F')]).tolist()
                else:
                    raise KeyError
                yield part_id, res
            except KeyError:
                yield part_id, 0
