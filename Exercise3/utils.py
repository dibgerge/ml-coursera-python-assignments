import sys
import numpy as np
from matplotlib import pyplot

sys.path.append('..')
from submission import SubmissionBase


def displayData(X, example_width=None, figsize=(10, 10)):
    """
    Displays 2D data stored in X in a nice grid.
    """
    # Compute rows, cols
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None]  # Promote to a 2 dimensional array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')

    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = n / example_width

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    fig, ax_array = pyplot.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_width, example_width, order='F'),
                  cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')


def sigmoid(z):
    """
    Computes the sigmoid of z.
    """
    return 1.0 / (1.0 + np.exp(-z))


class Grader(SubmissionBase):
    # Random Test Cases
    X = np.stack([np.ones(20),
                  np.exp(1) * np.sin(np.arange(1, 21)),
                  np.exp(0.5) * np.cos(np.arange(1, 21))], axis=1)

    y = (np.sin(X[:, 0] + X[:, 1]) > 0).astype(float)

    Xm = np.array([[-1, -1],
                   [-1, -2],
                   [-2, -1],
                   [-2, -2],
                   [1, 1],
                   [1, 2],
                   [2, 1],
                   [2, 2],
                   [-1, 1],
                   [-1, 2],
                   [-2, 1],
                   [-2, 2],
                   [1, -1],
                   [1, -2],
                   [-2, -1],
                   [-2, -2]])
    ym = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])

    t1 = np.sin(np.reshape(np.arange(1, 25, 2), (4, 3), order='F'))
    t2 = np.cos(np.reshape(np.arange(1, 41, 2), (4, 5), order='F'))

    def __init__(self):
        part_names = ['Regularized Logistic Regression',
                      'One-vs-All Classifier Training',
                      'One-vs-All Classifier Prediction',
                      'Neural Network Prediction Function']

        super().__init__('multi-class-classification-and-neural-networks', part_names)

    def __iter__(self):
        for part_id in range(1, 5):
            try:
                func = self.functions[part_id]

                # Each part has different expected arguments/different function
                if part_id == 1:
                    res = func(np.array([0.25, 0.5, -0.5]), self.X, self.y, 0.1)
                    res = np.hstack(res).tolist()
                elif part_id == 2:
                    res = func(self.Xm, self.ym, 4, 0.1)
                elif part_id == 3:
                    res = func(self.t1, self.Xm) + 1
                elif part_id == 4:
                    res = func(self.t1, self.t2, self.Xm) + 1
                else:
                    raise KeyError
                yield part_id, res
            except KeyError:
                yield part_id, 0
