import numpy as np
import sys
sys.path.append('..')

from submission import SubmissionBase


class Grader(SubmissionBase):
    X1 = np.column_stack((np.ones(20), np.exp(1) + np.exp(2) * np.linspace(0.1, 2, 20)))
    Y1 = X1[:, 1] + np.sin(X1[:, 0]) + np.cos(X1[:, 1])
    X2 = np.column_stack((X1, X1[:, 1]**0.5, X1[:, 1]**0.25))
    Y2 = np.power(Y1, 0.5) + Y1

    def __init__(self):
        part_names = ['Warm up exercise',
                      'Computing Cost (for one variable)',
                      'Gradient Descent (for one variable)',
                      'Feature Normalization',
                      'Computing Cost (for multiple variables)',
                      'Gradient Descent (for multiple variables)',
                      'Normal Equations']
        super().__init__('linear-regression', part_names)

    def __iter__(self):
        for part_id in range(1, 8):
            try:
                func = self.functions[part_id]

                # Each part has different expected arguments/different function
                if part_id == 1:
                    res = func()
                elif part_id == 2:
                    res = func(self.X1, self.Y1, np.array([0.5, -0.5]))
                elif part_id == 3:
                    res = func(self.X1, self.Y1, np.array([0.5, -0.5]), 0.01, 10)
                elif part_id == 4:
                    res = func(self.X2[:, 1:4])
                elif part_id == 5:
                    res = func(self.X2, self.Y2, np.array([0.1, 0.2, 0.3, 0.4]))
                elif part_id == 6:
                    res = func(self.X2, self.Y2, np.array([-0.1, -0.2, -0.3, -0.4]), 0.01, 10)
                elif part_id == 7:
                    res = func(self.X2, self.Y2)
                else:
                    raise KeyError
                yield part_id, res
            except KeyError:
                yield part_id, 0
