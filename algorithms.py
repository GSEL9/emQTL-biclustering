# -*- coding: utf-8 -*-
#
# cluster.py
#

"""
Wrappers for R biclustering algorithms.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


import numpy as np
import rpy2.robjects as robjects

from base import RBiclusterBase


class ChengChurch(RBiclusterBase):
    """A wrapper for the R BCCC algorithm.

    Constant values

    """

    MODEL = 'BCCC'

    # Hyperparameters
    params = {
        'delta': 0.1,
        'alpha': 1.5,
        'number': 10
    }

    def __init__(self, random_state=0, **kwargs):

        super().__init__(random_state, **kwargs)

        # Update parameters.
        for key in kwargs:
            if key in self.params.keys():
                self.params[key] = kwargs[key]

        self.set_params(**kwargs)

        # NOTE:
        self._output = None

        self.rows_ = None
        self.columns_ = None
        self.biclusters_ = None

    def fit(self, X, y=None, **kwargs):

        self._fit(self.MODEL, X, self.params)

        return self


class XMotifs(RBiclusterBase):
    """A wrapper for the R BCXmotifs algorithm.

    Coherent  correlation  over  rows and  columns

    """

    MODEL = 'BCXmotifs'

    # Hyperparameters
    params = {
        'number': 10,
        'ns': 200,
        'nd': 100,
        'sd': 5,
        'alpha': 0.05
    }

    def __init__(self, random_state=0, **kwargs):

        super().__init__(random_state, **kwargs)

        # Update parameters.
        for key in kwargs:
            if key in self.params.keys():
                self.params[key] = kwargs[key]

        self.set_params(**kwargs)

        # NOTE:
        self._output = None

        self.rows_ = None
        self.columns_ = None
        self.biclusters_ = None

    def fit(self, X, y=None, **kwargs):

        self._fit(self.MODEL, X, self.params)

        return self


class Plaid(RBiclusterBase):
    """A wrapper for R the BCPlaid algorithm.

    Constant values over rows or columns

    """

    MODEL = 'BCPlaid'

    # Hyperparameters
    params = {
        'cluster': 'b',
        'fit_model': robjects.r('y ~ m + a + b'),
        'background': True,
        'row_release': 0.7,
        'col_release': 0.7,
        'shuffle': 3,
        'back_fit': 0,
        'max_layers': 20,
        'iter_startup': 5,
        'iter_layer': 10,
        'back_fit': 0,
        'verbose': False,
    }

    def __init__(self, random_state=0, **kwargs):

        super().__init__(random_state, **kwargs)

        # Update parameters.
        for key in kwargs:
            if key in self.params.keys():
                self.params[key] = kwargs[key]

        self.set_params(**kwargs)

        # NOTE:
        self._output = None

        self.rows_ = None
        self.columns_ = None
        self.biclusters_ = None

    def fit(self, X, y=None, **kwargs):

        self._fit(self.MODEL, X, self.params)

        return self


if __name__ == '__main__':

    import bibench.all as bb

    import numpy as np
    from matplotlib import pyplot as plt

    """
    import os

    from rpy2 import robjects as r
    try:
        from rpy2.robjects.packages import importr
    except ImportError:
        from rpy2.interactive import importr

    #enables automatic conversion from numpy to R
    import rpy2.robjects.numpy2ri

    rpy2.robjects.numpy2ri.activate()

    #from bibench.bicluster import get_row_col_matrices
    """
    import numpy as np
    from matplotlib import pyplot as plt


    from sklearn.datasets import make_biclusters, make_checkerboard
    from sklearn.datasets import samples_generator as sg
    from sklearn.cluster.bicluster import SpectralCoclustering
    from sklearn.metrics import consensus_score

    n_clusters = 9
    #data, rows, columns = make_biclusters(
    #    shape=(300, 200), n_clusters=n_clusters, noise=5,
    #    shuffle=False, random_state=0)

    data, rows, columns = make_checkerboard(
        shape=(50, 10), n_clusters=n_clusters, noise=5,
        shuffle=False, random_state=0)

    shuf, row_idx, col_idx = sg._shuffle(data, random_state=0)

    model = SpectralCoclustering(
        n_clusters=n_clusters,random_state=0)
    model.fit(data)

    #model2 = Plaid()
    #model2.fit(data)
