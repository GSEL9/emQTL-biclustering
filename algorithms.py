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
        'number': 2
    }

    def __init__(self, random_state=0, n_clusters=1, **kwargs):

        super().__init__(random_state=random_state, **kwargs)

        # NOTE: Hack to allow sklearn API in specifying number of clusters.
        self.params['number'] = n_clusters

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
        'number': 2,
        'ns': 200,
        'nd': 100,
        'sd': 5,
        'alpha': 0.05
    }

    def __init__(self, random_state=0, n_clusters=1, **kwargs):

        super().__init__(random_state=random_state, **kwargs)

        # NOTE: Hack to allow sklearn API in specifying number of clusters.
        self.params['number'] = n_clusters

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

    def __init__(self, random_state=0, n_clusters=1, **kwargs):

        super().__init__(random_state=random_state, **kwargs)

        # NOTE: n_clusters param is ignored.

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
