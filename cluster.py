# -*- coding: utf-8 -*-
#
# cluster.py
#

"""
Various scikit-learn compatible clustering algorithms.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


import os
import subprocess

import numpy as np
import rpy2.robjects as robjects

from utils import PathError
from base import RBiclusterBase, BinaryBiclusteringBase
from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.metrics import consensus_score


class ChengChurch(RBiclusterBase):
    """A wrapper for the R BCCC algorithm.

    Kwargs:
        delta ():
        alpha ():
        number ():

    Attributes:
        rows_ ():
        columns_ ():
        biclusters_():
        row_labels_ ():
        column_labels_ ():

    """

    # Hyperparameters
    params = {
        'delta': 0.1,
        'alpha': 1.5,
        'number': 100
    }

    def __init__(self, method='BCCC', **kwargs):

        super().__init__()

        self.method = method

        # Iterate through kwargs and update parameters.
        for key in kwargs:
            if key in self.params.keys():
                self.params[key] = kwargs[key]

    def fit(self, X, y=None, **kwargs):

        # Run R biclustering algorithm.
        self.execute_r_function(self.method, X, self.params)

        return self

    def transform(self, X, y=None, **kwargs):

        # TODO: Check is fitted

        # Format R biclustering algorithm output to numpy.narray.
        self.fetch_biclusters(X)

        return self.biclusters_

    def fit_transform(self, X, y=None, **kwargs):

        self.fit(X, y=y, **kwargs)

        return self.transform(X, y=y, **kwargs)


class Plaid(RBiclusterBase):
    """A wrapper for R the BCPlaid algorithm.

    Args:
        method (str): The R biclust function method name.

    Kwargs:
        cluster (str, {r, c, b}): Determines to cluster rows, columns or both.
            Defaults to both.
        model (str): The model formula to fit each layer. Defaults to linear
            model y ~ m + a + b.

    Attributes:
        rows_ ():
        columns_ ():
        biclusters_():
        row_labels_ ():
        column_labels_ ():

    """

    # Hyperparameters
    params = {
        'cluster': 'b',
        'fit.model': robjects.r('y ~ m + a + b'),
        'background': True,
        'row.release': 0.7,
        'col.release': 0.7,
        'shuffle': 3,
        'back.fit': 0,
        'max.layers': 20,
        'iter.startup': 5,
        'iter.layer': 10,
        'verbose': False
    }

    def __init__(self, method='BCPlaid', **kwargs):

        super().__init__()

        self.method = method

        # Iterate through kwargs and update parameters.
        for key in kwargs:
            if key in self.params.keys():
                self.params[key] = kwargs[key]

    def fit(self, X, y=None, **kwargs):

        # Run R biclustering algorithm.
        self.execute_r_function(self.method, X, self.params)

        return self

    def transform(self, X, y=None, **kwargs):

        # TODO: Check is fitted

        # Format R biclustering algorithm output to numpy.narray.
        self.fetch_biclusters(X)

        return self.biclusters_

    def fit_transform(self, X, y=None, **kwargs):

        self.fit(X, y=y, **kwargs)

        return self.transform(X, y=y, **kwargs)


class XMotifs(RBiclusterBase):
    """A wrapper for the R BCXmotifs algorithm.

    Args:
        number (int): Number of bicluster to be found.
        ns (int): Number of seeds.
        nd (int): Number of determinants.
        sd (int): Size of discriminating set; generated for each seed.
        alpha (float): Scaling factor for column.

    Attributes:
        rows_ ():
        columns_ ():
        biclusters_():
        row_labels_ ():
        column_labels_ ():

    """

    # Hyperparameters
    params = {
        'number': 1,
        'ns': 200,
        'nd': 100,
        'sd': 5,
        'alpha': 0.05
    }

    def __init__(self, method='BCXmotifs', **kwargs):

        super().__init__()

        self.method = method

        # Iterate through kwargs and update parameters.
        for key in kwargs:
            if key in self.params.keys():
                self.params[key] = kwargs[key]

    def fit(self, X, y=None, **kwargs):

        X_discrete = X.astype(int)

        # Run R biclustering algorithm.
        self.execute_r_function(self.method, X_discrete, self.params)

        return self

    def transform(self, X, y=None, **kwargs):

        # TODO: Check is fitted

        # Format R biclustering algorithm output to numpy.narray.
        self.fetch_biclusters(X)

        return self.biclusters_

    def fit_transform(self, X, y=None, **kwargs):

        self.fit(X, y=y, **kwargs)

        return self.transform(X, y=y, **kwargs)


class Spectral:
    """A wrapper for the scikit-learn spectral biclustering algorithms.

    Kwargs:
        n_clusters (int or tuple): (n_row_clusters, n_column_clusters),
        method (str, {bistochastic, log, scale}): Method of normalizing and
            converting singular vectors into biclusters.,
        n_components (int): Number of singular vectors to check.,
        n_best=3,
        svd_method=’randomized’,
        n_svd_vecs=None,
        mini_batch=False,
        init=’k-means++’,
        n_init=10,
        n_jobs=1,
        random_state=None

    Attributes:
        rows_ ():
        columns_ ():
        biclusters_():
        row_labels_ ():
        column_labels_ ():

    """

    def __init__(self, model='bi', **kwargs):

        # NOTE: All kwargs are directly passed to algorithm.
        if model == 'bi':
            self.model = SpectralBiclustering(**kwargs)
        elif model == 'co':
            self.model = SpectralCoclustering(**kwargs)
        else:
            raise ValueError('Invalid model: `{}` not among [`bi`, `co`]'
                             ''.format(model))

    @property
    def row_labels_(self):

        return self.model.row_labels_

    @property
    def column_labels_(self):

        return self.model.column_labels_

    def fit(self, X, y=None, **kwargs):

        self.model.fit(X, y=y, **kwargs)

        return self

    def transform(self, X, y=None, **kwargs):

        return self.model.biclusters_

    def fit_transform(self, X, y=None, **kwargs):

        self.fit(X, y=y, **kwargs)

        return self.transform(X, y=y, **kwargs)


if __name__ == '__main__':

    # QUESTION: How to visualize the clustering result?

    pass
