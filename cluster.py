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

        # Format R biclustering algorithm output to numpy.narray.
        self.fetch_biclusters(X)

        return self


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

        # Format R biclustering algorithm output to numpy.narray.
        self.fetch_biclusters(X)

        return self


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

        # Format R biclustering algorithm output to numpy.narray.
        self.fetch_biclusters(X)

        return self


if __name__ == '__main__':

    import datasets
    import pandas as pd

    data_feats = pd.read_csv(
        './../data/data_characteristics.csv', sep='\t', index_col=0
    )

    feats = data_feats.loc[data_feats.index[0], :]
    data, _, _ = datasets.gen_biclusters(
        feats, sparse=False, shape=(300, 200), n_clusters=4, seed=0
    )

    model = Spectral()
    rows, cols = model.fit_transform(data)

    print(rows.shape)
    print(cols.shape)
    print(model.rows_.shape, model.columns_.shape)
