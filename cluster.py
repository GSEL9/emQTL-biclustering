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

# TEMP:
from sklearn.base import BaseEstimator, ClusterMixin


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


class BiMax:

    def __init__(self):

        pass

    def fit(self, X, y=None, **kwargs):

        binary_data = np.where(a>threshold, upper, lower)


class Plaid(RBiclusterBase, BaseEstimator, ClusterMixin):
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
        'verbose': False
    }

    def __init__(self, random_state=0, **kwargs):

        super().__init__()

        self.random_state = random_state

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

    @property
    def biclusters_(self):

        return self._biclusters

    @biclusters_.setter
    def biclusters_(self, value):

        if value is None:
            return
        else:
            # TODO: Type checking
            self._biclusters = value

    def set_params(self, **kwargs):

        # Assign parameters to attributes.
        for key, value in self.params.items():
            # Add underscore instead of dot to attribute
            _key = key.replace('.', '_')
            setattr(self, _key, kwargs.get(_key, value))

    def get_params(self, deep=False):

        return self.params

    def fit(self, X, y=None, **kwargs):

        # Run R biclustering algorithm.
        self.execute_r_function('BCPlaid', X, self.params)

        # Format R biclustering algorithm output to numpy.narray.
        self.rows_, self.cols_ = self.fetch_biclusters(X)
        # Assign to attribute.
        self.biclusters_ = (self.rows_, self.cols_)

        return self


if __name__ == '__main__':

    import datasets
    import model_selection

    import numpy as np
    import pandas as pd

    from sklearn.datasets import samples_generator as sgen
    from sklearn.cluster import SpectralBiclustering
    from sklearn.metrics import consensus_score
    from sklearn.datasets import make_biclusters
    from sklearn.datasets import samples_generator as sg

    data_feats = pd.read_csv(
        './../data/data_characteristics.csv', sep='\t', index_col=0
    )
    test_data, rows, cols = datasets.gen_test_sets(
        data_feats,
        sparse=[False, True, False, True],
        non_neg=[False, True, False, True],
        shape=(500, 300),
        n_clusters=5,
        seed=0
    )
    rmodels_and_params = [
        (
            Plaid, {
                'row_release': [0.5, 0.7],
            }
        ),
    ]

    #X = test_data[data_feats.index[0]]
    #test_rows = rows[data_feats.index[0]]
    #test_cols = cols[data_feats.index[0]]


    """data, rows, columns = make_biclusters(
        shape=(500, 500), n_clusters=5, noise=5,
        shuffle=False, random_state=0)
    data, row_idx, col_idx = sg._shuffle(X, random_state=0)


    ref_mod = SpectralBiclustering()
    ref_mod.fit(data)

    ref_score = consensus_score(
        ref_mod.biclusters_, (rows[:, row_idx], columns[:, col_idx])
    )
    print(ref_score)"""

    #model = Plaid()
    #model.fit(data)
    #score = consensus_score(
    #    model.biclusters_, (rows[:, row_idx], columns[:, col_idx])
    #)
    #print(score)

    experiment = model_selection.Experiment(rmodels_and_params, verbose=1)
    experiment.execute(test_data, (rows, cols), target='score')
    #for num, (test_data, test_rows, test_cols) in enumerate(cluster_exp_data):
    #    experiment.execute(test_data, (test_rows, test_cols), target='score')
