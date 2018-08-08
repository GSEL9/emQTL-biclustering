# -*- coding: utf-8 -*-
#
# model_selection.py
#

"""
Model selection framework.

The framework applies models with different hyperparemeter settings to classes
of test data. For each model, the the Jaccard coefficient and the time
complexity is recorded.
size.

"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


import time

from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_array
from sklearn.metrics import consensus_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import samples_generator as sgen


class Experiment:
    """Perform experiments by applying an algorithm to data and measure the
    performance.

    Args:
        data (array-like):
        rows (array-like):
        cols (array-like):

    """

    def __init__(self, data, rows, cols, verbose=True, seed=None):

        self.data = check_array(data)
        self.rows = check_array(rows)
        self.cols = check_array(cols)

        #self.models_and_params = models_and_params
        self.verbose = verbose
        self.seed = seed

        # NOTE: Attributes set with instance.
        self.grid = None
        self.scores = None
        self.extimes = None
        self.best_settings = None
        self.best_model = None

    def execute(self, models_and_params):
        """Conducts a grid search experiments to determine the optimal
        hyperparameter settings with respect to the test data.

        args:
            models_and_params (list of typles): Holds tuples of algorithm and
                corresponding parameter grid.
            
        """

        train, self.row_idx, self.col_idx = sgen._shuffle(
            self.data, random_state=self.seed
        )

        for model, param_grid in models_and_params:

            if self.verbose:
                print('Testing model: {}'.format(model.__name__))

            self.grid = GridSearchCV(
                SpectralBiclustering(random_state=self.seed),
                param_grid, scoring=self.jaccard, cv=[(slice(None), slice(None))]
            )
            self.grid.fit(train, y=None)

        return self

    def jaccard(self, estimator, train):
        """Computes the Jaccard coefficient as a measure of similarity between
        two sets of biclusters.

        Args:
            target_coords (tuple): The target row and column bicluster
                indicators.
            pred_coords (tuple): The predicted biclsuters with row and column
                indicators.

        Returns:
            (float): The Jaccard coefficient value.

        """

        ypred = estimator.biclusters_
        ytrue = (self.rows[:, self.row_idx], self.cols[:, self.col_idx])

        score = consensus_score(ypred, ytrue)

        return score


if __name__ == '__main__':

    import numpy as np
    import pandas as pd

    from sklearn.cluster import SpectralBiclustering
    from sklearn.cluster import SpectralCoclustering

    from datasets import gen_test_sets

    SEED = 0

    data_feats = pd.read_csv(
        './../data/data_characteristics.csv', sep='\t', index_col=0
    )
    test_data, test_rows, test_cols = gen_test_sets(
        data_feats, sparse=True, shape=(500, 300), n_clusters=3, seed=SEED
    )

    models_and_params = [
        (
            SpectralBiclustering,
            {'n_clusters': [2, 4], 'method': ['log']}
        ),
        (
            SpectralCoclustering,
            {'n_clusters': [2, 4]}
        )
    ]

    data = test_data[data_feats.index[0]]
    rows, cols = test_rows[data_feats.index[0]], test_cols[data_feats.index[0]]

    experiment = Experiment(data, rows, cols)
    experiment.execute(models_and_params)
