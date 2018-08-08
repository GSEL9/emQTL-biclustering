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

from sklearn.preprocessing import StandardScaler


class Experiment:
    """Perform experiments by applying an algorithm to data and measure the
    performance.

    Args:
        data (array-like):
        rows (array-like):
        cols (array-like):

    """

    def __init__(self, models_and_params, verbose=True, seed=None):

        self.models_and_params = models_and_params
        self.verbose = verbose
        self.seed = seed

        # NOTE: Necessary to scale data to avoid sklearn inf/NaN error.
        self.scaler = StandardScaler()

        # NOTE: A cross val split producing only train and no test split.
        self.dummy_cv = [(slice(None), slice(None))]

        # NOTE: Attributes set with instance.
        self.grid = None
        self.model_stats = None

        self._data = None
        self._rows = None
        self._cols = None

    @property
    def gen_gs_report(self):
        """Generates and returns a report with grid search results."""

        # NOTE: Only one CV split => avg score = score from single run.
        target_cols = ['split0_test_score', 'mean_fit_time', 'params']

        # Collect unnecessary column labels.
        droppings = []
        for key in self.grid.cv_results_.keys():
            if key not in target_cols:
                droppings.append(key)

        # Remove unnecessary columns.
        for key in droppings:
            self.grid.cv_results_.pop(key)

        # Create a properly formatted pandas.DataFrame
        report = pd.DataFrame(
            self.grid.cv_results_, columns=['score', 'train_time', 'params']

        )
        report.index.name = 'param_grid_num'

        return report

    def execute(self, data, indicators):
        """Performs model comparison for each class of test data."""

        rows, cols = indicators

        if self.verbose:
            print('Experiment initiated:\n{}'.format('-' * 21))

        self.model_stats = {}
        for key in data.keys():

            if self.verbose:
                print('Training set: `{}`'.format(key))

            self._data = data[key]
            self._rows, self._cols = rows[key], cols[key]

            # Holds the winning model for each class of test data.
            self.model_stats[key] = self.compare_models()

            #return self

    def compare_models(self):
        """Compare the performance of models with optimal hyperparemeter
        settings obtained from a grid search.

        """

        _train, self.row_idx, self.col_idx = sgen._shuffle(
            self._data, random_state=self.seed
        )
        _train_std = self.scaler.fit_transform(_train)

        winning_model, best_score = None, -np.float('inf')
        for model, param_grid in self.models_and_params:

            # Determine the best hyperparameter combo for that model
            self.grid = GridSearchCV(
                SpectralBiclustering(random_state=self.seed),
                param_grid, scoring=self.jaccard, cv=self.dummy_cv,
                return_train_score=True
            )
            self.grid.fit(_train_std, y=None)

            if self.verbose:
                print(
                    'Model performance:\nName: {}\nScore: {}\n'
                     ''.format(model.__name__, self.grid.best_score_)
                )

            if self.grid.best_score_ > best_score:
                best_score = self.grid.best_score_
                winning_model = model.__name__

        return winning_model

    def jaccard(self, estimator, train=None):
        """Computes the Jaccard coefficient as a measure of similarity between
        two sets of biclusters.

        Args:
            estimator ():
            train: Ignored

        Returns:
            (float): The Jaccard coefficient value.

        """

        ypred = estimator.biclusters_
        ytrue = (self._rows[:, self.row_idx], self._cols[:, self.col_idx])

        return consensus_score(ypred, ytrue)


if __name__ == '__main__':

    import numpy as np
    import pandas as pd

    from sklearn.cluster import SpectralBiclustering
    from sklearn.cluster import SpectralCoclustering

    from datasets import gen_test_sets

    SEED = 0

    # NOTE: Grid Search
    # * Do not want to optimize with respect to the number of clusters. Pass
    #   the correct number of clusters to allow optimization of other params.
    # * Passing data to execute() and not constructor allows running
    #   experiments with same models and params over different datasets.

    # NOTE: Experiment
    # * One execution = one run for each class of test data with given models
    #   and params.
    # * When conducting experiments (in ipynb) with a generated test dataset,
    #   run the same experiment with the same dataset again as second
    #   parallell addressing variability between experiments.
    # * Record the winning model for each class of test data. The model with
    #   the most wins per test data class is the recommended model for clustering
    #   that type of data.
    # * Necessary to standardize data in order to avoid sklearn inf/NaN error.
    #   Also necessary to standardize when clustering reference data?

    data_feats = pd.read_csv(
        './../data/data_characteristics.csv', sep='\t', index_col=0
    )
    test_data, test_rows, test_cols = gen_test_sets(
        data_feats, sparse=True, shape=(500, 300), n_clusters=3, seed=SEED
    )

    models_and_params = [
        (
            SpectralBiclustering,
            {'n_clusters': [3], 'method': ['log', 'bistochastic']}
        ),
        (
            SpectralCoclustering,
            {'n_clusters': [3], }
        )
    ]

    # Parallell num 1
    experiment = Experiment(models_and_params, verbose=True)
    experiment.execute(test_data, (test_rows, test_cols))
    print(experiment.model_stats)

    # Paralell num 2
