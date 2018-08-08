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

    def __init__(self, models_and_params, verbose=True, seed=None):

        self.models_and_params = models_and_params
        self.verbose = verbose
        self.seed = seed

        # NOTE: A cross val split producing only train and no test split.
        self.dummy_cv = [(slice(None), slice(None))]

        # NOTE: Attributes set with instance.
        self.data = None
        self.rows = None
        self.cols = None
        # The latest (in case of execute()) model selection grid search object 
        self.grid = None
        self.best_model = None
        #
        self.model_stats = None

    @property
    def gen_gs_report(self):
        """Generates and returns a report with grid search results."""

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

    # NOTE: Passing data to execute() and not constructor allows running
    # experiments with same models and params over different datasets.
    def execute(self, data, indicators):
        """Performs a model comparison experiment by applying """

        rows, cols = indicators

        if self.verbose:
            print('Conducting experiment:')

        self.model_stats = {}
        for key in test_data.keys():

            if self.verbose:
                print('Training set: {}'.format(key))

            self.model_stats[key] = {}
            self.data, self.rows, self.cols = data[key], rows[key], cols[key]

            self.model_selection()

    def model_selection(self):
        """Compare the performance of models with optimal hyperparemeter
        settings obtained from a grid search.

        """

        train, self.row_idx, self.col_idx = sgen._shuffle(
            self.data, random_state=self.seed
        )
        # Completes a single run over the data with each model.
        for model, param_grid in self.models_and_params:

            if self.verbose:
                print('Testing model: {}'.format(model.__name__))

            self.grid = GridSearchCV(
                SpectralBiclustering(random_state=self.seed),
                param_grid, scoring=self.jaccard, cv=self.dummy_cv,
                return_train_score=True
            )
            self.grid.fit(train, y=None)

            # Best estimator from the grid search (i.e. the model fitted with
            # the optimal param combo) = self.grid.best_estimator_
            #

            # QUESTION: How to determine th ebest model for this data?

            if self.verbose:
                self.performance_report(model.__name__)

        return self

    def jaccard(self, estimator, train):
        """Computes the Jaccard coefficient as a measure of similarity between
        two sets of biclusters.

        Args:
            estimator ():
            train: Ignored

        Returns:
            (float): The Jaccard coefficient value.

        """

        ypred = estimator.biclusters_
        ytrue = (self.rows[:, self.row_idx], self.cols[:, self.col_idx])

        score = consensus_score(ypred, ytrue)

        return score

    def performance_report(self, name):
        """Prints a model performance report including training and test scores,
        and the difference between the training and test scores."""

        print('Model performance', '\n{}'.format('-' * 24))
        print('Name: {}'.format(name))


        print('Avg train time {}')


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

    # NOTE: Grid Search
    # * Do not want to optimize with respect to the number of clusters. Pass
    #   the correct number of clusters to allow optimization of other params.
    # * With only one CV split (test dataset), the avg params are actually the
    #   results for the whole run (which goes into the gs report).

    # NOTE: Experiment
    # * One execution = one run for each class of test data with given models
    #   and params.
    # * When conducting experiments (in ipynb) with a generated test dataset,
    #   run the same experiment with the same dataset again as second
    #   parallell addressing variability between experiments.

    models_and_params = [
        (
            SpectralBiclustering,
            {'n_clusters': [3], 'method': ['log', 'bistochastic']}
        ),
        #(
        #    SpectralCoclustering,
        #    {'n_clusters': [2, 4]}
        #)
    ]

    data = test_data[data_feats.index[0]]
    rows, cols = test_rows[data_feats.index[0]], test_cols[data_feats.index[0]]

    experiment = Experiment(models_and_params)
    experiment.execute(test_data, (test_rows, test_cols))
