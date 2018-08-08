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


import operator

import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_array
from sklearn.metrics import consensus_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import samples_generator as sgen

from sklearn.preprocessing import StandardScaler


class PerformanceTracker:
    """Determines the optimal algorithm for each class of test data.

    Args:
        test_classes (list of str): The labels for each class of test data.
        models (list of str): The labels for each model included in the
            experiment.

    """

    def __init__(self, test_classes, models):

        self.test_classes = test_classes
        self.models = models

        # NOTE: Attribute set with instance
        self.model_stats = self._setup_model_stats()

    def _setup_model_stats(self):
        # Returns an object that logs model performance for each class of test
        # data.

        _model_stats = {}
        for test_class in self.test_classes:
            _model_stats[test_class] = {
                model: 0 for model in self.models
            }

        return _model_stats

    def update_stats(self, results):
        """Updates the counter for how many times a model has been
        voted optimal for a particular test data class."""

        for test_class in self.test_classes:
            name, _ = results[test_class]
            self.model_stats[test_class][name] += 1

    @property
    def best_models(self):
        """Determines the optimal model for each class of test data."""

        winners = {}
        for test_class in self.test_classes:
            candidates = self.model_stats[test_class]
            winner, _ = max(candidates.items(), key=operator.itemgetter(1))
            winners[test_class] = winner

        return winners


class Experiment:
    """Perform experiments by applying an algorithm to data and measure the
    performance.

    Args:
        data (array-like):
        rows (array-like):
        cols (array-like):

    """

    def __init__(self, models_and_params, verbose=1, random_state=None):

        self.models_and_params = models_and_params
        self.verbose = verbose
        self.random_state = random_state

        # NOTE: Necessary to scale data to avoid sklearn inf/NaN error.
        self.scaler = StandardScaler()

        # NOTE: A cross val split producing only train and no test split.
        self.dummy_cv = [(slice(None), slice(None))]

        # NOTE: Attributes set with instance.
        self.grids = None
        self.results = None

        self._data = None
        self._rows = None
        self._cols = None

    # ERROR: Must gen report with results from all four classes of test data
    @property
    def performance_report(self):
        """Generates and returns a report with grid search results."""

        for grid in self.grids:
            pass

        """# NOTE: Only one CV split => avg score = score from single run.
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

        return report"""


    def execute(self, data, indicators, target='score'):
        """Performs model comparison for each class of test data."""

        rows, cols = indicators

        if self.verbose > 0:
            print('Experiment initiated:\n{}'.format('-' * 21))

        self.results, self.grids = {}, []
        for key in data.keys():

            if self.verbose > 0:
                print('Training set: `{}`'.format(key))

            self._data = data[key]
            self._rows, self._cols = rows[key], cols[key]

            # Holds the winning model for each class of test data.
            winning_model, best_score = self.compare_models(target=target)

            self.results[key] = winning_model

            if self.verbose > 0:
                name, _ = winning_model
                print('Best model: {}\nScore: {}\n'.format(name, best_score))

        return self

    def compare_models(self, target):
        """Compare model performance on target basis."""

        if target == 'score':
            return self.score_selection()
        elif target == 'time':
            return self.time_selection()
        else:
            raise ValueError('')

    def score_selection(self):
        """Compare the model performance with respect to a score metric."""

        _train, self.row_idx, self.col_idx = sgen._shuffle(
            self._data, random_state=self.random_state
        )
        _train_std = self.scaler.fit_transform(_train)

        winning_model, best_score = None, -np.float('inf')
        for model, param_grid in self.models_and_params:

            # Determine the best hyperparameter combo for that model
            grid = GridSearchCV(
                model(random_state=self.random_state), param_grid,
                scoring=self.jaccard, cv=self.dummy_cv,
                return_train_score=True, refit=False
            )
            grid.fit(_train_std, y=None)

            if self.verbose > 1:
                print('Model performance:\nName: {}\nScore: {}\n'
                      ''.format(model.__name__, grid.best_score_))

            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                winning_model = (model.__name__, model(**grid.best_params_))

            # Store grid to generate performance report.
            self.grids.append(grid)

        return winning_model, best_score

    def jaccard(self, estimator, train=None):
        """Computes the Jaccard coefficient as a measure of similarity between
        two sets of biclusters.

        Args:
            estimator ():
            train: Ignored

        Returns:
            (float): The Jaccard coefficient value.

        """

        rows, cols = estimator.biclusters_
        if len(rows) == 0 or len(cols) == 0:
            return 0.0
        else:
            ytrue = (self._rows[:, self.row_idx], self._cols[:, self.col_idx])
            return consensus_score((rows, cols), ytrue)


    # NOTE:
    # * Not performing grid search, but only logging time of fitting
    #   model to data?
    def time_selection(self):

        pass


class MultiExperiment(Experiment):

    def __init__(self, models_and_params, nruns=2, verbose=1, random_state=1):

        super().__init__(models_and_params, verbose, random_state)

        self.nruns = nruns

        # NOTE:
        self._tracker = None
        self._multi_results = None

    @property
    def best_models(self):

        return self._tracker.best_models

    @property
    def model_labels(self):

        return [model.__name__ for model, _ in self.models_and_params]

    def execute_all(self, dataset, test_classes, target='score'):

        self._multi_results = {str(num): None for num in range(self.nruns)}
        self._tracker = PerformanceTracker(test_classes, self.model_labels)
        for num in range(self.nruns):

            if self.verbose > 0:
                print('Run number: {}'.format(num))

            # Perform single experiment with dataset.
            for num, (data, rows, cols) in enumerate(dataset):

                # Perform model comparison with test data class.
                self.execute(data, (rows, cols), target=target)
                # NOTE: The counter is updated for each run. The winning models
                # is thus the winners across all runs.
                self._tracker.update_stats(self.results)

            # Store the results for each experimental run.
            self._multi_results[num] = self.results

        return self


if __name__ == '__main__':

    import datasets
    import pandas as pd

    from sklearn.cluster import SpectralBiclustering
    from sklearn.cluster import SpectralCoclustering

    from datasets import gen_test_sets

    SEED = 0

    data_feats = pd.read_csv(
        './../data/data_characteristics.csv', sep='\t', index_col=0
    )
    array_size = (1000, 100)
    var_num_clusters = [2, 3]

    cluster_exp_data = [
        datasets.gen_test_sets(
            data_feats, sparse=[False, True, False, True],
            non_neg=[False, True, False, True],
            shape=array_size, n_clusters=n_clusters, seed=0
        )
        for n_clusters in var_num_clusters
    ]

    skmodels_and_params = [
        (
            SpectralBiclustering, {
                'n_clusters': [5], 'method': ['log', 'bistochastic'],
                'n_components': [6, 9, 12], 'n_best': [3, 6]
            }
        ),
        (
            SpectralCoclustering, {'n_clusters': [5]}
        )
    ]

    me = MultiExperiment(skmodels_and_params)
    me.execute_all(cluster_exp_data, data_feats.index)
