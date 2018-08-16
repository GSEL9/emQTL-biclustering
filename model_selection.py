# -*- coding: utf-8 -*-
#
# model_selection.py
#

"""
Model selection framework.

The framework applies models with different hyperparemeter settings to classes
of test data. For each model, the the Jaccard coefficient is recorded.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


import ast
import operator

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
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
        self.model_scores = self._setup_model_score_stats()
        self.winning_stats = self._setup_model_win_stats()

    def _setup_model_score_stats(self):
        # Returns an object that logs winning model score for each class of
        # test data.

        _model_stats = {}
        for test_class in self.test_classes:
            _model_stats[test_class] = {
                model: [] for model in self.models
            }

        return _model_stats

    def _setup_model_win_stats(self):
        # Returns an object that counts model wins for each class of test data.

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
            name, _, score = results[test_class]
            self.winning_stats[test_class][name] += 1
            self.model_scores[test_class][name].append(score)

        return self

    @property
    def winner_models(self):
        """Determines the optimal model for each class of test data."""

        winners = {}
        for test_class in self.test_classes:
            candidates = self.winning_stats[test_class]
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

    def __init__(self, grid, verbose):

        self.grid = grid
        self.verbose = verbose

        # NOTE: Necessary to scale data to avoid sklearn inf/NaN error.
        self.scaler = StandardScaler()

        # NOTE: A cross val split producing only train and no test split.
        self.dummy_cv = [(slice(None), slice(None))]

        # NOTE: Attributes set with instance.
        self.results = None

        self._data = None
        self._rows = None
        self._cols = None

    def execute(self, data, rows, cols, random_state):
        """Performs model comparison for each class of test data."""

        if self.verbose > 0:
            print('Experiment initiated:\n{}'.format('-' * 21))

        self.results = {}
        for num, key in enumerate(data.keys()):

            if self.verbose > 0:
                print('Training set: `{}`'.format(key))

            self._data = data[key]
            self._rows, self._cols = rows[key], cols[key]

            # Winning model, hest hparams, best score
            self.results[key] = self.compare_models(
                random_state=random_state, n_clusters=self._rows.shape[0]
            )
            if self.verbose > 0:
                name, _, score = self.results[key]
                print('Best model: {}\nScore: {}\n'.format(name, score))

        return self

    def compare_models(self, random_state, n_clusters):
        """Compare the model performance with respect to a score metric."""

        _train, self.row_idx, self.col_idx = sgen._shuffle(
            self._data, random_state=random_state
        )
        _train_std = self.scaler.fit_transform(_train)

        winning_model, best_params, best_score = None, None, -np.float('inf')
        for model, params in self.grid:

            # Determine the best hyperparameter combo for that model
            _grid = GridSearchCV(
                model(random_state=random_state, n_clusters=n_clusters),
                params,
                scoring=self.jaccard,
                cv=self.dummy_cv,
                return_train_score=True,
                refit=False
            )
            _grid.fit(_train_std, y=None)

            if self.verbose > 1:
                print('Model performance:\nName: {}\nScore: {}\n'
                      ''.format(model.__name__, _grid.best_score_))

            if _grid.best_score_ > best_score:
                best_score = _grid.best_score_
                winner_name = model.__name__
                winner_model = model(**_grid.best_params_)

        # NOTE: Model returned is not fitted.
        return (winner_name, winner_model, best_score)


    def jaccard(self, estimator, train=None):
        """Computes the Jaccard coefficient as a measure of similarity between
        two sets of biclusters.

        The Jaccard index achieves its minimum of zero when the biclusters to
        not overlap at all and its maximum of one when they are identical.

        Args:
            estimator (object): The fitted model holding bicluster esitmates.
            train: Ignored.

        Returns:
            (float): The Jaccard coefficient value.

        """

        rows, cols = estimator.biclusters_

        if len(rows) == 0 or len(cols) == 0:
            return 0.0
        else:
            ytrue = (self._rows[:, self.row_idx], self._cols[:, self.col_idx])
            return consensus_score((rows, cols), ytrue)


class MultiExperiment(Experiment):

    # NOTE: Inferes num clusters from shape of test data.
    def __init__(self, grid, random_states, verbose=1):

        super().__init__(grid, verbose)

        self.random_states = random_states

        # NOTE:
        self._tracker = None
        self._multi_results = None

    @property
    def best_models(self):

        return self.results

    @property
    def class_winners(self):

        return self._tracker.winner_models

    @property
    def model_votes(self):

        return self._tracker.winning_stats

    @property
    def performance_report(self):

        score_stats, test_classes = [], []
        for test_class, winner in self.class_winners.items():

            win_scores = self._tracker.model_scores[test_class][winner]
            score_stats.append(
                (winner, np.mean(win_scores), np.std(win_scores))
            )
            test_classes.append(test_class)

        _report = pd.DataFrame(
            score_stats, columns=['model', 'avg score', 'std score'],
            index=test_classes
        )
        _report.index.name = 'test_class'

        return _report

    @property
    def model_labels(self):

        return [model.__name__ for model, _ in self.grid]

    def execute_all(self, dataset, test_classes):

        self._tracker = PerformanceTracker(
            test_classes, self.model_labels
        )
        for random_state in self.random_states:

            if self.verbose > 0:
                print('Experiment seed: {}'.format(random_state))

            # Perform single experiment with dataset.
            for class_num, (data, rows, cols) in enumerate(dataset):

                if self.verbose > 0:
                    print('Test set number: {}'.format(class_num + 1))

                # Perform model comparison with test data class.
                self.execute(data, rows, cols, random_state)

                # NOTE: Num wins counter is continued for each run.
                self._tracker.update_stats(self.results)

        return self


if __name__ == '__main__':

    pass
