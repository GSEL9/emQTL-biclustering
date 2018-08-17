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

        # NOTE: Attributes set with instance.
        self.model_params = self._model_param_stats()
        self.model_scores = self._model_score_stats()
        self.winning_stats = self._model_win_stats()

    def _model_param_stats(self):
        # Returns an object that logs winning model params for each class of
        # test data.

        _model_stats = {}
        for test_class in self.test_classes:
            _model_stats[test_class] = {
                model: [] for model in self.models
            }

        return _model_stats

    def _model_score_stats(self):
        # Returns an object that logs winning model score for each class of
        # test data.

        _model_stats = {}
        for test_class in self.test_classes:
            _model_stats[test_class] = {
                model: [] for model in self.models
            }

        return _model_stats

    def _model_win_stats(self):
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

        for test_class, (model, params, score) in results.items():

            self.winning_stats[test_class][model] += 1
            self.model_scores[test_class][model].append(score)
            self.model_params[test_class][model].append(params)

        return self

    def winner_models(self, results):
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

    def __init__(self, models_and_params, random_state, verbose):

        self.models_and_params = models_and_params
        self.random_state = random_state
        self.verbose = verbose

        # NOTE: Necessary to scale data to avoid sklearn inf/NaN error.
        self.scaler = StandardScaler()

        # NOTE: A single cross-val split producing only train data.
        self.dummy_cv = [(slice(None), slice(None))]

        # NOTE: Attribute set with instance.
        self.results = None

        self._data = None
        self._rows = None
        self._cols = None

    def execute(self, data, rows, cols, exp_id):
        """Performs model comparison for each class of test data.

        Args:
            data (array-like): The original data matrix.
            rows (array-like): The original row cluster membership indicators.
            cols (array-like): The original column cluster membership
                indicators.
            exp_id (int):

        """

        if self.verbose > 0:
            print('Experiment initiated:\n{}'.format('-' * 21))

        if self.results is None:
            self.results = {exp_id: {}}

        for num, key in enumerate(data.keys()):

            if self.verbose > 0:
                print('Training set: `{}`'.format(key))

            self.results[exp_id][key] = {}

            self._data = data[key]
            self._rows, self._cols = rows[key], cols[key]

            # Winning model, best hyperparams, best score.
            name, params, score = self.compare_models(
                n_clusters=self._rows.shape[0]
            )
            self.results[exp_id][key] = (name, params, score)

            if self.verbose > 0:
                print('Best model: {}\nScore: {}\n'.format(name, score))

        return self

    def compare_models(self, n_clusters):
        """Compare the model performance with respect to a score metric.

        Args:
            random_state (int):
            n_clusters (int):

        Returns:
            (tuple): The name and score of the selected model, in addition to
                a model instance with the optimal hyperparameter settings.

        """

        # Shuffle data matrix, row and col indicators for random state.
        _train, self.row_idx, self.col_idx = sgen._shuffle(
            self._data, random_state=self.random_state
        )
        # NB: Subtract mean and divide by std.
        _train_std = self.scaler.fit_transform(_train)

        best_score = -np.float('inf')
        winning_model, best_params = None, None
        for model, params in self.models_and_params:

            # Determine the best hyperparameter combo for that model
            _grid = GridSearchCV(
                model(random_state=self.random_state, n_clusters=n_clusters),
                param_grid=params,
                scoring=self.jaccard,
                n_jobs=16,
                cv=self.dummy_cv,
                return_train_score=True,
                refit=False
            )
            _grid.fit(_train_std, y=None)

            if self.verbose > 1:
                print('Model performance:\nName: {}\nScore: {}\n'
                      ''.format(model.__name__, _grid.best_score_))

            if _grid.best_score_ > best_score:
                winner_model = model.__name__
                best_score = _grid.best_score_
                winner_params = _grid.best_params_

        return (winner_model, winner_params, best_score)


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
    def __init__(self, models_and_params, random_state, verbose=1):

        super().__init__(models_and_params, random_state, verbose)

        self._tracker = None

    @property
    def best_setup(self):

        best_setup = {}
        for test_class, name in self.best_models.items():

            _best_model = None
            for model, _ in self.models_and_params:
                if model.__name__ == name:
                    _best_model = model
                else:
                    pass

            coll_params = pd.DataFrame(self.model_params[test_class][name])
            target_params = coll_params.T.mode(axis=1)

            try:
                target_params.columns = [test_class]
            except:
                target_params.dropna(axis=1, inplace=True)
                target_params.columns = [test_class]

            best_setup[test_class] = _best_model(
                **target_params.to_dict()[test_class]
            )

        return best_setup

    @property
    def best_models(self):

        return self._tracker.winner_models(self.results)

    @property
    def model_votes(self):

        return self._tracker.winning_stats

    @property
    def model_params(self):

        return self._tracker.model_params

    @property
    def performance_report(self):

        score_stats, test_classes = [], []
        for test_class, winner in self.best_models.items():

            win_scores = self._tracker.model_scores[test_class][winner]
            score_stats.append(
                (winner, np.mean(win_scores), np.std(win_scores))
            )
            test_classes.append(test_class)

        _report = pd.DataFrame(
            score_stats, columns=['model', 'score_avg', 'score_std'],
            index=test_classes
        )
        _report.index.name = 'test_class'

        return _report

    @property
    def model_labels(self):

        return [model.__name__ for model, _ in self.models_and_params]

    def execute_all(self, datasets, test_classes):

        self._tracker = PerformanceTracker(test_classes, self.model_labels)

        self.results = {}
        for exp_id, (dataset, rows, cols) in enumerate(datasets):

            if self.verbose > 0:
                print('Dataset number: {}'.format(exp_id + 1))

            self.results[exp_id] = {}

            # Perform model comparison with test data class.
            self.execute(dataset, rows, cols, exp_id)

            # NOTE: Num wins counter is continued for each run.
            self._tracker.update_stats(self.results[exp_id])

        return self


if __name__ == '__main__':

    import testsets

    import numpy as np

    from sklearn.cluster.bicluster import SpectralBiclustering
    from sklearn.cluster.bicluster import SpectralCoclustering

    SEED = 0

    data_feats = pd.read_csv(
        './../data/data_id/data_characteristics.csv', sep='\t', index_col=0
    )

    # NOTE: The reference data contains approx. ten times more rows than columns.
    array_size = (100, 1000)
    n_clusters = [5, 7]

    # random_state = seed
    # set_num = num of datasets with different features (e.g. num clusters)

    # NOTE: Each list element is a tuple of data, rows, cols
    cluster_exp_data = [
        testsets.gen_test_sets(
            data_feats, sparse=[False, True, False, True],
            non_neg=[True, True, False, False],
            shape=array_size, n_clusters=num, seed=SEED
        )
        for num in n_clusters
    ]

    sk_models_and_params = [
        (
            SpectralBiclustering, {
                'n_components': [6],
            }
        ),
        (
            SpectralCoclustering, {
                'n_init': [10, 20],
                'mini_batch': [True, False]
            }
        )
    ]

    sk_multi_exper = MultiExperiment(
        sk_models_and_params, verbose=0, random_state=SEED
    )
    sk_multi_exper.execute_all(
        cluster_exp_data, data_feats.index
    )

    b = sk_multi_exper.best_setup
    print(b)
