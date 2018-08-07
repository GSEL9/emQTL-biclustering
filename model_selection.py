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



class Experiment:
    """Perform experiments by applying an algorithm to data and measure the
    performance."""

    def __init__(self, models_and_params, verbose=True, seed=None):

        self.models_and_params = models_and_params
        self.verbose = verbose
        self.seed = seed

        # NOTE: Attributes set with instance.
        self.scores = None
        self.extimes = None
        self.best_settings = None
        self.best_model = None

    def execute(self, X, y=None, score='index', **kwargs):

        print(X)

        for model, param_grid in self.models_and_params:

            if self.verbose:
                print('Testing model: {}'.format(model.__name__))

            # NOTE: Iterate through each dataset
            grid = GridSearchCV(
                model(random_state=self.seed), param_grid,
                scoring=self.jaccard,
                cv=[(slice(None), slice(None))]
            )

            # TODO: Make scoring funciton capable to deal with
            # need to bicluster output from wrapper predict,
            # and provide a target to the grid

            # for data, row, col in X:
            #     grid.fit(data, target)



        return self


    def run_test(self, estimator, dataset, score):

        data, rows, cols = dataset

        start = time.time()
        row_idx, col_idx = estimator.fit_transform(data)
        extime = time.time() - start

        jaccard = self._score((rows, cols), (row_idx, col_idx))

        if score == 'indesx':
            return jaccard
        elif score == 'time':
            return extime

    def jaccard(self, target_coords, pred_coords):
        """Computes the Jaccard coefficient as a measure of similarity between
        two sets of biclusters.

        Args:
            target_coords (tuple): The original row and column bicluster
                indicators.
            pred_coords (tuple): The predicted row and column bicluster
                indicators.

        Returns:
            (float): The Jaccard coefficient value.

        """

        rows, cols = target_coords
        row_idx, col_idx = pred_coords

        score = consensus_score(
            self.biclusters_, (rows[:, row_idx], columns[:, col_idx])
        )

        return score

    def results(self):

        pass

    def reports(self, graphical=True):

        pass


class GridSearchWrapper:
    """A wrapper to enable scikit-learn grid search with biclustering
    algorithms."""

    def __init__(self):

        pass

    def fit(self):

        pass

    def score(self):
        pass



if __name__ == '__main__':
    # Goal: Determine the model and hyperparameter settings corresponding to
    # the best score (time/J index) across datasets.
    #
    # Algorithm: Grid Search
    # ------------------------
    # winning model = None
    # for each model
    #     best_local_scores, best_settings = None, None
    #     for each hparam setting
    #         for each class of test data
    #             score, extime = run_test()
    #         avg_scores = mean(score)
    #         if avg_scores > best_local_scores
    #             best_settings = num_hparam_setting
    #

    import numpy as np
    import pandas as pd

    from datasets import gen_test_sets
    from cluster import Spectral

    SEED = 0

    data_feats = pd.read_csv(
        './../data/data_characteristics.csv', sep='\t', index_col=0
    )
    test_data, rows, cols = gen_test_sets(
        data_feats, sparse=True, shape=(500, 300), n_clusters=3, seed=SEED
    )

    test_data_configs = {
        'shape': [],
        'n_clusters': []
    }

    models_and_params = [
        (
            Spectral,
            {'n_clusters': [2, 4], 'method': ['log']}
        )
    ]

    #X = test_data[data_feats.index[0]]

    #experiment = Experiment(models_and_params)
    #experiment.execute((X, rows[0], cols[0]))
