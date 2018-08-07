# -*- coding: utf-8 -*-
#
# model_selection.py
#

"""
Tools to perform model selection.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


import time

from sklearn.model_selection import GridSearchCV


# NOTE:
# * Use verbose to track experiment development. Remove when performing final
#   tests to not disturb actual execution time.
class Experiment:
    """Perform experiments by applying an algorithm to data and measure the
    performance."""

    def __init__(self, model, params, verbose=0):

        self.model = model
        self.params = params
        self.verbose = verbose

        # NOTE: Variables set with instance.
        self.score = None
        self.exec_times = None

    def execute(self, X, y=None, **kwargs):

        self.score, self.exec_times = [], []

        start = time.time()
        # do test
        exec_time = time.time() - start

        self.exec_times.append(exec_time)

    def _run_test(self):

        if verbose > 0:
            pass
        else:
            pass

    def results(self):

        pass


if __name__ == '__main__':
    # Algorithm: Grid Search
    # for each dataset, rows, cols (represents n-th fold)
    #     for each parameter combination
    #          determine the clustering J score

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

    spectral_hparams = {
        'n_clusters': [2, 4, 8, 12],
        'method': ['bistochastic', 'log'],

    }

    #model = Spectral(random_state=SEED)

    for key, value in spectral_hparams.items():
        print(key, value)
