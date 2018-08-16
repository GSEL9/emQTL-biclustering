# -*- coding: utf-8 -*-
#
# algorithms.py
#

"""
Wrappers for R biclustering algorithms.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


import numpy as np
import rpy2.robjects as robjects

from base import RBiclusterBase

# NOTE: To binarize Bimax input.
from sklearn.preprocessing import binarize


class ChengChurch(RBiclusterBase):
    """A wrapper for the R BCCC algorithm.

    Constant values

    """

    MODEL = 'BCCC'

    # Hyperparameters
    params = {
        'delta': 0.1,
        'alpha': 1.5,
        'number': 2
    }

    def __init__(self, random_state=0, n_clusters=2, **kwargs):

        super().__init__(random_state=random_state, **kwargs)

        # NOTE: Gets assigned to params['number'].
        self.n_clusters = n_clusters

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
    def n_clusters(self):

        return self.params['number']

    @n_clusters.setter
    def n_clusters(self, value):

        self.params['number'] = value

    def fit(self, X, y=None, **kwargs):

        self._fit(self.MODEL, X, self.params)

        return self


class Xmotifs(RBiclusterBase):
    """A wrapper for the R BCXmotifs algorithm.

    Coherent  correlation  over  rows and  columns

    """

    MODEL = 'BCXmotifs'

    # Hyperparameters
    params = {
        'number': 2,
        'ns': 200,
        'nd': 100,
        'sd': 5,
        'alpha': 0.05
    }

    def __init__(self, random_state=0, n_clusters=2, **kwargs):

        super().__init__(random_state=random_state, **kwargs)

        # NOTE: Gets assigned to params['number'].
        self.n_clusters = n_clusters

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
    def n_clusters(self):

        return self.params['number']

    @n_clusters.setter
    def n_clusters(self, value):

        self.params['number'] = value

    def fit(self, X, y=None, **kwargs):

        # NOTE: Requires discrete input.
        X_discrete = X.astype(int)

        self._fit(self.MODEL, X_discrete, self.params)

        return self


class Plaid(RBiclusterBase):
    """A wrapper for R the BCPlaid algorithm.

    Constant values over rows or columns

    """

    MODEL = 'BCPlaid'

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
        'verbose': False,
    }

    def __init__(self, random_state=0, n_clusters=2, **kwargs):

        super().__init__(random_state=random_state, **kwargs)

        # NOTE: n_clusters param is ignored.
        self.n_clusters = n_clusters

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

    def fit(self, X, y=None, **kwargs):

        self._fit(self.MODEL, X, self.params)

        return self


class Bimax(RBiclusterBase):
    """A wrapper for the R biclust package BCBimax algorithm.

    Searches for submatrices of ones in a logical matrix.

    """

    MODEL = 'BCBimax'

    # Hyperparameters
    params = {
        'minr': 4,
        'minc': 4,
        'maxc': 10000,
        'number': 2,
    }

    def __init__(self, random_state=0, n_clusters=2, **kwargs):

        super().__init__(random_state=random_state, **kwargs)

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

    def fit(self, X, y=None, **kwargs):

        # NOTE: Requires binary input. Thresh = mean input matrix.
        _thresh = np.mean(X, axis=None)
        X_binary = binarize(X, threshold=_thresh)

        self._fit(self.MODEL, X_binary, self.params)

        return self


class Quest(RBiclusterBase):
    """A wrapper for the R biclust package BCQuest algorithm.

    Searches subgroups of questionairs with same or similar
    answer to some questions.

    """

    MODEL = 'BCQuest'

    # Hyperparameters
    params = {
        'nd': 10,
        'ns': 10,
        'sd': 5,
        'alpha': 0.05,
        'number': 2,
    }

    def __init__(self, random_state=0, n_clusters=2, **kwargs):

        super().__init__(random_state=random_state, **kwargs)

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

    def fit(self, X, y=None, **kwargs):

        self._fit(self.MODEL, X, self.params)

        return self


class Spectral(RBiclusterBase):
    """A wrapper for the R biclust package BCSpectral algorithm.
    """

    MODEL = 'BCSpectral'

    # Hyperparameters
    params = {
        'normalization': 'log',
        'numberOfEigenvalues': 3,
        'minc': 4,
        'minr': 4,
        'withinVar': 5
    }

    def __init__(self, random_state=0, n_clusters=2, **kwargs):

        super().__init__(random_state=random_state, **kwargs)

        # NOTE: n_clusters param is ignored.
        self.n_clusters = n_clusters

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

    def fit(self, X, y=None, **kwargs):

        self._fit(self.MODEL, X, self.params)

        return self


if __name__ == '__main__':

    import numpy as np
    import pandas as pd

    from sklearn.datasets import make_biclusters
    from sklearn.datasets import samples_generator as sg
    from sklearn.metrics import consensus_score

    data, rows, columns = make_biclusters(
        shape=(500, 30), n_clusters=10, noise=5,
        shuffle=False, random_state=0
    )
    data, row_idx, col_idx = sg._shuffle(data, random_state=0)

    model = Spectral()
    model.fit(data)

    score = consensus_score(model.biclusters_,
                        (rows[:, row_idx], columns[:, col_idx]))
    print("consensus score: {:.3f}".format(score))

    """
    import testsets
    import model_selection

    data_feats = pd.read_csv(
        './../data/data_ids/data_characteristics.csv',
        sep='\t', index_col=0
    )

    array_size = (1000, 100)
    var_num_clusters = [2, 4, 6]

    # NOTE: Each list element is a tuple of data, rows, cols
    cluster_exp_data = [
        testsets.gen_test_sets(
            data_feats, sparse=[False, True, False, True],
            non_neg=[False, True, False, True],
            shape=array_size, n_clusters=n_clusters, seed=0
        )
        for n_clusters in var_num_clusters
    ]

    models_and_params = [
    (
        Spectral, {
            'minc': [4, 5],
        }
    )
    ]

    sk_multi_exper = model_selection.MultiExperiment(
        models_and_params, verbose=0, random_states=[0, 1]
    )
    sk_multi_exper.execute_all(
        cluster_exp_data, data_feats.index
    )
    """
