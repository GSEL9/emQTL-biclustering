# -*- coding: utf-8 -*-
#
# testsets.py
#

"""
Synthetic test data generators.

Test data is generated in resemblance to the experimentally obtained datasets.
The motivation generating synthetic data is the availability of ground truth
data which can be utilized in hyperparameter selection and performance
evaluation.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


import numpy as np
import pandas as pd

from sklearn.datasets import make_biclusters


def gen_test_sets(feats, sparse, **kwargs):
    """Generate datasets with similar characteristics to reference datasets."""

    datasets, rows, columns = {}, {}, {}
    for sparse, key in zip(sparse, feats.index):
        datasets[key], rows[key], columns[key] = gen_biclusters(
            feats.loc[key, :], sparse=sparse, **kwargs
        )

    return datasets, rows, columns


def gen_biclusters(feats, sparse=False, **kwargs):
    """Geenrates synthetic biclsuter data from reference characteristics.

    Args:
        feats():
        sparse (bool):

    Kwargs:
        shape (tuple):
        n_clusters (int):
        random_state (int):

    Returns:
        (tuple): Data matrix with row and column indicators.

    """

    # Generates dense data.
    _data, rows, columns = make_biclusters(
        shape=kwargs['shape'],
        n_clusters=kwargs['n_clusters'],
        minval=feats['min'],
        maxval=feats['max'],
        noise=feats['std'],
        random_state=kwargs['seed'],
        shuffle=False
    )
    # Suppress negative values occuring from the noise term.
    data = np.absolute(_data)

    if sparse:
        return percentile_filter(data, feats), rows, columns
    else:
        return data, rows, columns


def percentile_filter(data, feats):
    """Suppresses values according to a fraction of nonzero values in a
    reference matrix.

    Args:
        data (array-like): The input data matrix.
        feats():

    Returns:
        (array-like): The filtered data matrix.

    """

    # Determines the fraction of nonzero values in the data.
    sparsity_frac = feats['nonzeros'] / (feats['nrows'] * feats['ncols'])

    # Determine the threshold valeu from the sprasity fraction percentile.
    thresh = np.percentile(data.ravel(), q=100 * (1 - sparsity_frac))

    # Replace p-values below threshold with zero.
    data[data < thresh] = 0

    return data


if __name__ == '__main__':

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    from sklearn.datasets import samples_generator as sgen

    # Characteristics samples from experimental data

    data_feats = pd.read_csv(
        './../data/data_characteristics.csv', sep='\t', index_col=0
    )
    test_data, rows, cols = gen_test_sets(
        data_feats, sparse=[False, True, False, True],
        shape=(500, 300), n_clusters=5, seed=0
    )
