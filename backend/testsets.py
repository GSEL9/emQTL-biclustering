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

from sklearn.datasets import make_biclusters, make_checkerboard


def gen_testsets(feats, sparse, non_neg, kind='bicluster', **kwargs):
    """Generate datasets with similar characteristics to reference datasets.

    Args:
        feats (pandas.DataFrame): The characteristics of each dataset to be
            generated. The requirements are max, min and std.
        sparse (list): Boolean indicators to whether the data is generated as
            sparse or not.
        non_neg (list): Boolean indicators to whether the generated data
            contains negative values or not.

    Returns:
        (tuple):

    """

    if kind == 'bicluster':
        generator = make_biclusters
    elif kind == 'checkerboard':
        generator = make_checkerboard
    else:
        raise ValueError('Invalid generator function: `{}`'.format(kind))

    datasets, rows, columns = {}, {}, {}
    for key_num, key in enumerate(feats.index):
        datasets[key], rows[key], columns[key] = gen_testdata(
            generator,
            feats.loc[key, :],
            sparse=sparse[key_num], non_neg=non_neg[key_num],
            **kwargs
        )

    return datasets, rows, columns


def gen_testdata(generator, feats, sparse=False, non_neg=False, **kwargs):
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
    data, rows, columns = generator(
        shape=kwargs['shape'],
        n_clusters=kwargs['n_clusters'],
        minval=feats['min'],
        maxval=feats['max'],
        noise=feats['std'],
        random_state=kwargs['seed'],
        shuffle=False
    )
    # Detmerine if suppressing negative values occuring from the noise term,
    # and filtering values from threshold.
    if sparse:
        if non_neg:
            return percentile_filter(np.absolute(data), feats), rows, columns
        else:
            return percentile_filter(data, feats), rows, columns
    else:
        if non_neg:
            return np.absolute(data), rows, columns
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

    # NOTE: Added convenience term derived from experience.
    thresh = np.percentile(data.ravel(), q=100 * (1 - (sparsity_frac + 0.1)))

    # Replace p-values below threshold with zero.
    data[(data > 0) & (data < thresh)] = 0

    return data


if __name__ == '__main__':

    import numpy as np
    import pandas as pd

    import seaborn as sns
    import matplotlib.pyplot as plt

    from sklearn.datasets import samples_generator as sgen

    # Characteristics samples from experimental data
    data_feats = pd.read_csv(
        './../data/data_id/data_characteristics.csv',
        sep='\t', index_col=0
    )
    # NOTE: Every other dataset is sparse and the oposite sets contains negative values
    sample_data, _, _ = gen_testsets(
        data_feats, sparse=[False, True, False, True],
        non_neg=[True, True, False, False],
        shape=(500, 300), n_clusters=3, seed=0,
        kind='bicluster'
    )


    nrows, ncols = 2, 2
    labels = list(sample_data.keys())

    fig, axes = plt.subplots(
        ncols=ncols, nrows=nrows, figsize=(12, 10), sharey=True, sharex=True
    )
    num = 0
    for row in range(nrows):
        for col in range(ncols):
            _data = sample_data[labels[num]]
            sns.heatmap(
                _data, ax=axes[row, col], cbar=True,
                vmin=np.min(_data), vmax=np.max(_data)
            )
            axes[row, col].set_title(labels[num])
            axes[row, col].axis('off')

            num += 1

    plt.tight_layout()
    plt.show()
