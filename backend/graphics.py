# -*- coding: utf-8 -*-
#
# graphics.py
#

"""
Tools to display detecte biclusters.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


import numpy as np
import matplotlib.pyplot as plt


def fetch_model_dumps(path_to_models, labels):
    """Read model dumps from disk."""

    models = {}
    for num, path in enumerate(path_to_models):
        models[labels[num]] = joblib.load(path)

    return models


# NOTE: Relies on sklearn API.
def _reconstruct_data(sk_models, ref_data):
    # Shuffles data according to detected biclusters.

    recon_data = {}
    for label, model in sk_models.items():

        _data = ref_data[label].values
        row_sorted_data = _data[np.argsort(model.row_labels_), :]
        sorted_col_idx = np.argsort(model.column_labels_)
        recon_data[label] = row_sorted_data[:, sorted_col_idx]

    return recon_data


def sklearn_graphics(data, title, savefig=True, out_path=None):
    """Generate a heatmap and save figure to disk."""

    plt.figure(figsize=(10, 10))
    plt.title(title)
    sns.heatmap(
        data, robust=True,
        cmap=plt.cm.RdBu_r, fmt='f',
        vmin=np.min(data), vmax=np.max(data),
    )
    plt.axis('off')
    plt.tight_layout()

    if savefig:
        plt.savefig(out_path)

    return plt


def r_graphics():

    # NOTE: Necessary to execute biclust_graphics.R
    # with model, hparams and dataset to produce hm.
    pass


if __name__ == '__main__':

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    from sklearn.externals import joblib

    ref_labels = [
        'orig_pvalues', 'sel_pvalues','orig_pcc', 'sel_pcc'
    ]
    ref_data = {
        ref_labels[0]: pd.read_csv(
            './../data/train/orig_pvalues_prep.csv', sep=',', index_col=0
        ).T,
        #ref_labels[1]: pd.read_csv(
        #    './../data/train/sel_pvalues_prep.csv', sep=',', index_col=0
        #).T,
        #ref_labels[2]: pd.read_csv(
        #    './../data/train/orig_pcc_prep.csv', sep=',', index_col=0
        #).T,
        #ref_labels[3]: pd.read_csv(
        #    './../data/train/sel_pcc_prep.csv', sep=',', index_col=0
        #).T,
    }
    # NB: Spectral Coclustering
    path_to_skcheck = [
        './../_model_dumps/sk_bic_orig_prep_pvalues.pkl',
        #'./../_model_dumps/sk_bic_sel_prep_pvalues.pkl',
        #'./../_model_dumps/sk_bic_orig_prep_pcc.pkl',
        #'./../_model_dumps/sk_bic_sel_prep_pcc.pkl',
    ]
    # Spectral biclustering
    path_to_skbic = [
        './../_model_dumps/sk_checker_orig_prep_pvalues.pkl',
        './../_model_dumps/sk_checker_sel_prep_pvalues.pkl',
        './../_model_dumps/sk_checker_orig_prep_pcc.pkl',
        './../_model_dumps/sk_checker_sel_prep_pcc.pkl',
    ]
    sk_models = fetch_model_dumps(
        path_to_skcheck, ref_labels
    )
    reconstr_data = _reconstruct_data(
        sk_models, ref_data
    )
    sklearn_graphics(
        reconstr_data[ref_labels[0]],
        'Biclustering results of preprocessed\n'
        'Bonferroni corrected p-values',
        './../predictions/imgs/org_prep_pvalues.png'
    )
