# -*- coding: utf-8 -*-
#
# validation.py
#

"""

"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


import ast

import numpy as np


class Bicluster:
    """Utility representation of a bicluster."""

    def __init__(self, rows, cols, data):

        self.rows = rows
        self.cols = cols
        self.data = data

        self.setup()

        # NOTE: Attributes set with instance.
        self.nbiclusters = None
        self.nrows = None
        self.ncols = None

    def setup(self):

        sample, ref = np.shape(self.rows)[0], np.shape(self.cols)[0]
        if not sample == ref:
            raise RuntimeError('Sample clusters: {}, ref clusters {}'
                               ''.format(sample, ref))
        self.nbiclusters = sample

        self.nrows, self.ncols = np.shape(self.data)

        return self

    def bools(self):

        pass

    @property
    def indicators(self):
        """Determine coordiantes of row and column indicators
        for each bicluster.
        """

        row_idx, col_idx = [], []
        for cluster_num in range(self.nbiclusters):

            rows_bools = self.rows[cluster_num, :] != 0
            cols_bools = self.cols[cluster_num, :] != 0

            rows = [index for index, elt in enumerate(rows_bools) if elt]
            cols = [index for index, elt in enumerate(cols_bools) if elt]

            row_idx.append(rows), col_idx.append(cols)

        return row_idx, col_idx

    @property
    def cluster_stats(self):

        pass


class ReferenceMatching:

    @classmethod
    def from_files(cls, path_to_cpgs, path_to_genes, num_ref_clusters=2):

        targets = {str(num + 1): [] for num in range(num_ref_clusters)}
        # Read target CpG data.
        target_cpgs = targets.copy()
        with open(path_to_cpgs, 'r') as cpgfile:

            cpg_contents = cpgfile.read().split('\n')
            # Skip header line.
            for row in cpg_contents[1:]:
                try:
                    value, idx, _ = row.split()
                    target_cpgs[idx].append(ast.literal_eval(value))
                except:
                    pass
        # Read target gene data.
        target_genes = targets.copy()
        with open(path_to_genes, 'r') as genefile:

            gene_contents = genefile.read().split('\n')
            # Skip header line.
            for row in gene_contents[1:]:
                try:
                    value, idx = row.split()
                    target_genes[idx].append(ast.literal_eval(value))
                except:
                    pass

        return ReferenceMatching(cpgs=target_cpgs, genes=target_genes)

    def __init__(self, cpgs, genes):

        self.cpgs = cpgs
        self.genes = genes

    @property
    def cpgs1(self):

        return self.cpgs['1']

    @property
    def cpgs2(self):

        return self.cpgs['2']

    @property
    def genes1(self):

        return self.genes['1']

    @property
    def genes2(self):

        return self.genes['2']

    def recovery_score(self, pred):
        """The fraction of true items among the predicted
        items."""

        return np.isin(pred, true).sum() / np.size(true)

    def relevance_score(self, pred):
        """The fraction of predicted items not among the
        true items."""

        return np.isin(true, pred).sum() / np.size(pred)

    def cluster_scores(preds, refs, targets):
        scores = {
            'cl1_recovery': [], 'cl1_relevance': [],
            'cl2_recovery': [], 'cl2_relevance': [],
        }
        for num, pred_ids in enumerate(preds):

            pred = refs[pred_ids]
            true1, true2 = targets['1'], targets['2']
            # Frac targets among predicted (true positives).
            scores['cl1_recovery'].append(recovery_score(true1, pred))
            scores['cl2_recovery'].append(recovery_score(true2, pred))
            # Frac detected targets compared to cluster size.
            scores['cl1_relevance'].append(relevance_score(true1, pred))
            scores['cl2_relevance'].append(relevance_score(true2, pred))

        df_scores = pd.DataFrame(scores).T
        df_scores.columns = [
            'cluster_{}'.format(str(num + 1))
            for num in range(np.shape(preds)[0])
        ]
        df_scores.index = pd.MultiIndex.from_product(
            [('cluster1', 'cluster2'), ('recovery', 'relevance')]
        )

        return df_scores


if __name__ == '__main__':

    path_target_genes = './../data/test/emQTL_Cluster_genes.txt'
    path_target_cpgs = './../data/test/emQTL_Clusters_CpGs.txt'

    rm = ReferenceMatching.from_files(
        path_target_cpgs, path_target_genes
    )

    print(rm.indicators())
