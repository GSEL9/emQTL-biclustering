# -*- coding: utf-8 -*-
#
# cluster.py
#

"""
Various scikit-learn compatible clustering algorithms.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


import os
import subprocess

import numpy as np
import rpy2.robjects as robjects

from utils import PathError
from base import RBiclusterBase, BinaryBiclusteringBase
from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.cluster.bicluster import SpectralCoclustering


class ChengChurch(RBiclusterBase):
    """A wrapper for the R BCCC algorithm.

    Kwargs:
        delta ():
        alpha ():
        number ():

    Attributes:
        rows_ ():
        columns_ ():
        biclusters_():
        row_labels_ ():
        column_labels_ ():

    """

    # Hyperparameters
    params = {
        'delta': 0.1,
        'alpha': 1.5,
        'number': 100
    }

    def __init__(self, method='BCCC', **kwargs):

        super().__init__()

        self.method = method

        # Iterate through kwargs and update parameters.
        for key in kwargs:
            if key in self.params.keys():
                self.params[key] = kwargs[key]

    def fit(self, X, y=None, **kwargs):

        # Run R biclustering algorithm.
        self.execute_r_function(self.method, X, self.params)

        return self

    def transform(self, X, y=None, **kwargs):

        # TODO: Check is fitted

        # Format R biclustering algorithm output to numpy.narray.
        self.fetch_biclusters(X)

        return self.biclusters_

    def fit_transform(self, X, y=None, **kwargs):

        self.fit(X, y=y, **kwargs)

        return self.transform(X, y=y, **kwargs)


class Plaid(RBiclusterBase):
    """A wrapper for R the BCPlaid algorithm.

    Args:
        method (str): The R biclust function method name.

    Kwargs:
        cluster (str, {r, c, b}): Determines to cluster rows, columns or both.
            Defaults to both.
        model (str): The model formula to fit each layer. Defaults to linear
            model y ~ m + a + b.

    Attributes:
        rows_ ():
        columns_ ():
        biclusters_():
        row_labels_ ():
        column_labels_ ():

    """

    # Hyperparameters
    params = {
        'cluster': 'b',
        'fit.model': robjects.r('y ~ m + a + b'),
        'background': True,
        'row.release': 0.7,
        'col.release': 0.7,
        'shuffle': 3,
        'back.fit': 0,
        'max.layers': 20,
        'iter.startup': 5,
        'iter.layer': 10,
        'verbose': False
    }

    def __init__(self, method='BCPlaid', **kwargs):

        super().__init__()

        self.method = method

        # Iterate through kwargs and update parameters.
        for key in kwargs:
            if key in self.params.keys():
                self.params[key] = kwargs[key]

    def fit(self, X, y=None, **kwargs):

        # Run R biclustering algorithm.
        self.execute_r_function(self.method, X, self.params)

        return self

    def transform(self, X, y=None, **kwargs):

        # TODO: Check is fitted

        # Format R biclustering algorithm output to numpy.narray.
        self.fetch_biclusters(X)

        return self.biclusters_

    def fit_transform(self, X, y=None, **kwargs):

        self.fit(X, y=y, **kwargs)

        return self.transform(X, y=y, **kwargs)


class XMotifs(RBiclusterBase):
    """A wrapper for the R BCXmotifs algorithm.

    Args:
        number (int): Number of bicluster to be found.
        ns (int): Number of seeds.
        nd (int): Number of determinants.
        sd (int): Size of discriminating set; generated for each seed.
        alpha (float): Scaling factor for column.

    Attributes:
        rows_ ():
        columns_ ():
        biclusters_():
        row_labels_ ():
        column_labels_ ():

    """

    # Hyperparameters
    params = {
        'number': 1,
        'ns': 200,
        'nd': 100,
        'sd': 5,
        'alpha': 0.05
    }

    def __init__(self, method='BCXmotifs', **kwargs):

        super().__init__()

        self.method = method

        # Iterate through kwargs and update parameters.
        for key in kwargs:
            if key in self.params.keys():
                self.params[key] = kwargs[key]

    def fit(self, X, y=None, **kwargs):

        X_discrete = X.astype(int)

        # Run R biclustering algorithm.
        self.execute_r_function(self.method, X_discrete, self.params)

        return self

    def transform(self, X, y=None, **kwargs):

        # TODO: Check is fitted

        # Format R biclustering algorithm output to numpy.narray.
        self.fetch_biclusters(X)

        return self.biclusters_

    def fit_transform(self, X, y=None, **kwargs):

        self.fit(X, y=y, **kwargs)

        return self.transform(X, y=y, **kwargs)


class Spectral:
    """A wrapper for the scikit-learn spectral biclustering algorithms.

    Attributes:
        rows_ ():
        columns_ ():
        biclusters_():
        row_labels_ ():
        column_labels_ ():

    """

    def __init__(self, model='bi', **kwargs):

        # NOTE: All kwargs are directly passed to algorithm.
        if model == 'bi':
            self.model = SpectralBiclustering(**kwargs)
        elif model == 'co':
            self.model = SpectralCoclustering(**kwargs)
        else:
            raise ValueError('Invalid model: `{}` not among [`bi`, `co`]'
                             ''.format(model))

    @property
    def row_labels_(self):

        return self.model.row_labels_

    @property
    def column_labels_(self):

        return self.model.column_labels_

    def fit(self, X, y=None, **kwargs):

        self.model.fit(X, y=y, **kwargs)

        return self

    def transform(self, X, y=None, **kwargs):

        return self.model.biclusters_

    def fit_transform(self, X, y=None, **kwargs):

        self.fit(X, y=y, **kwargs)

        return self.transform(X, y=y, **kwargs)


class CPB(BinaryBiclusteringBase):
    """A wrapper for the CPB binary algorithm.

    Detects biclusters based on row correlations.

    Attributes:
        rows_ ():
        columns_ ():
        biclusters_():
        row_labels_ ():
        column_labels_ ():

    """

    # Initial bicluster
    INIT_BINARY = 'init_bicluster'

    # Stem of output files holding the biclustering results.
    STEM_OUTPUT_FILE = '.out'

    # Hyperparameters
    params = {
        'nclus': 2,
        'targetpcc': 0.9,
        'fixed_row': -1,
        'fixed_col': -1,
        'fixw': 0,
        'min_seed_rows': 3,
        'max_seed_rows': None,
        'targetpcc': 0.9,
        'fixw': 0
    }

    def __init__(self, model='cpb', file_format='txt', temp=False, **kwargs):

        super().__init__(model, file_format, temp)

        # Iterate through kwargs and update parameters.
        for key in kwargs:
            if key in self.params.keys():
                self.params[key] = kwargs[key]

    def fit(self, X, y=None, **kwargs):

        #_X = check_array(X)

        # Call to base method to set paths for temp dir and data.
        self.setup_io()
        # Creates file of input data according to algorithm requirements.
        self.format_input(X)
        # Call to application
        self.exec_clustering()

    def transform(self, X, y=None, **kwargs):

        # TODO: Check is fitted

        self.fetch_biclusters(X)

        if self.temp:
            self.io_teardown_temp()

        return self.biclusters_

    def fit_transform(self, X, y=None, **kwargs):

        self.fit(X, y=y, **kwargs)

        return self.transform(X, y=y, **kwargs)

    def format_input(self, X):

        self.params['nrows'], self.params['ncols'] = np.shape(X)
        with open(self.path_data, 'w') as outfile:

            outfile.write(
                '{0} {1}'.format(self.params['nrows'], self.params['ncols'])
            )
            for row in X:
                outfile.write('\n')
                # TODO: Convert map row data to str instead of list comp?
                outfile.write(' '.join([str(value) for value in row]))
            outfile.write('\n')
            outfile.close()

        return self

    def exec_clustering(self):

        # Create a file to hold the results.
        self.create_results_file()
        # Change to current working dir.
        current_loc = os.getcwd()
        try:
            os.chdir(self.path_dir)
            command = (
                '{model} {outfile} {initfile} 1 {targetpcc} {fixw}'.format(
                    model=self.model,
                    outfile=os.path.abspath(self.path_data),
                    **self.params
                )
            )
            subprocess.check_call(command.split())

        except OSError:
            raise PathError('CPB not on $PATH')

        finally:
            os.chdir(current_loc)

        return self

    def create_results_file(self):

        self.params['initfile'] = os.path.join(
            self.path_dir, 'initfile.{0}'.format(self.file_format)
        )
        self.params['init_binary'] = self.INIT_BINARY

        # Create initial bicluster file
        command = (
            '{init_binary} '
            '{nrows} {ncols} {nclus} '
            '{min_seed_rows} {max_seed_rows} {fixed_row} {fixed_col} '
            '{initfile}'.format(**self.params)
        )
        try:
            subprocess.check_call(command.split())
        except OSError:
            raise PathError('Executable {0} not on $PATH'
                            ''.format(self.params['init_binary']))

        return self

    def fetch_biclusters(self, X):
        """

        """

        results_dir = os.path.abspath(self.path_dir)
        dir_files = os.listdir(results_dir)

        num_biclusters = 0
        rows, cols = [], []
        for _file in dir_files:

            _, stem = os.path.splitext(_file)
            if stem == self.STEM_OUTPUT_FILE:

                output_file = os.path.join(results_dir, _file)
                row_idx, col_idx = self.format_output(output_file, X)
                rows.append(row_idx), cols.append(col_idx)

                num_biclusters += 1

        # Create arrays of False.
        row_clusters = np.zeros((num_biclusters, X.shape[0]), dtype=bool)
        col_clusters = np.zeros((num_biclusters, X.shape[1]), dtype=bool)

        for num in range(num_biclusters):
            row_clusters[num, :][rows[num]] = True
            col_clusters[num, :][cols[num]] = True

        self.rows_, self.columns_ = row_clusters, col_clusters
        self.biclusters_ = (self.rows_, self.columns_)

        return self

    def format_output(self, filename, X):
        """Reads the bicluster in a single CPB output file.

        Args:
            filename (str):
            X (array-like):

        Returns:
            (tuple): The bicluster row and column indicators.

        """

        rows, cols = [], []
        with open(filename, 'r') as outfile:

            target = rows
            for line in outfile:
                if line[0] == 'R':
                    continue
                elif line[0] == 'C':
                    target = cols
                    continue
                else:
                    target.append(int(line.split()[0]))

            rows.sort(), cols.sort()

        return rows, cols


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    from sklearn.datasets import make_checkerboard
    from sklearn.datasets import samples_generator as sg

    from sklearn.metrics import consensus_score

    # Generate sklearn sample data.
    # rows: array of shape (n_clusters, X.shape[0],)
    # columns: array of shape (n_clusters, X.shape[1],)
    target, rows, columns = make_checkerboard(
        shape=(500, 300), n_clusters=(4, 3), noise=10, shuffle=False,
        random_state=0
    )
    data, row_idx, col_idx = sg._shuffle(target, random_state=0)

    # Use sklearn model as reference to output structure
    #ref_model = SpectralBiclustering()
    #ref_model.fit(data)
    #rows, cols = ref_model.rows_, ref_model.columns_
    #print(rows.shape, cols.shape)

    """
    model = Plaid()
    model.fit_transform(data)
    rows, cols = model.rows_, model.columns_
    print(rows.shape, cols.shape)

    model = XMotifs()
    model.fit_transform(data)
    rows, cols = model.rows_, model.columns_
    print(rows.shape, cols.shape)
    """

    model = Spectral()
    model.fit_transform(data)
    print(model.row_labels_)

    """new_params = {
        'nclus': 5,
        'targetpcc': 5,
        'fixed_row': 3,
        'fixed_col': 4,
    }
    model = CPB(**new_params)
    biclusters = model.fit_transform(data)
    score = consensus_score(
            biclusters, (rows[:, row_idx], columns[:, col_idx])
    )"""

    # NB: Moved to temp.
    # NOTE: Super slow. Writes nothing to outfile check. Check if needs to
    # change data or error with write output method.
    #model = CCS()
    #model.fit(data)
    #model.transform(data)
