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

import numpy as np
import rpy2.robjects as robjects

from base import RBiclusterBase
from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.cluster.bicluster import SpectralCoclustering


class ChengChurch(RBiclusterBase):
    """A wrapper for R BCCC algorithm.

    Args:

    Returns:
        (list):

    """

    params = {
        'delta': 0.1,
        'alpha': 1.5,
        'number': 100
    }

    def __init__(self, method='BCCC', **kwargs):

        super().__init__()

        self.method = method

        # TODO: Iterate through kwargs, if any kwargs has key same as params,
        # update value. Necessary for sklearn Grid Search

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
    """A wrapper to R BCPlaid algorithm.

    Args:
        method (str): The R biclust function method name.

    Kwargs:
        cluster (str, {r, c, b}): Determines to cluster rows, columns or both.
            Defaults to both.
        model (str): The model formula to fit each layer. Defaults to linear
            model y ~ m + a + b.

    """

    # Algorithm parameters.
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

        # TODO: Iterate through kwargs and update self.params

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
    """

    Args:
        number (int): Number of bicluster to be found.
        ns (int): Number of seeds.
        nd (int): Number of determinants.
        sd (int): Size of discriminating set; generated for each seed.
        alpha (float): Scaling factor for column.

    """

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

        # TODO: Iterate through kwargs, if any kwargs has key same as params,
        # update value. Necessary for sklearn Grid Search

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
    """A scikit-learn wrapper for spectral biclustering algorithms.

    """

    def __init__(self, model='bi', **kwargs):

        if model == 'bi':
            self.model = SpectralBiclustering(**kwargs)
        elif model == 'co':
            self.model = SpectralCoclustering(**kwargs)
        else:
            raise ValueError('Invalid model: `{}` not among [`bi`, `co`]'
                             ''.format(model))

    def fit(self, X, y=None, **kwargs):

        self.model.fit(X, y=y, **kwargs)

        return self

    def transform(self, X, y=None, **kwargs):

        return self.model.biclusters_

    def fit_transform(self, X, y=None, **kwargs):

        self.fit(X, y=y, **kwargs)

        return self.transform(X, y=y, **kwargs)





"""Binary utility functions"""

import os
import shutil
import tempfile
import subprocess


class PathError(Exception):
    """Error raised if binary algorithm executable is not located on $PATH."""

    def __init__(self, message):

        super().__init__(message)


class BinaryBiclusteringBase:
    """

    Attribtues:
        binary ():
        temp_dir ():

    """

    INPUT_FILE = 'input'

    def __init__(self, model, file_format='txt', temp=False):

        self.model = model#self.check_on_path(model)
        self.file_format = file_format
        self.temp = temp

        # NOTE: Variables set with instance.
        self.path_dir = None
        self.path_data = None

    # ERROR: Something mysterious goingin on
    @staticmethod
    def check_on_path(model, path_var='PATH'):
        """Check if an executable is included in $PATH environment variable."""

        def _is_exec(fpath):
            #

            return os.path.exists(fpath) and os.access(fpath, os.X_OK)

        def _path_error(model):
            # Raises path error if executable not found on path.

            raise PathError('Executable {0} not on $PATH'.format(model))

        fpath, fname = os.path.split(model)
        if fpath:
            if _is_exec(model):
                return model
        else:
            for path in os.environ[path_var].split(os.pathsep):
                exe_file = os.path.join(path, model)
                if _is_exec(exe_file):
                    return model

    @property
    def path_dir(self):

        return self._path_dir

    @path_dir.setter
    def path_dir(self, value):

        self._path_dir = value

    @property
    def path_data(self):

        return self._path_data

    @path_data.setter
    def path_data(self, value):

        if value is None:
            return
        else:
            if isinstance(value, str):
                self._path_data = value
            else:
                raise ValueError('file path should be <str>, not {}'
                                 ''.format(type(value)))

    def setup_io(self):
        """Create dir holding formatted input data and raw output data."""

        if self.temp:
            self.path_dir = tempfile.mkdtemp()
        else:
            current_loc = os.getcwd()
            dir_name = '{0}_data'.format(self.model)

            self.path_dir = os.path.join(current_loc, dir_name)
            if not os.path.exists(self.path_dir):
                os.makedirs(self.path_dir)

        self.path_data = os.path.join(
            self.path_dir, '{0}.{1}'.format(self.INPUT_FILE, self.file_format)
        )

    def io_teardown_temp(self):

        # Cleanup temporary directory
        shutil.rmtree(self.path_dir)


class CPB(BinaryBiclusteringBase):
    """A wrapper for the CPB binary algorithm.

    Detects biclusters based on row correlations.

    Attribtues:
        rows_ ():
        columns_ ():
        biclusters_ ():

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

        # TODO: Iterate through kwargs and update self.params

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

        # Create file holding results.
        self._results_file()

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

    def _results_file(self):

        _path_dir = os.path.abspath(self.path_dir)
        parent, _ = os.path.split(_path_dir)
        self.params['initfile'] = os.path.join(
            parent, 'initfile.{0}'.format(self.file_format)
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

        results_dir = os.path.abspath(self.path_dir)
        dir_files = os.listdir(results_dir)

        num_biclusters = 0
        rows, cols = [], []
        for _file in dir_files:

            _, stem = os.path.splitext(_file)
            if stem == self.STEM_OUTPUT_FILE:

                output_file = os.path.join(results_dir, _file)
                row_idx, col_idx = self.read_output_file(output_file, X)
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

    def read_output_file(self, filename, X):
        """Reads the bicluster in a single CPB output file.

        Returns:
            (tuple): The bicluster row and column indices.

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


class CCS(BinaryBiclusteringBase):
    """

    Args:
        thresh (float): A correlation threshold in range [0, 1]. Defaults to
            0.8.

    Attribtues:
        biclusters (list):

    """

    # Name of the file containing the algorithm output.
    OUTPUT_FILE = 'output'

    # The standard dimensions of the input file
    FILE_DIMS = 'Genes/Conditions'

    params = {
        'thresh': 0.8,
        'bases': 1000,
        'overlap': 100.0,
        'out_format': 0
    }

    def __init__(self, model='ccs', file_format='txt', temp=False, **kwargs):

        super().__init__(model, file_format, temp)

        # TODO: Iterate through kwargs and update self.params

    def fit(self, X, y=None, sep='\t', **kwargs):

        #_X = check_array(X)

        # Call to base method to set paths for temp dir and data.
        self.setup_io()
        # Creates file of input data according to algorithm requirements.
        self.format_input(X, sep=sep)
        # Call to application
        self.exec_clustering()

    def transform(self, X, y=None, **kwargs):

        # TODO: Check is fitted

        self.fetch_biclusters(X)

        #if self.temp:
        #    self.io_teardown_temp()

        #return self.biclusters_

    def fit_transform(self, X, y=None, **kwargs):

        self.fit(X, y=y, **kwargs)

        return self.transform(X, y=y, **kwargs)

    def format_input(self, X, sep='\t', **kwargs):

        try:
            rows = kwargs['']
        except:
            rows = ['row{0}'.format(str(num)) for num in range(X.shape[0])]
        try:
            cols = kwargs['genes']
        except:
            cols = [
                '{0}col{1}'.format(sep, str(num)) for num in range(X.shape[1])
            ]
        with open(self.path_data, 'w') as outfile:
            outfile.write(self.FILE_DIMS)
            outfile.write(('').join(cols))
            outfile.write('\n')
            for num, row in enumerate(X):
                outfile.write('{0}{1}'.format(rows[num], sep))
                outfile.write(
                    ('').join(['{0}{1}'.format(sep, str(val)) for val in row])
                )
                outfile.write('\n')
            outfile.close()

        return self

    def exec_clustering(self):

        # Create file holding results
        self._setup_exec()

        # Change to current working dir
        current_loc = os.getcwd()

        try:
            os.chdir(self.path_dir)
            # ccs -t 0.9 -i input_file -o output_file -m 50 -p 1 -g 100.0
            command = (
                '{model} -t {thresh} -i {infile} -o {outfile} '
                '-m {bases} -p {out_format} -g {overlap} {out_format}'
                ''.format(model=self.model, **self.params)
            )
            subprocess.check_call(command.split())

        except OSError:
            raise PathError('Executable {0} not on $PATH'.format(value))

        finally:
            os.chdir(current_loc)

    def _setup_exec(self):

        self.params['infile'] = os.path.abspath(self.path_data)
        self.params['outfile'] = os.path.join(
            os.path.abspath(self.path_dir),
            '{0}.{1}'.format(self.OUTPUT_FILE, self.file_format)
        )

    def fetch_biclusters(self, X):

        results_file = os.path.join(
            self.path_dir, '{0}.{1}'.format(self.OUTPUT_FILE, self.file_format)
        )
        with open(results_file, 'r') as infile:

            header = infile.readline().split()
            print(header)

    def format_output(self):

        pass


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    from sklearn.datasets import make_checkerboard
    from sklearn.datasets import samples_generator as sg

    from sklearn.metrics import consensus_score

    # Generate sklearn sample data.
    # rows: array of shape (n_clusters, X.shape[0],)
    # columns: array of shape (n_clusters, X.shape[1],)
    target, rows, columns = make_checkerboard(
        shape=(12, 5), n_clusters=(4, 3), noise=10, shuffle=False,
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

    model = CPB()
    biclusters = model.fit_transform(data)
    score = consensus_score(
            biclusters, (rows[:, row_idx], columns[:, col_idx])
    )
    print(score)
    """

    # NB: Super slow. Writes nothing to outfile check. Check if needs to
    # change data or error with write output method.
    model = CCS()
    model.fit(data)
    model.transform(data)
