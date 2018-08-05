# -*- coding: utf-8 -*-
#
# cluster.py
#

"""
Various scikit-learn compatible clustering algorithms.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


import sys
import logging
import rpy2.robjects.numpy2ri

import numpy as np
import rpy2.robjects as robjects

from io import StringIO
from rpy2.rinterface import RRuntimeError


rpy2.robjects.numpy2ri.activate()

import warnings
from rpy2.rinterface import RRuntimeWarning

verbose = False
if not verbose:
    warnings.filterwarnings('ignore', category=RRuntimeWarning)


"""Utility functions for dealing with R"""
#from rpy2 import interactive as r
#import rpy2.robjects as robjects

#try:
#    from rpy2.robjects.packages import importr
#except ImportError:
#    from rpy2.interactive import importr
#base = importr('base')
#def get_bioclite():
#    """
#    Note: requires an internet connection.
#
#    """
#    base.source("http://bioconductor.org/biocLite.R")
#    return robjects.r['biocLite']


class Biclusters(list):
    """A list of biclusters with extra attributes.

    model (): The ,pdeø algorithm that generated these biclusters
    args (): The model arguments.
    props (): The model properties.

    """
    def __init__(self, output, model=None, args=None, props=None):

        list.__init__(self, output)

        self.algorithm = model
        self.arguments = args
        self.properties = props


class Bicluster:
    """A bicluster representation.

    Args:
        rows: (list of ints): The bicluster row indices.
        cols: (list of ints): The bicluster column indices.
        data (numpy.ndarray): The dataset of which the bicluster originates.

    Returns:
        ():

    """

    def __init__(self, rows, cols, data=None):

        self.rows = rows
        self.cols = cols
        self.data = data

    @property
    def rows(self):

        return self._rows

    @rows.setter
    def rows(self, value):

        self._rows = value

    def __eq__(self, other):
        """Test two biclusters for equality.

        Two biclusters are equal if the have the same rows and
        columns, and they have the same object as their data
        member. It is not enough that their data be equal; it must be
        the same object.

        Args:
            other: A bicluster to compare.

        """

        # Compare bicluster elements.
        rows_equal = set(self.rows) == set(other.rows)
        cols_equal = set(self.cols) == set(other.cols)
        data_equal = id(self.data) == id(other.data)

        return rows_equal and cols_equal and data_equal

    def copy(self):
        """Returns a deep copy of the bicluster."""

        rows_cp, cols_cp = copy.copy(self.rows), copy.copy(self.cols)
        other = Bicluster(rows_cp, cols_cp, self.data)

        return other

    def array(self, rows=None, cols=None):
        """Get a numpy array bicluster from data, using the indices in
        bic_indices.

        Note: requires that this Bicluster's data member is not None.

        Args:
            rows: the row indices to use; defaults to this bicluster's rows.
            cols: the column indices; defaults to this bicuster's columns.

        """

        if self.data is None:
            array = None
        else:
            rows = self.rows if rows is None else rows
            cols = self.cols if cols is None else cols
            array = self.data.take(rows, axis=0).take(list(cols), axis=1)

        return array

    def filter_rows(self):
        """Returns the dataset with only the rows from this bicluster."""

        # NOTE: requires that this Bicluster's data member is not None.
        return self.array(cols=np.arange(self.data.shape[1]))

    def filter_cols(self):
        """Returns the dataset with only the columns from this bicluster."""

        # NOTE: requires that this Bicluster's data member is not None.
        return self.array(rows=np.arange(self.data.shape[0]))

    def intersection(self, other):
        """Determines the intersecting rows and columns of two biclusters.

        Args:
            other: a Bicluster

        Returns:
            (Bicluster): A Bicluster instance, with rows and columns common to
                both self and other.

            If other and self have the same data attribute, the
            returned Bicluster also has it; else its data attribute is
            None.

        """

        rows = set(self.rows).intersection(set(other.rows))
        cols = set(self.cols).intersection(set(other.cols))

        return Bicluster(rows, cols, _get_data_(self.data, other.data))

    def union(self, other):
        """Determines the union rows and columns of two biclusters.

        Args:
            other: a Bicluster

        Returns:
            A Bicluster instance with all rows and columns from both self
            and other.

            If other and self have the same data attribute, the
            returned Bicluster also has it; else its data attribute is
            None.

        """

        rows = set(self.rows).union(set(other.rows))
        cols = set(self.cols).union(set(other.cols))

        return Bicluster(rows, cols, _get_data_(self.data, other.data))

    def symmetric_difference(self, other):
        """Returns a new bicluster with only unique rows and columns,
        i.e. the inverse of the intersection.

        Args:
            other: a Bicluster

        Returns:
            A Bicluster instance with all rows and columns unique to either self
            or other.

            If other and self have the same data attribute, the
            returned Bicluster also has it; else its data attribute is
            None.

        """

        rows = set(self.rows).symmetric_difference(set(other.rows))
        cols = set(self.cols).symmetric_difference(set(other.cols))

        return Bicluster(rows, cols, _get_data_(self.data, other.data))

    def difference(self, other):
        """
        Returns the difference of two biclusters.

        Args:
            * other: a Bicluster

        Returns:
            A Bicluster instance with self's rows and columns, but not other's.

            If other and self have the same data attribute, the
            returned Bicluster also has it; else its data attribute is
            None.


        """

        rows = set(self.rows).difference(set(other.rows))
        cols = set(self.cols).difference(set(other.cols))

        return Bicluster(rows, cols)

    def issubset(self, other):
        """
        Returns True if self's rows and columns are both subsets of
        other's; else False.

        """

        return (set(self.rows).issubset(set(other.rows)) and
                set(self.cols).issubset(set(other.cols)))

    def shape(self):
        """Returns the number of rows and columns in this bicluster."""
        return len(self.rows), len(self.cols)

    def area(self):
        """Returns the number of elements in this bicluster."""
        return len(self.rows) * len(self.cols)

    def overlap(self, other):
        """Returns the ratio of the overlap area to self's total size."""
        return self.intersection(other).area() / self.area()

    def __repr__(self):
        return "Bicluster({0}, {1})".format(repr(self.rows), repr(self.cols))


class RBiclusterBase:

    GLOBAL_METHOD = 'biclust'

    def __init__(self):

        # NOTE:
        self._output = None

        self.rows_ = None
        self.columns_ = None
        self.biclusters_ = None

        self.row_labels = None
        self.column_labels = None

    @property
    def rows_(self):

        return self._rows

    @rows_.setter
    def rows_(self, value):

        if value is None:
            return
        else:
            self._rows = value
            #if isinstance(value, (list, np.ndarray, tuple)):
            #    self._rows = np.array(value)
            #else:
            #    raise ValueError('Bicluster rows should be <numpy.ndarray>, '
            #                     'not {}'.format(type(value)))

    @property
    def columns_(self):

        return self._columns

    @columns_.setter
    def columns_(self, value):

        if value is None:
            return
        else:
            print(type(value))
            self._columns = value
            #if isinstance(value, (list, np.ndarray, tuple)):
            #    self._columns = np.array(value)
            #else:
            #    raise ValueError('Bicluster columns should be <numpy.ndarray>, '
            #                     'not {}'.format(type(value)))

    @property
    def biclusters_(self):

        return self._biclusters

    @biclusters_.setter
    def biclusters_(self, value):

        if value is None:
            return
        else:
            if isinstance(value, (tuple, list)):
                self._biclusters = value
            else:
                raise ValueError('Biclusters should be <tuple>, not {}'
                                 ''.format(type(value)))


    def execute_r_function(self, method, data, params):
        """Executes the R function with given data and parameters."""

        # Run biclustering algorithm.
        robjects.r.library(self.GLOBAL_METHOD)
        function = robjects.r[self.GLOBAL_METHOD]
        #function = robjects.r[method]

        # Output is rpy2.robjects.methods.RS4 object
        try:
            self._output = function(data, method=method, **params)
        except:
            raise RuntimeError('Error in R with {}'.format(method))

    def fetch_biclusters(self, X):
        # Set rows and columns attributes.

        # Collect logical R matrices of biclusters rows and columns.
        _row_mat = np.array(self._output.do_slot('RowxNumber'), dtype=bool)
        _col_mat = np.array(self._output.do_slot('NumberxCol'), dtype=bool)

        row_mat, col_mat = self._check_cluster_coords(_row_mat, _col_mat)

        self.rows_, self.columns_ = row_mat.T, col_mat
        self.biclusters_ = (self.rows_, self.columns_)

        """
        # Collect biclusters
        rows, cols = [], []
        for col_num in range(row_mat.shape[1]):

            row_bools, col_bools = row_mat[:, num] != 0, col_mat[num, :] != 0
            # Value: True/False

            _rows = [num for num, value in enumerate(row_bools) if value]
            _cols = [num for num, value in enumerate(col_bools) if value]
            rows.append(np.array(_rows, dtype=bool))
            cols.append(np.array(_cols, dtype=bool))

        self.rows_, self.columns_ = rows, cols
        self.biclusters_ = (self.rows_, self.columns_)

        return self"""

    @staticmethod
    def _check_cluster_coords(row_mat, col_mat):
        # NOTE: Cheng and Church can sometimes return column matrix transpose.

        num_clusters = row_mat.shape[1]
        if num_clusters == col_mat.shape[0]:
            return row_mat, col_mat

        else:

            rows_cols = (row_mat.shape[0], col_mat.shape[0])
            if num_clusters == col_mat.shape[1] and rows_cols == data.shape:
                return row_mat, col_matrix.T
            else:
                raise RuntimeError('Invalid formatted array returned from {}'
                                   ''.format(model_name))


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

    def transform(self, X, y=None, **kwargs):

        # TODO: Check is fitted

        # Format R biclustering algorithm output to numpy.narray.
        self.fetch_biclusters(X)

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

    def transform(self, X, y=None, **kwargs):

        # TODO: Check is fitted

        # Format R biclustering algorithm output to numpy.narray.
        self.fetch_biclusters(X)

    def fit_transform(self, X, y=None, **kwargs):

        self.fit(X, y=y, **kwargs)

        return self.transform(X, y=y, **kwargs)


class Spectral(RBiclusterBase):

    params = {
        'normalization': 'log',
        'numberOfEigenvalues': 3,
        'minr': 2,
        'minc': 2,
        'withinVar': 1
    }

    def __init__(self, method='BCSpectral', **kwargs):

        super().__init__()

        self.method = method

        # TODO: Iterate through kwargs, if any kwargs has key same as params,
        # update value. Necessary for sklearn Grid Search

    def fit(self, X, y=None, **kwargs):

        # Run R biclustering algorithm.
        self.execute_r_function(self.method, X, self.params)

    def transform(self, X, y=None, **kwargs):

        self.format_biclusters()

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

    def transform(self, X, y=None, **kwargs):
        # QUESTION: Return biclusters objects?

        # TODO: Check is fitted

        # Format R biclustering algorithm output to numpy.narray.
        #self.format_biclusters()

        #return self.biclusters
        pass

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


# TODO: Replace with check_binary which raises error instead of returning None?
def path_to_program(program, path_var='PATH'):
    """Check if an executable is included in $PATH environment variable."""

    def is_exec(fpath):

        return os.path.exists(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exec(program):
            return program
        #else:
        #    return None
    else:
        for path in os.environ[path_var].split(os.pathsep):

            exe_file = os.path.join(path, program)
            if is_exec(exe_file):
                return exe_file
            #else:
            #    return None

    return None


class BinaryBiclusteringBase:
    """

    Attribtues:
        binary ():
        temp_dir ():

    """

    INPUT_FILE = 'data'
    OUTPUT_FILE = 'results'

    def __init__(self, binary=None, file_format='txt', temp=False):

        self.binary = binary
        self.file_format = file_format
        self.temp = temp

        # NOTE: Variables set with instance.
        self.path_dir = None
        self.path_data = None

    @property
    def binary(self):

        return self._binary

    @binary.setter
    def binary(self, value):

        if path_to_program(value) is None:
            raise PathError('Executable {0} not on $PATH'.format(value))
        else:
            self._binary = value

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

    def io_setup(self):
        # Set paths to temp dir holding the input and output data.

        if self.temp:
            self.path_dir = tempfile.mkdtemp()
        else:
            self.path_dir = '{0}_{1}'.format(self.binary, self.OUTPUT_FILE)

            if not os.path.exists(self.path_dir):
                os.makedirs(self.path_dir)

        self.path_data = os.path.join(
            self.path_dir, '{0}.{1}'.format(self.INPUT_FILE, self.file_format)
        )

    def io_teardown(self):

        # Cleanup temporary directory
        shutil.rmtree(self.path_dir)


class CPB(BinaryBiclusteringBase):
    """A wrapper for the CPB binary algorithm.

    Detects biclusters based on row correlations.

    Attribtues:
        biclusters (list):

    """

    # Name
    MODEL = 'cpb'

    # File format of binary output data
    FILE_FORMAT = 'txt'

    # Initial bicluster
    INIT_BINARY = 'init_bicluster'

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

    def __init__(self, temp=False, **kwargs):

        super().__init__(self.MODEL, self.FILE_FORMAT, temp)

        # TODO: Iterate through kwargs and update self.params

    @property
    def biclusters(self):

        return self._biclusters

    @biclusters.setter
    def biclusters(self, value):

        if value is None:
            return
        else:
            if isinstance(value, list):
                self._biclusters = value
            else:
                raise ValueError('biclusters should be <list>, not {}'
                                 ''.format(type(value)))

    def fit(self, X, y=None, **kwargs):
        """"""

        #_X = check_array(X)

        self.format_input(X)
        self.exec_clustering()

    def transform(self, X, y=None, **kwargs):

        self.biclusters = self.collect_output(X)

        if self.temp:
            self.io_teardown()

        return self.biclusters

    def fit_transform(self, X, y=None, **kwargs):

        self.fit(X)

        return self.transform(X)

    # NOTE: Replace wit hmy own `pandas` method? Produces same result?
    def format_input(self, X):

        # NOTE: Base method to set paths for temp dir and data.
        self.io_setup()

        self.params['nrows'], self.params['ncols'] = np.shape(X)
        with open(self.path_data, 'w') as outfile:

            outfile.write(
                '{0} {1}'.format(self.params['nrows'], self.params['ncols'])
            )
            for row in data:
                outfile.write('\n')
                # TODO: Convert map row data to str instead of list comp?
                outfile.write(' '.join([str(value) for value in row]))
            outfile.write('\n')
            outfile.close()

        return self

    def exec_clustering(self):

        # Create file holding results
        self._setup_exec()

        # Change to current working dir
        current_loc = os.getcwd()

        _path_data = os.path.abspath(self.path_data)
        try:
            os.chdir(self.path_dir)
            command = (
                '{model} {outfile} {initfile} 1 {targetpcc} {fixw}'.format(
                    model=self.MODEL, outfile=_path_data, **self.params
                )
            )
            subprocess.check_call(command.split())

        except OSError:
            raise PathError('Executable {0} not on $PATH'.format(value))

        finally:
            os.chdir(current_loc)

    def _setup_exec(self):

        _path_dir = os.path.abspath(self.path_dir)
        parent, _ = os.path.split(_path_dir)
        self.params['initfile'] = os.path.join(
            parent, 'initfile.{0}'.format(self.FILE_FORMAT)
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

    def collect_output(self, X):

        results_dir = os.path.abspath(self.path_dir)
        dir_files = os.listdir(results_dir)

        biclusters = []
        for _file in dir_files:

            _, stem = os.path.splitext(_file)
            if stem == '.out':

                output_file = os.path.join(results_dir, _file)
                biclusters.append(self.format_output(output_file, X))

        return biclusters

    def format_output(self, outfile, X):
        """Reads the bicluster in a single CPB output file."""
        rows, cols = [], []
        with open(outfile, 'r') as resfile:

            target = rows
            for line in resfile:
                if line[0] == 'R':
                    continue
                elif line[0] == 'C':
                    target = cols
                    continue
                else:
                    target.append(int(line.split()[0]))

            rows.sort(), cols.sort()

        return Bicluster(rows, cols, data=X)


class CCS(BinaryBiclusteringBase):
    """

    Args:
        thresh (float): A correlation threshold in range [0, 1]. Defaults to
            0.8.

    Attribtues:
        biclusters (list):

    """

    # Name
    MODEL = 'ccs'

    # File format of binary output data
    FILE_FORMAT = 'txt'

    # The standard dimensions of the input file
    FILE_DIMS = 'Genes/Conditions'

    params = {
        'thresh': 0.8,
        'bases': 1000,
        'overlap': 100.0,
        'out_format': 0
    }

    """Parameters:

    -m [1 - number of gene/rows in the data matrix]: Set the number of base gene that are to be considered for forming biclusters.
        Default value is 1000 or maximum number of genes when that is less than 1000.

    -g [0.0 - 100.0]: Minimum gene set overlap required for merging the overlapped biclusters.
        Default value is 100.0 for 100% overlap.

    -p [0/1]: Set the output format. Default is 0. 0 - Print output in 3 rows.

    Call syntax

        input_file = ./Results/Synthetic_data_results/Data/Data_Constant_100_1_bicluster.txt
        output_file = ./Results/Output_standard.txt

        ccs -t 0.9 -i input_file -o output_file -m 50 -p 1 -g 100.0

        bases, overlap, output

    """

    def __init__(self, temp=False):

        super().__init__(self.MODEL, self.FILE_FORMAT, temp)

    @property
    def biclusters(self):

        return self._biclusters

    @biclusters.setter
    def biclusters(self, value):

        if value is None:
            return
        else:
            if isinstance(value, list):
                self._biclusters = value
            else:
                raise ValueError('biclusters should be <list>, not {}'
                                 ''.format(type(value)))

    def fit(self, X, y=None, sep='\t', **kwargs):

        # TODO: Check if cuda is available in setup

        self.format_input(X, sep=sep, **kwargs)
        self.exec_clustering()

    def transform(self, X, y=None, **kwargs):

        self.biclusters = self.collect_output(X)

        #if self.temp:
        #    self.io_teardown()

        #return self.biclusters

    def fit_transform(self, X, y=None, **kwargs):

        self.fit(X)

        return self.transform(X)

    def format_input(self, X, sep='\t', **kwargs):

        # NOTE: Base method to set paths for temp dir and data.
        self.io_setup()

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
                ''.format(model=self.MODEL, **self.params)
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
            '{}_results.{}'.format(self.MODEL, self.FILE_FORMAT)
        )

    def collect_output(self, X):

        # NOTE: Need safe way to handle file names and directories
        with open('./ccs_results/ccs_results.txt', 'r') as results:
            rows, cols = [], []
            header = results.readline().split()

    def format_output(self):

        pass


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from sklearn.cluster.bicluster import SpectralBiclustering


    from sklearn.datasets import make_checkerboard
    from sklearn.datasets import samples_generator as sg

    from sklearn.metrics import consensus_score

    # Generate sklearn sample dataš
    target, rows, columns = make_checkerboard(
        shape=(10, 5), n_clusters=(4, 3), noise=10, shuffle=False,
        random_state=0
    )
    data, row_idx, col_idx = sg._shuffle(target, random_state=0)


    # Use sklearn model as reference to output structure
    ref_model = SpectralBiclustering()
    ref_model.fit(data)
    score = consensus_score(
        ref_model.biclusters_, (rows[:, row_idx], columns[:, col_idx])
    )
    print('sk score: ', score)

    model = ChengChurch()
    model.fit_transform(data)
    score = consensus_score(
        model.biclusters_, (rows[:, row_idx], columns[:, col_idx])
    )
    print('CC score: ', score)

    model = Plaid()
    model.fit_transform(data)
    score = consensus_score(
        model.biclusters_, (rows[:, row_idx], columns[:, col_idx])
    )
    print('Plaid score: ', score)

    # TODO: CUDA accel
    #model = CCS()
    #model.fit(data)
    ##model.transform(data)
