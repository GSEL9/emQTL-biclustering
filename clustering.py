# -*- coding: utf-8 -*-

# TODO: Merge with data-science tools.
# TODO: Specify version rpy2 >= 2.2.2 in dependencies.
# TODO: Include verbose to R algs to supress warnings
# TODO: Include BiMax

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


class BiclusterList(list):
    """A list of biclusters with extra attributes.

    model (): The ,pdeø algorithm that generated these biclusters
    args (): The model arguments.
    props (): The model properties.

    """
    def __init__(self, itr, model=None, args=None, props=None):

        list.__init__(self, itr)

        self.algorithm = model
        self.arguments = args
        self.properties = props


class Bicluster:
    """A representation of a bicluster.

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

        self.result = None

    @property
    def result(self):

        return self._result

    @result.setter
    def result(self, value):

        self._result = value

    def execute_algorithm(self, method, data, params):
        """Convenience function for the various methods in the R `biclust`
        function.

        Performs biclustering on the dataset and returns a set of biclusters.

        """

        # Run biclustering algorithm.
        robjects.r.library(self.GLOBAL_METHOD)
        biclust = robjects.r[self.GLOBAL_METHOD]
        function = robjects.r[method]

        # NB: Assumes no biclusters were found if error.
        try:
            self.result = biclust(data, method=method, **params)
        except:
            raise RuntimeError('Error in R with {}'.format(method))

    # TODO: Should not pass self.result as input to method.
    def struct_clusters(self, result):

        # Collect logical R matrices of biclusters rows and columns.
        _row_mat = np.array(result.do_slot('RowxNumber'), dtype=int)
        _col_mat = np.array(result.do_slot('NumberxCol'), dtype=int)

        row_mat, col_mat = self._check_cluster_rows_cols(_row_mat, _col_mat)

        # Collect biclusters
        biclusters, num_clusters = [], row_mat.shape[1]
        for num in range(num_clusters):

            # Value: True/False
            row_bools, col_bools = row_mat[:, num] != 0, col_mat[num, :] != 0
            rows = [num for num, value in enumerate(row_bools) if value]
            cols = [num for num, value in enumerate(col_bools) if value]

            biclusters.append(Bicluster(rows, cols, data=data))

        return biclusters

    @staticmethod
    def _check_cluster_rows_cols(row_mat, col_mat):
        # NOTE: Cheng and Church can sometimes return column matrix transpose.

        num_clusters = row_mat.shape[1]
        if not num_clusters == col_mat.shape[0]:

            rows_cols = (row_mat.shape[0], col_mat.shape[0])
            if num_clusters == col_mat.shape[1] and rows_cols == data.shape:
                return row_mat, col_matrix.T
            else:
                raise RuntimeError('Invalid formatted array returned from {}'
                                   ''.format(model_name))
        else:
            return row_mat, col_mat


class ChengChurch(RBiclusterBase):
    """A wrapper for R BCCC algorithm.

    Args:

    Returns:
        (list):

    """

    METHOD = 'BCCC'

    def __init__(self, max_score=1.0, scale=1.5, num_clusters=100):

        super().__init__()

        self.params = {
            'delta': max_score,
            'alpha': scale,
            'number': num_clusters
        }

    def fit(self, X, y=None, **kwargs):

        # Run R biclustering algorithm.
        self.execute_algorithm(self.METHOD, X, self.params)

    def transform(self, X, y=None, **kwargs):

        # TODO: Should not pass self.result as input to method.
        return self.struct_clusters(self.result)

    def fit_transform(self, X, y=None, **kwargs):

        self.fit(X, y=y, **kwargs)

        return self.transform(X, y=y, **kwargs)

# ERROR: Passing fit.model produces error `$ operator is invalid for atomic vectors`
class Plaid(RBiclusterBase):
    """

    Kwargs:
        cluster (str, {r, c, b}): Determines to cluster rows, columns or both.
            Defaults to both.
        model (str): The model formula to fit each layer. Defaults to linear
            model y ~ m + a + b.

    """

    METHOD = 'BCPlaid'

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

    def __init__(self, **kwargs):

        super().__init__()

        # TODO: Iterate through kwargs and update self.params

    def fit(self, X, y=None, **kwargs):

        # Run R biclustering algorithm.
        self.execute_algorithm(self.METHOD, X, self.params)

    def transform(self, X, y=None, **kwargs):

        # TODO: Should not pass self.result as input to method.
        biclusters = self.struct_clusters(self.result)

        if len(biclusters) == 1:
            if biclusters[0].shape() == (1, 1):
                return []
            else:
                return biclusters
        else:
            return biclusters

    def fit_transform(self, X, y=None, **kwargs):

        self.fit(X, y=y, **kwargs)

        return self.transform(X, y=y, **kwargs)


class Spectral(RBiclusterBase):

    METHOD = 'BCSpectral'

    params = {
        'normalization': 'log',
        'numberOfEigenvalues': 3,
        'minr': 2,
        'minc': 2,
        'withinVar': 1
    }

    def __init__(self, **kwargs):

        super().__init__()

        # TODO: Iterate through kwargs and update self.params

    def fit(self, X, y=None, **kwargs):

        # Run R biclustering algorithm.
        self.execute_algorithm(self.METHOD, X, self.params)

    def transform(self, X, y=None, **kwargs):

        #stdout_save = sys.stdout
        #sys.stdout = stdout = StringIO()

        # TODO: Should not pass self.result as input to method.
        biclusters = self.struct_clusters(self.result)

        #sys.stdout = stdout_save
        #stdout_val = stdout.getvalue()

        #if stdout_val.find('No biclusters found') >= 0:
        #    return []
        #else:
        #    return biclusters

        return biclusters

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

    METHOD = 'BCXmotifs'

    # TODO: Use more informative names in constructor
    def __init__(self, number=1, ns=200, nd=100, sd=5, alpha=0.05):

        super().__init__()

        self.params = {
            'number': number,
            'ns': ns,
            'nd': nd,
            'sd': sd,
            'alpha': alpha
        }

    def fit(self, X, y=None, **kwargs):

        X_discrete = X.astype(int)

        # Run R biclustering algorithm.
        self.execute_algorithm(self.METHOD, X_discrete, self.params)

    def transform(self, X, y=None, **kwargs):

        # TODO: Should not pass self.result as input to method.
        return self.struct_clusters(self.result)

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
            self.path_dir = '{0}_data'.format(self.binary)

            if not os.path.exists(self.path_dir):
                os.makedirs(self.path_dir)

        self.path_data = os.path.join(
            self.path_dir, 'data.{}'.format(self.file_format)
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

        if self.temp:
            self.io_teardown()

        return self.biclusters

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

    def collect_output(self):

        pass

    def format_output(self):

        pass


if __name__ == '__main__':

    pass
