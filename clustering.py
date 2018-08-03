# -*- coding: utf-8 -*-

# TODO: Merge with data-science tools.
# TODO: Specify version rpy2 >= 2.2.2 in dependencies.

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
        #'fit.model': 'y ~ m + a + b',
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


class BinaryBiclusteringBase:

    def make_temp_dir():

        pass

    def write_data():

        pass

    def hold_results():

        pass

    def read_back_results():

        pass

    def cleanup_temp_dirs():

        pass


class CBP(BinaryBiclusteringBase):
    """A wrapper for the CPB binary.

    Detects biclusters based on row correlations.

    """

    params = {
        'nclus': 2,
        'targetpcc': 0.9,
        'fixed_row': -1,
        'fixed_col': -1,
        'fixw': 0,
        'min_seed_rows': 3,
        'max_seed_rows': None
    }

    def __init__(self, **kwargs):

        # TODO: Iterate through kwargs and update self.params

        pass

    def fit(self, X, y=None, **kwargs):

        nrows, ncols = np.shape(X)

        if self.parmas['max_seed_rows'] is None:
            self.parmas['max_seed_rows'] = nrows

    def transform(self, X, y=None, **kwargs):

        pass

    def fit_transform(self, X, y=None, **kwargs):

        pass


class CCS(BinaryBiclusteringBase):

    def __init__(self):

        pass

    def fit(self, X, y=None, **kwargs):

        pass

    def transform(self, X, y=None, **kwargs):

        pass

    def fit_transform(self, X, y=None, **kwargs):

        pass


if __name__ == '__main__':

    pass
