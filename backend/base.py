# -*- coding: utf-8 -*-
#
# base.py
#

"""
The emQTL-mining base module.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


import os
import shutil
import logging
import tempfile
import warnings
import rpy2.robjects.numpy2ri

import numpy as np
import rpy2.robjects as robjects

from rpy2.rinterface import RRuntimeError
from rpy2.rinterface import RRuntimeWarning

from sklearn.base import BaseEstimator, ClusterMixin


rpy2.robjects.numpy2ri.activate()


verbose = False
if not verbose:
    warnings.filterwarnings('ignore', category=RRuntimeWarning)


class RBiclusterBase(BaseEstimator, ClusterMixin):

    FUNCTION = 'biclust'

    # Limits to bicluster size
    MIN_ROWS = 2
    MIN_COLS = 2

    def __init__(self, random_state=0, **kwargs):

        self.random_state = random_state

        # Update parameters.
        for key in kwargs:
            if key in self.params.keys():
                self.params[key] = kwargs[key]

        self.set_params(**kwargs)

        # NOTE:
        self._r = None
        self._output = None

        self.rows_ = None
        self.columns_ = None
        self.biclusters_ = None

    def set_params(self, **kwargs):

        # Assign parameters to attributes.
        for key, value in self.params.items():
            # Add underscore instead of dot to attribute
            _key = key.replace('.', '_')
            setattr(self, _key, kwargs.get(_key, value))

        robjects.r('set.seed({})'.format(self.random_state))

        return self

    def get_params(self, deep=False):

        return self.params

    @property
    def rows_(self):

        return self._rows

    @rows_.setter
    def rows_(self, value):

        if value is None:
            return
        else:
            if isinstance(value, (list, np.ndarray, tuple)):
                self._rows = np.array(value)
            else:
                raise ValueError('Bicluster rows should be <numpy.ndarray>, '
                                 'not {}'.format(type(value)))

    @property
    def columns_(self):

        return self._columns

    @columns_.setter
    def columns_(self, value):

        if value is None:
            return
        else:
            if isinstance(value, (list, np.ndarray, tuple)):
                self._columns = np.array(value)
            else:
                raise ValueError('Bicluster columns should be <numpy.ndarray>, '
                                 'not {}'.format(type(value)))

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

    def _fit(self, model, X, params):

        # Run R biclustering algorithm.
        self.execute_r_function(model, X, params)

        # Format R biclustering algorithm output to numpy.narray.
        self.rows_, self.columns_ = self.fetch_biclusters(X)

        # Assign to attribute.
        self.biclusters_ = (self.rows_, self.columns_)

        return self

    def execute_r_function(self, method, data, params):
        """Executes the R function with given data and parameters."""

        # NOTE: Replace underscore with dot for valid R params.
        for key in params:
            params[key.replace('_', '.')] = params.pop(key)

        # Import function from R library.
        robjects.r.library(self.FUNCTION)

        # Function instance.
        function = robjects.r[self.FUNCTION]

        # Output is rpy2.robjects.methods.RS4 object
        try:
            self._output = function(data, method=method, **params)
        except RRuntimeError as r_error:
            logging.error(r_error)

        return self

    def fetch_biclusters(self, X):
        # Set rows and columns attributes.

        # Collect logical R matrices of biclusters rows and columns.
        row_mat_raw = np.array(self._output.do_slot('RowxNumber'), dtype=bool)
        col_mat_raw = np.array(self._output.do_slot('NumberxCol'), dtype=bool)

        # NOTE: Format biclusters before filtereing because affects filtering.
        row_mat_form, col_mat_form = self.format_biclusters(
            row_mat_raw, col_mat_raw, X
        )
        row_mat, col_mat = self.filter_bilusters(
            row_mat_form, col_mat_form
        )

        return row_mat, col_mat

    @staticmethod
    def format_biclusters(row_mat, col_mat, X):
        # Format row and column biclusters.

        num_rows, num_cols = np.shape(X)
        row_mat_rows, row_mat_cols = np.shape(row_mat)
        col_mat_rows, col_mat_cols = np.shape(col_mat)

        # Row clusters: (n_row_clusters, n_rows)
        if row_mat_rows == num_rows:
            # Col clusters: (n_column_clusters, n_columns)
            if col_mat_cols == num_cols:
                return row_mat.T, col_mat
            else:
                return row_mat.T, col_mat.T
        else:
            if col_mat_cols == num_cols:
                return row_mat, col_mat
            else:
                return row_mat, col_mat.T

    def filter_bilusters(self, row_mat, col_mat):

        row_mat_rows, row_mat_cols = np.shape(row_mat)
        col_mat_rows, col_mat_cols = np.shape(col_mat)

        # Filtering by size
        if row_mat_rows < self.MIN_ROWS and row_mat_cols < self.MIN_COLS:
            _row_mat = []
        else:
            _row_mat = row_mat

        if col_mat_rows < self.MIN_ROWS and col_mat_cols < self.MIN_COLS:
            _col_mat = []
        else:
            _col_mat = col_mat

        return _row_mat, _col_mat


if __name__ == '__main__':

    pass
