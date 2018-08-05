# -*- coding: utf-8 -*-
#
# base.py
#

"""
The emQTL-mining base module.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'

import logging
import warnings
import rpy2.robjects.numpy2ri

import numpy as np
import rpy2.robjects as robjects

from rpy2.rinterface import RRuntimeError
from rpy2.rinterface import RRuntimeWarning

rpy2.robjects.numpy2ri.activate()

verbose = False
if not verbose:
    warnings.filterwarnings('ignore', category=RRuntimeWarning)


class RBiclusterBase:

    FUNCTION = 'biclust'

    def __init__(self):

        # NOTE:
        self._output = None

        self.rows_ = None
        self.columns_ = None
        self.biclusters_ = None

        self.row_labels_ = None
        self.column_labels_ = None

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


    def execute_r_function(self, method, data, params):
        """Executes the R function with given data and parameters."""

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
        _row_mat = np.array(self._output.do_slot('RowxNumber'), dtype=bool)
        _col_mat = np.array(self._output.do_slot('NumberxCol'), dtype=bool)

        row_mat, col_mat = self._check_cluster_coords(_row_mat, _col_mat, X)

        self.rows_, self.columns_ = row_mat.T, col_mat
        self.biclusters_ = (self.rows_, self.columns_)

    @staticmethod
    def _check_cluster_coords(row_mat, col_mat, X):
        # NOTE: Cheng and Church can sometimes return column matrix transpose.

        num_clusters = row_mat.shape[1]
        if num_clusters == col_mat.shape[0]:
            return row_mat, col_mat

        else:

            rows_cols = (row_mat.shape[0], col_mat.shape[0])
            if num_clusters == col_mat.shape[1] and rows_cols == X.shape:
                return row_mat, col_mat.T
            else:
                raise RuntimeError('Invalid formatted array returned from {}'
                                   ''.format(model_name))
