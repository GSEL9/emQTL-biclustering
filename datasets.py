# -*- coding: utf-8 -*-
#
# testsets.py
#

"""
Synthetic test data generators.

Test data is generated in resemblance to the experimental datasets in order to
evaluate the performance of the applied biclustering algorithms.

The goal is to assess the algorithms capability of restoring the original test
clusters. The test clusters are generated in similar proportions that was found
to be biologically relevant in the experimental dataset.

Through exploration, the complete experimental dataset appear to exhibit a
checkerboard structure. The reduced version of the same data contains the
same general structure, but with with 1e7 fewer datapoints. This results in a
sparse and a dense version of each dataset.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


def gen_test_pvalues(sparse=True):
    # If sparse: similar to selected data, if dense: similar to original data.

    pass


def gen_test_cpp(sparse=True):
    # If sparse: similar to selected data, if dense: similar to original data.

    pass


if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt

    # Start of with a checkerboard and modify the data
