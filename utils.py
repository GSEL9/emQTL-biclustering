# -*- coding: utf-8 -*-
#
# utils.py
#

"""
the emQTL-mining utility module.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


class PathError(Exception):
    """Error raised if binary algorithm executable is not located on $PATH."""

    def __init__(self, message):

        super().__init__(message)
