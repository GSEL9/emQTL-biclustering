# -*- coding: utf-8 -*-
#
# utils.py
#

"""
the emQTL-mining utility module.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'


def check_parameter(name, param, target_dtype):
    """Perform parameter type checking."""

    if isinstance(param, target_dtype):
        return
    else:
        raise TypeError('Invalid type {} to parameter `{}`. Should be {}'
                        ''.format(type(param), name, target_dtype))


class PathError(Exception):
    """Error raised if binary algorithm executable is not located on $PATH."""

    def __init__(self, message):

        super().__init__(message)
