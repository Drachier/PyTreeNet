"""
Contains exceptions specific for TTNs.
"""

class NoConnectionException(Exception):
    """
    Raised when tensors of two nodes in a tree are to be contracted, but the
    nodes are not actually connected.
    """
    pass