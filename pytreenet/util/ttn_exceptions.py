"""
Contains exceptions specific for TTNs.
"""
from __future__ import annotations
from typing import Union

class NoConnectionException(Exception):
    """
    Raised when tensors of two nodes in a tree are to be contracted, but the
    nodes are not actually connected.
    """
    pass

class NotCompatibleException(Exception):
    """
    Raised when compatibility of two TTN is checked, but not fulfilled.
    """
    pass

def positivity_check(value: Union[int,float], name: Union[str,None] = None,
                    errstr: Union[None,str] = None):
    """
    Check if a given value is positive.

    Args:
        value (Union[int,float]): The value to check.
        name (Union[str,None], optional): A name of the value that is checked,
         will be inserted into a default error string. Defaults to None.
        errstr (Union[None,str], optional): An individual error string that
         overrides the default error string. Defaults to None.
    """
    if value <= 0:
        if errstr is None:
            if name is None:
                name = "value"
            errstr = f"The {name} has to be positive!"
        raise ValueError(errstr)

def non_negativity_check(value: Union[int,float], name: Union[str,None] = None,
                         errstr: Union[None,str] = None):
    """
    Check if a given value is non-negative, i.e >=0.

    Args:
        value (Union[int,float]): The value to check.
        name (Union[str,None], optional): A name of the value that is checked,
         will be inserted into a default error string. Defaults to None.
        errstr (Union[None,str], optional): An individual error string that
         overrides the default error string. Defaults to None.
    """
    if value < 0:
        if errstr is None:
            if name is None:
                name = "value"
            errstr = f"The {name} has to be non-negative!"
        raise ValueError(errstr)
