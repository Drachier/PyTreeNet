"""
Utility functions used for multiple special tensor network classes.
"""

from pytreenet.util.ttn_exceptions import (positivity_check,
                                           non_negativity_check)

def check_product_state_parameters(state_value: int,
                                   dimension: int):
    """
    Checks the validity of parameters used to generate product states.

    Args:
        state_value (int): The value of the state.
        dimension (int): The dimension of the state.
    """
    positivity_check(dimension, "dimension")
    if state_value >= dimension:
        errstr = "State value cannot be larger than the state's dimension!"
        raise ValueError(errstr)
    non_negativity_check(state_value, "state value")
