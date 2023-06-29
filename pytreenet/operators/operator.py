from __future__ import annotations
from collections import UserDict
from typing import List, Union, Dict

import numpy as np

class Operator:
    """
    An operator hold the information what operation to apply to which node in a TTN.
    """

    def __init__(self, operator: Union[str, np.ndarray], node_identifiers: List[str]):
        self.operator = operator
        self.node_identifiers = node_identifiers

class NumericOperator(Operator):
    """
    An operator that holds the operator associated with it directly as an array.
    """

    def __init__(self, operator: np.ndarray, node_identifiers: List[str]):
        super().__init__(operator, node_identifiers)

class SymbolicOperator(Operator):
    """
    An operator that holds the operator associated with it only as a symbolic value.
    That operator has to be converted before actual use.
    """

    def __init__(self, operator: str, node_identifiers: List[str]):
        super().__init__(operator, node_identifiers)

class Term(UserDict):
    """
    Contains multiple single site matrices and the identifiers of the nodes they are applied
     to. It is basically a dictionary, where the keys are node identifiers and the values
     are the operators that should be applied to the node with that identifier.
    """

    def __init__(self, matrix_dict: Dict[str, Union[np.ndarray, str]] = None):
        if matrix_dict is None:
            matrix_dict = {}
        super().__init__(matrix_dict)

    @classmethod
    def from_operators(cls, operators: List[Operator]) -> Term:
        """
        Obtain a term from a list of operators.
        """
        term = Term()
        for operator in operators:
            assert len(operator.node_identifiers) == 1
            term[operator.node_identifiers[0]] = operator.operator
