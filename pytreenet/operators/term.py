from typing import Union, Dict, List
from collections import UserDict

import numpy as np

from .operator import Operator, NumericOperator

class Term(UserDict):
    """
    Contains multiple single site matrices and the identifiers of the nodes they are applied
     to. It is basically a dictionary, where the keys are node identifiers and the values
     are the operators that should be applied to the node with that identifier.

    Represents: \bigotimes_{site_ids} operator_{site_id}
    """

    def __init__(self, matrix_dict: Dict[str, Union[np.ndarray, str]] = None):
        if matrix_dict is None:
            matrix_dict = {}
        super().__init__(matrix_dict)

    @classmethod
    def from_operators(cls, operators: List[Operator]) -> Term:
        """
        Obtain a term from a list of single site operators.
        """
        term = Term()
        for operator in operators:
            assert len(operator.node_identifiers) == 1
            term[operator.node_identifiers[0]] = operator.operator

    def into_operator(self,
                      conversion_dict: Union[Dict[str, np.ndarray], None] = None) -> NumericOperator:
        """
        Computes the numeric value of a term, by calculating their tensor product.
        If the term contains symbolic operators, a conversion dictionary has to be provided.

        Args:
            conversion_dict (Union[Dict[str, np.ndarray], None], optional): A dictionaty
             that contains the numeric values of all symbolic operators in this term.
             Defaults to None.

        Returns:
            NumericOperator: Numeric operator with the value of the computed tensor product of
                all contained terms.
        """
        total_operator = 1
        for operator in self.values():
            if isinstance(operator, str):
                if conversion_dict is not None:
                    operator = conversion_dict[operator]
                else:
                    errstr = "If the term contains symbolic operators, there must be a dictionary for conversion!"
                    raise TypeError(errstr)
            total_operator = np.kron(total_operator, operator)
        return NumericOperator(total_operator, list(self.keys()))   
