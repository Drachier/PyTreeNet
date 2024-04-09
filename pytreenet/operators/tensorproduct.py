from __future__ import annotations
from typing import Union, Dict, List
from numbers import Number
from collections import UserDict
from copy import deepcopy

import numpy as np
from scipy.linalg import expm

from .operator import Operator, NumericOperator

class TensorProduct(UserDict):
    """
    Contains multiple single site matrices and the identifiers of the nodes they are applied
     to. It is basically a dictionary, where the keys are node identifiers and the values
     are the operators that should be applied to the node with that identifier.

    Represents: bigotimes_{site_ids} operator_{site_id}
    """

    def __init__(self, matrix_dict: Union[Dict[str, Union[np.ndarray, str]], None] = None):
        if matrix_dict is None:
            matrix_dict = {}
        super().__init__(matrix_dict)

    @classmethod
    def from_operators(cls, operators: List[Operator]) -> TensorProduct:
        """
        Obtain a tensor_product from a list of single site operators.
        """
        tensor_product = TensorProduct()
        for operator in operators:
            assert len(operator.node_identifiers) == 1
            tensor_product[operator.node_identifiers[0]] = operator.operator
        return tensor_product

    def pad_with_identities(self, ttn: TreeTensorNetwork,
                                symbolic: bool = False) -> TensorProduct:
        """
        Pads this tensor product with identites for sites, which are not acted upon
         non-trivially. This means any node in 'ttn' that has no operator associated
         to it in this tensor product will be added to this tensot product with an
         appropriately sized identity acting upon it.

        Args:
            ttn (TreeTensorNetworkState): The TTN to be considered for the padding.
            symbolic (bool): True adds `"I"` instead of the appropriately sized numpy
                array.

        Returns:
            TensorProduct: The padded tensor product.

        Raises:
            KeyError: Raised if this tensor product contains single site operators
             acting on sites that do not exist in the TTN.
        """
        node_ids = list(self.keys())
        padded_tp = deepcopy(self)
        for node_id, node in ttn.nodes.items():
            if node_id in node_ids:
                node_ids.remove(node_id)
            else:
                dim = node.open_dimension()
                if symbolic:
                    identity = "I" + str(dim)
                else:
                    identity = np.eye(dim)
                padded_tp[node_id] = identity
        if len(node_ids) != 0:
            errstr = "Single site operators in this tensor product are applied to nodes that do not exist in the TTN!"
            raise KeyError(errstr)
        return padded_tp

    def into_operator(self,
                      conversion_dict: Union[Dict[str, np.ndarray], None] = None,
                      order: Union[List[str],None] = None) -> NumericOperator:
        """
        Computes the numeric value of a tensor_product, by calculating their tensor product.
         If the tensor_product contains symbolic operators, a conversion dictionary has to be
         provided.

        Args:
            conversion_dict (Union[Dict[str, np.ndarray], None], optional): A dictionaty
             that contains the numeric values of all symbolic operators in this tensor_product.
             Defaults to None.
            order (Union[List[str],None], optional): Give a specific order in which
             the factors should be multiplied. This can make a difference, since the
             tensor product is not commutative. Defaults to None.

        Returns:
            NumericOperator: Numeric operator with the value of the computed tensor product of
                all contained tensor_products.
        """
        if order is None:
            order = list(self.keys())
        total_operator = 1
        for identifier in order:
            operator = self[identifier]
            if isinstance(operator, str):
                if conversion_dict is not None:
                    operator = conversion_dict[operator]
                else:
                    errstr = "If the tensor_product contains symbolic operators, there must be a dictionary for conversion!"
                    raise TypeError(errstr)
            total_operator = np.kron(total_operator, operator)
        return NumericOperator(total_operator, list(self.keys()))

    def exp(self, factor: Number = 1) -> NumericOperator:
        """
        Compute the exponential of a tensor product. Notably it will not be a tensor
         product anymore but a general operator.

        Args:
            factor (Number, optional): A factor that is multiplied to the exponent.
             Defaults to 1.

        Returns:
            NumericOperator: The exponentiated term. The identifiers in the operator
             are in the same order as the tensor product keys. 
        """
        total_operator = self.into_operator()
        exponentiated_operator = expm(factor * total_operator.operator)
        return  NumericOperator(exponentiated_operator,
                                    total_operator.node_identifiers)
