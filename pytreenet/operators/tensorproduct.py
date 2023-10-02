from __future__ import annotations
from typing import Union, Dict, List
from numbers import Number
from collections import UserDict
from copy import deepcopy
from random import sample

import numpy as np
from scipy.linalg import expm

from .operator import Operator, NumericOperator
from ..util import crandn

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
                if symbolic:
                    identity = "I"
                else:
                    dim = node.open_dimension()
                    identity = np.eye(dim)
                padded_tp[node_id] = identity
        if len(node_ids) != 0:
            errstr = "Single site operators in this tensor product are applied to nodes that do not exist in the TTN!"
            raise KeyError(errstr)
        return padded_tp

    def into_operator(self,
                      conversion_dict: Union[Dict[str, np.ndarray], None] = None) -> NumericOperator:
        """
        Computes the numeric value of a tensor_product, by calculating their tensor product.
         If the tensor_product contains symbolic operators, a conversion dictionary has to be
         provided.

        Args:
            conversion_dict (Union[Dict[str, np.ndarray], None], optional): A dictionaty
             that contains the numeric values of all symbolic operators in this tensor_product.
             Defaults to None.

        Returns:
            NumericOperator: Numeric operator with the value of the computed tensor product of
                all contained tensor_products.
        """
        total_operator = 1
        for operator in self.values():
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
    
def random_tensor_product(reference_tree: TreeTensorNetwork,
                          num_factors: int = 1) -> TensorProduct:
    """
    Generates a random tensor product that is compatible with the reference
     TreeTensorNetwork.

    Args:
        reference_tree (TreeTensorNetwork): A reference TreeTensorNetwork.
         It provides the identifiers and dimensions for the operators in the
         tensor product.
        num_factors (int): The number of factors to use. The nodes to which they
         are applied are drawn randomly from all nodes.
    """
    if num_factors < 0:
        errstr = "The number of factors must be non-negative!"
        errstr =+ f"{num_factors} < 1!"
        raise ValueError(errstr)
    if num_factors > len(reference_tree.nodes):
        errstr = "There cannot be more factors than nodes in the tree!"
        errstr =+ f"{num_factors} > {len(reference_tree.nodes)}!"
        raise ValueError(errstr)

    random_tp = TensorProduct()
    chosen_nodes = sample(list(reference_tree.nodes.values()), num_factors)
    for node in chosen_nodes:
        factor = crandn(node.open_dimension())
        random_tp[node.identifier] = factor
    return random_tp
