"""
The tensor product class is used to represent the tensor product of multiple
single site operators.  It is basically a dictionary mapping a node to an
operator that should be applied to it. This works for both symbolic and numeric
operators. In general tensor products form the basis of many general operators
and are used frequently in conjuction with tensor networks. In general the
tensor product is defined as:

.. math:: \\bigotimes_{i} A^{[i]}

where :math:`A^{[i]}` is an operator acting on the i-th site. The operator can
be any appropriately sized operator and does not need to be unitary or
hermitian. In many cases sites on which only the identity is applied are left
out of the tensor product. For example for three nodes we would have:

.. math:: A^{[1]} \\otimes A^{[2]} \\otimes I^{[3]} = A^{[1]} \\otimes A^{[2]}

where :math:`I^{[3]}` is the identity operator acting on the third site.

"""
from __future__ import annotations
from typing import Union, Dict, List
from numbers import Number
from collections import UserDict
from copy import deepcopy

import numpy as np
from scipy.linalg import expm

from .operator import NumericOperator
from ..core.ttn import TreeTensorNetwork

class TensorProduct(UserDict):
    """
    A class representing a tensor product of multiple single site operators.

    .. math:: \\bigotimes_{i} A^{[i]}

    where :math:`A^{[i]}` is an operator acting on the i-th site.

    The tensor product is represented as a dictionary mapping a node to an operator.
    """

    def __init__(self,
                 matrix_dict: Union[Dict[str, Union[np.ndarray, str]], None] = None):
        """
        Initialises a tensor product object.

        Args:
            matrix_dict (Union[Dict[str, Union[np.ndarray, str]], None], optional):
                A dictionary mapping a node to an operator. The operator can be
                either a numpy array or a string representing a symbolic
                operator.
             
        """
        if matrix_dict is None:
            matrix_dict = {}
        super().__init__(matrix_dict)

    @classmethod
    def from_operators(cls, operators: List[NumericOperator]) -> TensorProduct:
        """
        Obtain a tensor_product from multiple single site operators.
        
        Args:
            operators (List[NumericOperator]): A list of single site operators.
        
        Returns:
            TensorProduct: The tensor product of the operators.
        """
        tensor_product = TensorProduct()
        for operator in operators:
            assert len(operator.node_identifiers) == 1
            tensor_product[operator.node_identifiers[0]] = operator.operator
        return tensor_product

    def allclose(self, other: TensorProduct) -> bool:
        """
        Returns, whether the two tensor products are close to each other.

        This means that the operators corresponding to each identifier are
        close.
        """
        if len(self) != len(other):
            # To avoid subdicts
            return False
        for identifier, operator in self.items():
            if identifier not in other:
                # Avoids KeyError
                return False
            other_op = other[identifier]
            if not np.allclose(operator, other_op):
                return False
        return True

    def pad_with_identities(self,
                            ttn: TreeTensorNetwork,
                            symbolic: bool = False) -> TensorProduct:
        """
        Pads this tensor product with identities.

        All sites that are in the TTN but not in the tensor product are padded
        with an appropriately sized identity operator.

        Args:
            ttn (TreeTensorNetworkState): The TTN to be considered for the
                padding.
            symbolic (bool): True adds `"I{size}"` instead of the appropriately
                sized numpy array.

        Returns:
            TensorProduct: The padded tensor product.

        Raises:
            KeyError: Raised if this tensor product contains single site
                operators acting on sites that do not exist in the TTN.
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
        Computes the numeric value of a tensor product.

        The actual numeric operator is obtained by performing the tensor
        product of all single site operators. If the tensor product contains
        symbolic operators, a conversion dictionary has to be provided.

        Args:
            conversion_dict (Union[Dict[str, np.ndarray], None], optional): A
                dictionary that contains the numeric values of all symbolic
                operators in this tensor_product. Defaults to None.
            order (Union[List[str],None], optional): Give a specific order in
                which the factors should be multiplied. This can make a
                difference, since the tensor product is not commutative.
                Defaults to None.

        Returns:
            NumericOperator: Numeric operator with the value of the computed
                tensor product of all contained tensor_products.
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
        Compute the exponential of a tensor product.
        
        Notably it will not be a tensor product anymore but a general operator.

        Args:
            factor (Number, optional): A factor that is multiplied to the
                exponent. Defaults to 1.

        Returns:
            NumericOperator: The exponentiated term. The identifiers in the
                operator are in the same order as the tensor product keys. 
        """
        total_operator = self.into_operator()
        exponentiated_operator = expm(factor * total_operator.operator)
        return  NumericOperator(exponentiated_operator,
                                    total_operator.node_identifiers)
