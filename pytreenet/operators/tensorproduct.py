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

    def add_suffix(self,
                    suffix: str
                    ) -> TensorProduct:
        """
        Adds a suffix to every node identifier in the tensor product.

        Args:
            suffix (str): The suffix to add.

        Returns:
            TensorProduct: A new tensor product with the suffix added.

        """
        new_dict = {node_id + suffix: operator for node_id, operator in self.items()}
        return TensorProduct(new_dict)

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

    def _local_action(self,
                      action: callable,
                      invariance_dict: Union[Dict[str, bool], None] = None,
                      id_suffix: str = "_id"
                      ) -> TensorProduct:
        """
        Performs an action on the single site operators making up the tensor
        product.

        Args:
            action (callable): The action to be performed on the operators.
            invariance_dict (Union[Dict[str, bool], None], optional): A
                dictionary mapping the identifiers to whether they are invariant
                under the action. Defaults to None.
            id_suffix (str, optional): The suffix for the identity operators.
                Defaults to "_id".
        
        Returns:
            TensorProduct: The tensor product with the action applied to the
                operators.
        
        """
        new_tp = TensorProduct()
        for identifier, operator in self.items():
            if isinstance(operator, str):
                if invariance_dict is not None and invariance_dict[operator]:
                    new_tp[identifier] = operator
                else:
                    new_tp[identifier] = operator + id_suffix
            else:
                new_tp[identifier] = action(operator)
        return new_tp

    def transpose(self,
                  sym_dict: Union[Dict[str, bool],None] = None,
                  transpose_suffix: str = "_T"
                  ) -> TensorProduct:
        """
        Transposes a tensor product.

        Args:
            sym_dict (Union[Dict[str, bool],None], optional): A dictionary
                mapping the identifiers to whether they are symmetric and
                thus needn't be transposed. Defaults to None.
            transpose_suffix (str, optional): The suffix for the transposed
                symbolic operators. Defaults to "_T".
        
        Returns:
            TebsorProduct: The a new tensor product with the transposed
                operators.

        """
        return self._local_action(np.transpose,
                                  invariance_dict=sym_dict,
                                  id_suffix=transpose_suffix)

    def conjugate(self,
                  real_dict: Union[Dict[str, bool],None] = None,
                  conjugate_suffix: str = "_conj"
                  ) -> TensorProduct:
        """
        Conjugates a tensor product.

        Args:
            real_dict (Union[Dict[str, bool],None], optional): A dictionary
                mapping the identifiers to whether they are real and thus
                needn't be conjugated. Defaults to None.
            conjugate_suffix (str, optional): The suffix for the conjugated
                symbolic operators. Defaults to "_conj".

        Returns:
            TensorProduct: A new tensor product with the conjugated operators.
        
        """
        return self._local_action(np.conjugate,
                                  invariance_dict=real_dict,
                                  id_suffix=conjugate_suffix)

    def conjugate_transpose(self,
                            herm_dict: Union[Dict[str, bool], None] = None,
                            herm_suffix: str = "_H"
                            ) -> TensorProduct:
        """
        Conjugate transposes a tensor product.

        Args:
            other (TensorProduct): The other tensor product.
            herm_dict (Union[Dict[str, bool], None], optional): A dictionary
                mapping the identifiers to whether they are hermitian and thus
                needn't be conjugate transposed. Defaults to None.
            herm_suffix (str, optional): The suffix for the hermitian
                operators. Defaults to "_herm".

        Returns:
            TensorProduct: The conjugate transposed tensor product.
        
        """
        return self._local_action(conjugate_transpose,
                                  invariance_dict=herm_dict,
                                  id_suffix=herm_suffix)

    def otimes(self,
               other: TensorProduct,
               to_copy: bool = True
               ) -> TensorProduct:
        """
        Computes the tensor product of two tensor products.

        Args:
            other (TensorProduct): The other tensor product.
            to_copy (bool, optional): If True, the original tensor products
                are not changed. Defaults to True.

        Returns:
            TensorProduct: The tensor product of the two tensor products.
        
        """
        if to_copy:
            new_tp = deepcopy(self)
        else:
            new_tp = self
        for identifier, operator in other.items():
            if identifier in new_tp:
                errstr = "The tensor products have a common identifier!"
                raise ValueError(errstr)
            new_tp[identifier] = operator
        return new_tp

    def multiply(self,
                 other: TensorProduct,
                 identity_dict: Union[Dict[str, bool], None] = None,
                 conversion_dict: Union[Dict[str, np.ndarray], None] = None,
                 multi_inset: str = "_mult_"
                 ) -> TensorProduct:
        """
        Multiplies two tensor products.

        The multiplication is the local matrix multiplication and thus not 
        commutative.

        Args:
            other (TensorProduct): The other tensor product.
            identity_dict (Union[Dict[str, np.ndarray], None], optional): A
                dictionary mapping operator labels to whether they are the
                identity. Defaults to None.
            conversion_dict (Union[Dict[str, np.ndarray], None], optional): A
                dictionary mapping symbolic operators to their numerical
                values. If provided, it will be updated accordingly with the
                new operators.
            multi_inset (str, optional): The string added in between the two
                original symbolic operators. Defaults to "_mult_".

        Returns:
            TensorProduct: The product of the two tensor products.
        
        """
        id_dict_exists = identity_dict is not None
        new_tp = TensorProduct()
        for node_id, operator in self.items():
            if node_id in other:
                other_operator = other[node_id]
                if id_dict_exists and identity_dict[operator]:
                    # In this case no actual multiplication is needed
                    new_tp[node_id] = other_operator
                elif id_dict_exists and identity_dict[other_operator]:
                    # In this case no actual multiplication is needed
                    new_tp[node_id] = operator
                else:
                    # Now neither operator is an identity
                    new_tp[node_id] = operator + multi_inset + other_operator
                    if conversion_dict is not None:
                        op_val = conversion_dict[operator]
                        other_op_val = conversion_dict[other_operator]
                        new_op_val = op_val @ other_op_val
                        conversion_dict[new_tp[node_id]] = new_op_val
            else:
                # In this case th other operator is implicitely the identity
                new_tp[node_id] = operator
        for node_id, operator in other.items():
            if node_id not in new_tp:
                # In this case the first operator is implicitely the identity
                new_tp[node_id] = operator
        return new_tp

    def __matmul__(self, other: TensorProduct) -> TensorProduct:
        """
        Overloads the matrix multiplication operator.

        Args:
            other (TensorProduct): The other tensor product.

        Returns:
            TensorProduct: The product of the two tensor products.
        
        """
        return self.multiply(other)

def conjugate_transpose(matrix: np.ndarray
                        ) -> np.ndarray:
    """
    Conjugate transposes a matrix.

    Args:
        matrix (np.ndarray): The matrix to be conjugate transposed.

    """
    return matrix.conj().T
