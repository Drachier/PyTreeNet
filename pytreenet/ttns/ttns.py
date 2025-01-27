"""
Provides a class representing tree tensor network states (TTNS)
"""
from __future__ import annotations
from typing import Union
from copy import deepcopy

import numpy as np
from numpy import sqrt

from ..core.ttn import TreeTensorNetwork
from ..ttno import TTNO
from ..operators.tensorproduct import TensorProduct
from ..contractions.state_state_contraction import contract_two_ttns
from ..contractions.state_operator_contraction import expectation_value

class TreeTensorNetworkState(TreeTensorNetwork):
    """
    A class representing a tree tensor network state (TTNS).

    A TTNS is a TTN representing a quantum state. This means that every node
    has exactly one physical leg. That leg can be trivial, i.e. of dimension 1.
    """

    def scalar_product(self,
                       other: Union[TreeTensorNetworkState,None] = None,
                       use_orthogonal_center: bool = True
                       ) -> complex:
        """
        Computes the scalar product of this TTNS.

        Args:
            other (Union[TreeTensorNetworkState,None], optional): The other
                TTNS to compute the scalar product with. If None, the scalar
                product is computed with itself. Defaults to None.
            use_orthogonal_center (bool, optional): Whether to use the current
                orthogonalization center to compute  the norm. This usually
                speeds up the computation. Defaults to True.

        Returns:
            complex: The resulting scalar product <TTNS|Other>

        """
        if other is None:
            if self.orthogonality_center_id is not None and use_orthogonal_center:
                tensor = self.tensors[self.orthogonality_center_id]
                tensor_conj = tensor.conj()
                legs = tuple(range(tensor.ndim))
                return complex(np.tensordot(tensor, tensor_conj, axes=(legs,legs)))
            # Very inefficient, fix later without copy
            other = self
        other_conj = other.conjugate()
        return contract_two_ttns(self, other_conj)

    def norm(self) -> float:
        """
        Compute the norm of the TTNS.

        Returns:
            float: The norm of the state.
        """
        scal_prod = self.scalar_product()
        assert scal_prod.imag == 0
        return sqrt(scal_prod.real)

    def normalise(self) -> float:
        """
        Normalises the MPS in place.

        Returns:
            float: The norm of the state before normalisation.
        """
        norm = self.norm()
        if self.orthogonality_center_id is not None:
            # Avoids destroying the orthogonality center
            self.tensors[self.orthogonality_center_id] /= norm
        else:
            self.tensors[self.root_id] /= norm
        return norm

    def single_site_operator_expectation_value(self, node_id: str,
                                               operator: np.ndarray) -> complex:
        """
        The expectation value with regards to a single-site operator.

        The single-site operator acts on the specified node.

        Args:
            node_id (str): The identifier of the node, the operator is applied
                to.
            operator (np.ndarray): The operator of which we determine the
                expectation value. Note that the state will be contracted with
                axis/leg 1 of this operator.

        Returns:
            complex: The resulting expectation value < TTNS| Operator| TTN >.
        """
        if self.orthogonality_center_id == node_id:
            tensor = deepcopy(self.tensors[node_id])
            tensor_op = np.tensordot(tensor, operator, axes=(-1,1))
            tensor_conj = tensor.conj()
            legs = tuple(range(tensor.ndim))
            return complex(np.tensordot(tensor_op, tensor_conj, axes=(legs,legs)))

        tensor_product = TensorProduct({node_id: operator})
        return self.operator_expectation_value(tensor_product)

    def tensor_product_expectation_value(self,
                                         operator: TensorProduct
                                         ) -> complex:
        """
        Computes the expectation value of a tensor product of operators.

        Args:
            operator (TensorProduct): The tensor product of operators.

        Returns:
            complex: The resulting expectation value < TTNS | tensor_product | TTNS>

        """
        if len(operator) == 0:
            return self.scalar_product()
        if len(operator) == 1:
            node_id = list(operator.keys())[0]
            if self.orthogonality_center_id == node_id:
                op = operator[node_id]
                return self.single_site_operator_expectation_value(node_id, op)
        # Very inefficient, fix later without copy
        ttn = deepcopy(self)
        conj_ttn = ttn.conjugate()
        ttn.apply_operator(operator)
        return contract_two_ttns(ttn, conj_ttn)

    def ttno_expectation_value(self, operator: TTNO) -> complex:
        """
        Computes the expectation value of the TTNS with respect to a TTNO.

        Args:
            operator (TTNO): The operator to compute the expectation value with.

        Returns:
            complex: The resulting expectation value < TTNS | operator | TTNS>
        """
        return expectation_value(self, operator)

    def operator_expectation_value(self,
                                   operator: Union[TensorProduct,TTNO]
                                   ) -> complex:
        """
        Finds the expectation value of the operator specified, given this TTNS.

        Args:
            operator (Union[TensorProduct,TTNO]): A TensorProduct representing
                the operator as many single site operators. Otherwise a a TTNO
                with the same structure as the TTNS.

        Returns:
            complex: The resulting expectation value < TTNS | operator | TTNS>
        """
        if isinstance(operator, TensorProduct):
            return self.tensor_product_expectation_value(operator)
        # Operator is a TTNO
        if isinstance(operator, TTNO):
            return self.ttno_expectation_value(operator)
        raise TypeError("Operator must be a TensorProduct or a TTNO!")

    def is_in_canonical_form(self, node_id: Union[None,str] = None) -> bool:
        """
        Returns whether the TTNS is in canonical form.
        
        If a node is specified, it will check as if that node is the
        orthogonalisation center. If no node is given, the current
        orthogonalisation center will be used.

        Args:
            node_id (Union[None,str], optional): The node to check. If None,
                the current orthogonalisation center will be used. Defaults
                to None.
        
        Returns:
            bool: Whether the TTNS is in canonical form.
        """
        if node_id is None:
            node_id = self.orthogonality_center_id
        if node_id is None:
            # I.e. no orth. center exists -> no canon. form
            return False
        total_contraction = self.scalar_product(use_orthogonal_center=False)
        local_tensor = self.tensors[node_id]
        legs = range(local_tensor.ndim)
        local_contraction = complex(np.tensordot(local_tensor, local_tensor.conj(),
                                                 axes=(legs,legs)))
        # If the TTNS is in canonical form, the contraction of the
        # orthogonality center should be equal to the norm of the state.
        return np.allclose(total_contraction, local_contraction)

    def apply_operator(self, operator: TensorProduct):
        """
        Applies a tensor product operator to the TTNS.

        Args:
            operator (TensorProduct): The operator to apply.
        """
        for node_id, single_site_operator in operator.items():
            self.absorb_into_open_legs(node_id, single_site_operator)

TTNS = TreeTensorNetworkState
