from __future__ import annotations
from typing import Union
from copy import deepcopy

import numpy as np

from ..core.ttn import TreeTensorNetwork
from ..ttno import TTNO
from ..operators.tensorproduct import TensorProduct
from ..contractions.state_state_contraction import contract_two_ttns
from ..contractions.state_operator_contraction import expectation_value

class TreeTensorNetworkState(TreeTensorNetwork):
    """
    This class holds methods commonly used with tree tensor networks
     representing a state.
    """

    def scalar_product(self) -> complex:
        """
        Computes the scalar product of this TTNS

        Returns:
            complex: The resulting scalar product <TTNS|TTNS>
        """
        if self.orthogonality_center_id is not None:
            tensor = self.tensors[self.orthogonality_center_id]
            tensor_conj = tensor.conj()
            legs = tuple(range(tensor.ndim))
            return complex(np.tensordot(tensor, tensor_conj, axes=(legs,legs)))
        # Very inefficient, fix later without copy
        ttn = deepcopy(self)
        return contract_two_ttns(ttn, ttn.conjugate())

    def single_site_operator_expectation_value(self, node_id: str,
                                               operator: np.ndarray) -> complex:
        """
        Find the expectation value of this TTNS given the single-site operator acting on
         the node specified.
        Assumes the node has only one open leg.

        Args:
            node_id (str): The identifier of the node, the operator is applied to.
            operator (np.ndarray): The operator of which we determine the expectation value.
             Note that the state will be contracted with axis/leg 0 of this operator.

        Returns:
            complex: The resulting expectation value < TTNS| Operator| TTN >
        """
        if self.orthogonality_center_id == node_id:
            tensor = deepcopy(self.tensors[node_id])
            tensor_op = np.tensordot(tensor, operator, axes=(-1,0))
            tensor_conj = tensor.conj()
            legs = tuple(range(tensor.ndim))
            return complex(np.tensordot(tensor_op, tensor_conj, axes=(legs,legs)))

        tensor_product = TensorProduct({node_id: operator})
        return self.operator_expectation_value(tensor_product)

    def operator_expectation_value(self, operator: Union[TensorProduct,TTNO]) -> complex:
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
            for node_id, single_site_operator in operator.items():
                ttn.absorb_into_open_legs(node_id, single_site_operator)
            return contract_two_ttns(ttn, conj_ttn)
        # Operator is a TTNO
        return expectation_value(self, operator)

TTNS = TreeTensorNetworkState
