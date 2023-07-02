from __future__ import annotations

from copy import deepcopy
import numpy as np

from .ttn import TreeTensorNetwork
from .operators.tensorproduct import TensorProduct

class TreeTensorNetworkState(TreeTensorNetwork):
    """
    This class holds methods commonly used with tree tensor networks representing a state.
    """

    def __init__(self):
        """
        Initialises in the same way as a normal TTN.
        """
        super().__init__()

    def single_site_operator_expectation_value(self, node_id: str, operator: np.ndarray,
                                               canon: bool=False) -> complex:
        """
        Find the expectation value of this TTNS given the single-site operator acting on
         the node specified.
        Assumes the node has only one open leg.

        Args:
            node_id (str): The identifier of the node, the operator is applied to.
            operator (np.ndarray): The operator of which we determine the expectation value.
            canon (bool, optional): Whether the node is the orthogonality center of the TTNS.
                                     Defaults to False.

        Returns:
            complex: The resulting expectation value < TTNS| Operator| TTN >
        """
        if canon:
            tensor = deepcopy(self.tensors[node_id])
            tensor_op = np.tensordot(tensor, operator, axes=(-1,1))
            tensor_conj = tensor.conj()
            legs = tuple(range(tensor.ndim))
            return complex(np.tensordot(tensor_op, tensor_conj, axes=(legs,legs)))

        # Very inefficient, fix later without copy
        ttns_conj = self.conjugate()
        self.absorb_into_open_legs(node_id, operator)
        return self.contract_two_ttn(ttns_conj)

    def operator_expectation_value(self, operator: TensorProduct) -> complex:
        """
        Finds the expectation value of the operator specified, given this TTNS.

        Args:
            operator (TensorProduct): A TensorProduct representing the operator
             as many single site operators.

        Returns:
            complex: The resulting expectation value < TTNS | operator | TTNS>
        """
        # Very inefficient, fix later without copy
        conj_ttn = self.conjugate()
        for node_id, single_site_operator in operator.items():
            self.absorb_into_open_legs(node_id, single_site_operator)
        return self.contract_two_ttn(conj_ttn)

    def scalar_product(self) -> complex:
        """
        Computes the scalar product of this TTNS

        Returns:
            complex: The resulting scalar product <TTNS|TTNS>
        """
        # Very inefficient, fix later without copy
        return self.contract_two_ttn(self.conjugate())
