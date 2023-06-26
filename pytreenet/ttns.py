from __future__ import annotations

from copy import deepcopy
import numpy as np

from .ttn import TreeTensorNetwork

class TreeTensorNetworkState(TreeTensorNetwork):
    """
    This class holds methods commonly used with tree tensor networks representing a state.
    """

    def __init__(self):
        """
        Initialises in the same way as a normal TTN.
        """
        super().__init__()

    def single_site_operator_expectation_value(self, node_id: str, operator: np.ndarray, canon: bool=False) -> complex:
        """
        Find the expectation value of this TTNS given the single-site operator acting on the node specified.
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
        ttns_conj = deepcopy(self)
        for node_id2, tensor in ttns_conj.tensors.items():
            ttns_conj.tensors[node_id2] = tensor.conj()
        self.absorb_into_open_legs(node_id, operator)
        return self.contract_two_ttn(ttns_conj)

    def operator_expectation_value(self, operator_dict):
        """
        Assuming ttn represents a quantum state, this function evaluates the
        expectation value of the operator.

        Parameters
        ----------
        operator_dict : dict
            A dictionary representing an operator applied to a quantum state.
            The keys are node identifiers to which the value, a matrix, is applied.

        Returns
        -------
        exp_value: complex
            The resulting expectation value.

        """
        return operator_expectation_value(self, operator_dict)

    def scalar_product(self):
        """
        Computes the scalar product for a state_like TTN, i.e. one where the open
        legs represent a quantum state.

        Parameters
        ----------
        None

        Returns
        -------
        sc_prod: complex
            The resulting scalar product.

        """
        return scalar_product(self)