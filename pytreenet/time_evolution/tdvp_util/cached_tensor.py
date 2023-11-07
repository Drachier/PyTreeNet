from __future__ import annotations
from typing import List

import numpy as np

class CachedSiteTensor():
    """
    Class used to cache site tensors during TDVP.
    """

    def __init__(self,
                 node_state: Node,
                 node_ham: Node,
                 ket_tensor: np.ndarray,
                 ham_tensor: np.ndarray) -> None:
        """
        Args:
            node_state (Node): The node this tensor belongs to in the TTNS.
            node_ham (Node): The node this tensor belongs to in the TTNO.
            ket_tensor (np.ndarray): The tensor corresponding to this site in
             the TTNS.
            ham_tensor (np.ndarray): The tensor corresponding to this site in
             the TTNO.
        """
        self.node_state = node_state
        self.node_ham = node_ham
        self.ket_tensor = ket_tensor
        self.ham_tensor = ham_tensor

    def compute(self) -> np.ndarray:
        """
        Copmutes and reshapes the tensor corresponding this site.
        """
        tensor = self.contract_tensor_sandwich()
        # Current leg ordering (all_ket_legs, all_ham_legs, all_bra_legs)
        tensor = tensor.transpose(self.leg_order())
        # Leg order now (leg1_ket, leg1_ham, leg1_bra, leg2_ket, ...)
        tensor = tensor.reshape(self.new_shape(tensor))
        # Leg order now ([leg1_ket, leg1_ham, leg1_bra], [leg2_ket, ...)
        # i.e. all legs pointing to the same neighbour are now one big leg.
        return tensor

    def cached_legs(self) -> int:
        """
        The number of neighbours/legs pointing in the same direction of this
         tensor.
        """
        return self.node_state.nneighbours()

    def contract_tensor_sandwich(self) -> np.ndarray:
        """
        Contracts the site tensors to the transfer tensor, i.e.
                         _____
                    ____|     |____
                        |  A* |
                        |_____|
                           |
                           |
                         __|__
                    ____|     |____
                        |  H  |
                        |_____|
                           |
                           |
                         __|__
                    ____|     |____
                        |  A  |
                        |_____|

        Returns:
            np.ndarray: The resulting tensor.
        """
        braham_tensor = np.tensordot(self.ket_tensor,
                                     self.ham_tensor,
                                     axes=(self._node_state_phys_leg(),
                                           self._node_operator_input_leg()))
        brahamket_tensor = np.tensordot(braham_tensor, self.ket_tensor.conj(),
                                        axes=(self.ket_tensor.ndim-1 + self._node_operator_output_leg(),
                                              self._node_state_phys_leg()))
        return brahamket_tensor

    def leg_order(self) -> List[int]:
        """
        Orders the legs of the main tensor such that all legs pointing to the
         same neighbour are next to each other.
        """
        num_cached_tensor_legs = self.cached_legs()
        ordered_legs = []
        for leg_num in range(num_cached_tensor_legs):
            ordered_legs += [leg_num + j*num_cached_tensor_legs for j in [0, 1, 2]]
        return ordered_legs

    def new_shape(self, tensor: np.ndarray) -> List[int]:
        """
        Reshapes the tensor such that all legs pointing to the same neighbour
         are one big leg.
        """
        shape = []
        for leg_num in range(self.cached_legs()):
            shape.append(np.prod([tensor.shape[3 * leg_num + j] for j in [0,1,2]]))
        return shape

    def _node_state_phys_leg(self) -> int:
        """
        Finds the leg of a node of the state that corresponds to the physical
         leg.

        Returns:
            int: The phyisical leg of a node.
        """
        return self.node_state.open_legs[-1]

    def _node_operator_input_leg(self) -> int:
        """
        Finds the leg of a node of the hamiltonian corresponding to the input.

        Returns:
            int: The index of the leg corresponding to input.
        """
        # Corr ket leg
        return self.node_ham.open_legs[-1]

    def _node_operator_output_leg(self) -> int:
        """
        Finds the leg of a node of the hamiltonian corresponding to the
         output.
        
        Returns:
            int: The index of the leg corresponding to output.
        """
        # Corr bra leg
        return self.node_ham.open_legs[-2]
        