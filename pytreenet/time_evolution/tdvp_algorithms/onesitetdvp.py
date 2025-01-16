"""
Implements the mother class for all one-site TDVP algorithms.

This class mostly contains functions to calculate the effective Hamiltonian
and to update the link tensors.
"""
from copy import deepcopy

import numpy as np

from .tdvp_algorithm import TDVPAlgorithm
from ..time_evolution import time_evolve
from ...util.tensor_util import tensor_matricisation_half
from ...util.tensor_splitting import SplitMode
from ...core.leg_specification import LegSpecification
from ...util.ttn_exceptions import NoConnectionException
from ...contractions.state_operator_contraction import contract_any

class OneSiteTDVP(TDVPAlgorithm):
    """
    The mother class for all One-Site TDVP algorithms.

    This class contains the functions to calculate the effective Hamiltonian
    and to update the link tensors.

    Has the same attributes as the TDVP-Algorithm class, but must still be
    extended with a time step running method, defining the order of the
    Trotter decomposition.
    """

    def _get_effective_link_hamiltonian(self, node_id: str,
                                        next_node_id: str) -> np.ndarray:
        """
        Obtains the effective link Hamiltonian.

        Args:
            node_id (str): The last node that was centered in the effective
                Hamiltonian.
            next_node_id (str): The next node to go to. The link for which
                this effective Hamiltonian is constructed is between the two
                nodes.

        Returns:
            np.ndarray: The effective link Hamiltonian::

                 _____       out         _____
                |     |____1      0_____|     |
                |     |                 |     |
                |     |                 |     |
                |     |                 |     |
                |     |                 |     |
                |     |_________________|     |
                |     |                 |     |
                |     |                 |     |
                |     |                 |     |
                |     |                 |     |
                |     |                 |     |
                |     |_____       _____|     |
                |_____|  2         3    |_____|
                              in
            
        """
        link_id = self.create_link_id(node_id, next_node_id)
        target_node = self.state.nodes[link_id]
        assert not target_node.is_root()
        assert len(target_node.children) == 1
        new_cache_tensor = self.partial_tree_cache.get_entry(node_id, next_node_id)
        # We get the cached tensor of the other neighbour of the link
        other_cache_tensor = self.partial_tree_cache.get_entry(next_node_id, node_id)
        # Contract the Hamiltonian legs
        if target_node.is_parent_of(node_id):
            tensor = np.tensordot(other_cache_tensor,
                                  new_cache_tensor,
                                  axes=(1,1))
        else:
            tensor = np.tensordot(new_cache_tensor,
                                  other_cache_tensor,
                                  axes=(1,1))
        tensor = np.transpose(tensor, axes=[1,3,0,2])
        return tensor_matricisation_half(tensor)

    def _time_evolve_link_tensor(self, node_id: str,
                                next_node_id: str,
                                time_step_factor: float = 1):
        """
        Time evolves a link tensor.

        The link tensor will appear as the R-tensor in the QR-decomposition
        after splitting a site. It is evolved backwards in time.

        Args:
            node_id (str): The node from which the link tensor originated.
            next_node_id (str): The other tensor the link connects to.
            time_step_factor (float, optional): A factor that should be
                multiplied with the internal time step size. Defaults to 1.
        """
        link_id = self.create_link_id(node_id, next_node_id)
        link_tensor = self.state.tensors[link_id]
        hamiltonian_eff_link = self._get_effective_link_hamiltonian(node_id,
                                                                    next_node_id)
        self.state.tensors[link_id] = time_evolve(link_tensor,
                                                  hamiltonian_eff_link,
                                                  self.time_step_size * time_step_factor,
                                                  forward=False,
                                                  mode=self.config.time_evo_mode)

    def _update_cache_after_split(self, node_id: str, next_node_id: str):
        """
        Updates the cached tensor after splitting a tensor.

        Args:
            node_id (str): Node to update
            next_node_id (str): Next node to which the link is found
        """
        link_id = self.create_link_id(node_id, next_node_id)
        new_tensor = contract_any(node_id, link_id,
                                  self.state, self.hamiltonian,
                                  self.partial_tree_cache)
        self.partial_tree_cache.add_entry(node_id, next_node_id, new_tensor)

    def _split_updated_site(self,
                            node_id: str,
                            next_node_id: str):
        """
        Splits a node using QR-decomposition and updates the cache.

        Args:
            node_id (str): Node to update
            next_node_id (str): Next node to which the link is found
        """
        node = self.state.nodes[node_id]
        if node.is_parent_of(next_node_id):
            q_children = deepcopy(node.children)
            q_children.remove(next_node_id)
            q_legs = LegSpecification(node.parent,
                                      q_children,
                                      node.open_legs,
                                      is_root=node.is_root())
            r_legs = LegSpecification(None, [next_node_id], [])
        elif node.is_child_of(next_node_id):
            q_legs = LegSpecification(None,
                                      deepcopy(node.children),
                                      node.open_legs)
            r_legs = LegSpecification(node.parent, [], [])
        else:
            errstr = f"Nodes {node_id} and {next_node_id} are not connected!"
            raise NoConnectionException(errstr)
        link_id = self.create_link_id(node_id, next_node_id)
        self.state.split_node_qr(node_id, q_legs, r_legs,
                                 q_identifier=node.identifier,
                                 r_identifier=link_id,
                                 mode=SplitMode.KEEP)
        self._update_cache_after_split(node_id, next_node_id)

    def _update_link(self, node_id: str,
                     next_node_id: str,
                     time_step_factor: float = 1):
        """
        Updates a link tensor between two nodes using the effective link
        Hamiltonian.
         
        To achieve this the site updated latest is split via
        QR-decomposition and the R-tensor is updated. The R-tensor is then
        contracted with next node to be updated.

        Args:
            node_id (str): The node from which the link tensor originated.
            next_node_id (str): The other tensor the link connects to.
            time_step_factor (float, optional): A factor that should be
                multiplied with the internal time step size. Defaults to 1.
        """
        assert self.state.orthogonality_center_id == node_id
        self._split_updated_site(node_id, next_node_id)
        self._time_evolve_link_tensor(node_id,next_node_id,
                                      time_step_factor=time_step_factor)
        link_id = self.create_link_id(node_id, next_node_id)
        self.state.contract_nodes(link_id, next_node_id,
                                  new_identifier=next_node_id)
        self.state.orthogonality_center_id = next_node_id

    @staticmethod
    def create_link_id(node_id: str, next_node_id: str) -> str:
        """
        Creates the identifier of a link node after a split happened.
        """
        return "link_" + node_id + "_with_" + next_node_id
