"""
Implements the mother class for all one-site TDVP algorithms.

This class mostly contains functions to calculate the effective Hamiltonian
and to update the link tensors.
"""
from copy import deepcopy

from .tdvp_algorithm import TDVPAlgorithm
from ..time_evolution import EvoDirection
from ...util.tensor_splitting import SplitMode
from ...core.leg_specification import LegSpecification
from ...util.ttn_exceptions import NoConnectionException
from ...contractions.state_operator_contraction import contract_any
from ..time_evo_util.effective_time_evolution import bond_time_evolution

class OneSiteTDVP(TDVPAlgorithm):
    """
    The mother class for all One-Site TDVP algorithms.

    This class contains the functions to calculate the effective Hamiltonian
    and to update the link tensors.

    Has the same attributes as the TDVP-Algorithm class, but must still be
    extended with a time step running method, defining the order of the
    Trotter decomposition.
    """

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
        updated_tensor = bond_time_evolution(link_id,
                                             self.state,
                                             time_step_factor * self.time_step_size,
                                             self.partial_tree_cache,
                                             forward=EvoDirection.BACKWARD,
                                             mode=self.config.time_evo_mode,
                                             solver_options=self.solver_options
                                             )
        self.state.tensors[link_id] = updated_tensor

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
