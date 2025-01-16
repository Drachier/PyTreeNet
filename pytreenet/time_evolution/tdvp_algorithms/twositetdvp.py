"""
Implements the mother class for all two-site TDVP algorithms.

This class mostly contains functions to contract the effective Hamiltonian
and to update the sites in the two site scheme.
"""
from typing import List, Union, Tuple
import numpy as np

from .tdvp_algorithm import TDVPAlgorithm
from ..ttn_time_evolution import TTNTimeEvolutionConfig
from ..time_evolution import time_evolve
from ...ttns.ttns import TreeTensorNetworkState
from ...ttno.ttno_class import TTNO
from ...operators.tensorproduct import TensorProduct
from ...util.tensor_splitting import SVDParameters
from ...util.tensor_util import tensor_matricisation_half
from ...contractions.contraction_util import contract_all_but_one_neighbour_block_to_hamiltonian
from ...util.ttn_exceptions import NotCompatibleException

class TwoSiteTDVP(TDVPAlgorithm):
    """
    The two site TDVP algorithm.

    It contracts two sites and updates them at the same time before splitting
    them using an SVD with truncation.

    Attributes:
        svd_parameters: Contains values for the SVD truncation.
    """

    def __init__(self, initial_state: TreeTensorNetworkState,
                 hamiltonian: TTNO,
                 time_step_size: float, final_time: float,
                 operators: Union[TensorProduct, List[TensorProduct]],
                 svd_parameters: Union[SVDParameters,None] = None,
                 config: Union[TTNTimeEvolutionConfig,None] = None) -> None:
        """
        Initialises an instance of a two-site TDVP algorithm.

        Args:
            intial_state (TreeTensorNetworkState): The initial state of the
                system.
            hamiltonian (TTNO): The Hamiltonian in TTNO form under which to
                time-evolve the system.
            time_step_size (float): The size of one time-step.
            final_time (float): The final time until which to run the evolution.
            operators (Union[TensorProduct, List[TensorProduct]]): Operators
                to be measured during the time-evolution.
            truncation_parameters (Union[SVDParameters,None]): The truncation
                parameters to use during the time-evolution. Can be used to set
                the maximum bond dimension, the absolute tolerance, and the
                relative tolerance.
        """
        super().__init__(initial_state, hamiltonian,
                         time_step_size, final_time,
                         operators,
                         config)
        if svd_parameters is None:
            self.svd_parameters = SVDParameters()
        else:
            self.svd_parameters = svd_parameters

    def _find_block_leg_target_node(self,
                                    target_node_id: str,
                                    next_node_id: str,
                                    neighbour_id: str) -> int:
        """
        Determines the leg index of the input leg of the contracted subtree
        block on the effective hamiltonian tensor corresponding to a given
        neighbour of the target node.

        Args:
            target_node_id (str): The id of the target node.
            next_node_id (str): The id of the next node.
            neighbour_id (str): The id of the neighbour of the target node.
        
        Returns:
            int: The leg index of the input leg of the contracted subtree
                block on the effective hamiltonian tensor.
        """
        target_node = self.hamiltonian.nodes[target_node_id]
        index_next_node = target_node.neighbour_index(next_node_id)
        ham_neighbour_index = target_node.neighbour_index(neighbour_id)
        constant = int(ham_neighbour_index < index_next_node)
        return 2 * (ham_neighbour_index + constant)

    def _find_block_leg_next_node(self,
                                  target_node_id: str,
                                  next_node_id: str,
                                  neighbour_id: str) -> int:
        """
        Determines the leg index of the input leg of the contracted subtree
        bnlock on the effective hamiltonian tensor corresponding to a given
        eighbour of the next node.

        Args:
            target_node_id (str): The id of the target node.
            next_node_id (str): The id of the next node.
            neighbour_id (str): The id of the neighbour of the next node.
        
        Returns:
            int: The leg index of the input leg of the contracted subtree
                block on the effective hamiltonian tensor.
        """
        # Luckily the situation is pretty much the same so we can reuse most
        # of the code.
        leg_index_temp = self._find_block_leg_target_node(next_node_id,
                                                          target_node_id,
                                                          neighbour_id)
        target_node = self.hamiltonian.nodes[target_node_id]
        target_node_numn = target_node.nneighbours()
        return 2 * target_node_numn + leg_index_temp

    def _determine_two_site_leg_permutation(self,
                                            target_node_id: str,
                                            next_node_id: str) -> Tuple[int]:
        """
        Determine the permutation of the effective Hamiltonian tensor.
        
        This is the leg permutation required on the two-site effective
        Hamiltonian tensor to fit with the underlying TTNS, assuming
        the two sites have already been contracted in the TTNS.

        Args:
            target_node_id (str): Id of the main node on which the update is
                performed.
            next_node_id (str): The id of the second node.
        
        Returns:
            Tuple[int]: The permutation of the legs of the two-site effective
                Hamiltonian tensor.
        """
        neighbours_target = self.hamiltonian.nodes[target_node_id].neighbouring_nodes()
        neighbours_next = self.hamiltonian.nodes[next_node_id].neighbouring_nodes()
        two_site_id = self.create_two_site_id(target_node_id, next_node_id)
        neighbours_two_site = self.state.nodes[two_site_id].neighbouring_nodes()
        # Determine the permutation of the legs
        input_legs = []
        for neighbour_id in neighbours_two_site:
            if neighbour_id in neighbours_target:
                block_leg = self._find_block_leg_target_node(target_node_id,
                                                             next_node_id,
                                                             neighbour_id)
            elif neighbour_id in neighbours_next:
                block_leg = self._find_block_leg_next_node(target_node_id,
                                                           next_node_id,
                                                           neighbour_id)
            else:
                errstr = "The two-site Hamiltonian has a neighbour that is not a neighbour of the two sites."
                raise NotCompatibleException(errstr)
            input_legs.append(block_leg)
        output_legs = [leg + 1 for leg in input_legs]
        target_num_neighbours = self.hamiltonian.nodes[target_node_id].nneighbours()
        output_legs = output_legs + [0,2*target_num_neighbours] # physical legs
        input_legs = input_legs + [1,2*target_num_neighbours + 1] # physical legs
        # As in matrices, the output legs are first
        return tuple(output_legs + input_legs)

    def _contract_all_except_two_nodes(self,
                                       target_node_id: str,
                                       next_node_id: str) -> np.ndarray:
        """
        Contracts the nodes for all but two sites.

        Uses all cached tensors to contract bra, ket, and hamiltonian tensors
        of all nodes except for the two given nodes. Of these nodes only the
        Hamiltonian nodes are contracted.
        IMPORTANT: This function assumes that the given nodes are already
        contracted in the TTNS.

        Args:
            target_node_id (str): The id of the node that should not be
                contracted.
            next_node_id (str): The id of the second node that should not
                be contracted.

        Returns:
            np.ndarray: The resulting effective two-site Hamiltonian tensor::

                 _____                out              _____
                |     |____n-1                  0_____|     |
                |     |                               |     |
                |     |        |n           |         |     |
                |     |     ___|__        __|___      |     |
                |     |    |      |      |      |     |     |
                |     |____|      |______|      |_____|     |
                |     |    |  H_1 |      | H_2  |     |     |
                |     |    |______|      |______|     |     |
                |     |        |            |         |     |
                |     |        |2n+1        |         |     |
                |     |                               |     |
                |     |_____                     _____|     |
                |_____|  2n                       n+1 |_____|
                                      in
        
        """
        # Contract all but one neighbouring block of each node
        h_target = self.hamiltonian.tensors[target_node_id]
        target_node = self.hamiltonian.nodes[target_node_id]
        target_block = contract_all_but_one_neighbour_block_to_hamiltonian(h_target,
                                                                           target_node,
                                                                           next_node_id,
                                                                           self.partial_tree_cache)
        h_next = self.hamiltonian.tensors[next_node_id]
        next_node = self.hamiltonian.nodes[next_node_id]
        next_block = contract_all_but_one_neighbour_block_to_hamiltonian(h_next,
                                                                         next_node,
                                                                         target_node_id,
                                                                         self.partial_tree_cache)
        # Contract the two blocks
        h_eff = np.tensordot(target_block, next_block, axes=(0,0))
        # Now we need to sort the legs to fit with the underlying TTNS.
        # Note that we assume, the two tensors have already been contracted.
        leg_permutation = self._determine_two_site_leg_permutation(target_node_id,
                                                                   next_node_id)
        return h_eff.transpose(leg_permutation)

    def _get_effective_two_site_hamiltonian(self,
                                            target_node_id: str,
                                            next_node_id: str) -> np.ndarray:
        """
        Obtain the effective two site Hamiltonian as a matrix.

        Args:
            target_node_id (str): The id of the target node.
            next_node_id (str): The id of the next node.

        Returns:
            np.ndarray: The effective two site Hamiltonian as a matrix.
        """
        h_eff_tensor = self._contract_all_except_two_nodes(target_node_id,
                                                           next_node_id)
        # Reshape the tensor to a matrix
        return tensor_matricisation_half(h_eff_tensor)

    def _update_two_site_nodes(self,
                               target_node_id: str,
                               next_node_id: str,
                               time_step_factor: float = 1):
        """
        Perform the two-site update on the two given sites.
        
        The target node is the original orthogonalisation center. The center
        will be moved to the next node in this update.

        Args:
            target_node_id (str): The id of the target node.
            next_node_id (str): The id of the next node.
            time_step_factor (float): The factor by which to multiply the
                time step size. Default is 1.
        """
        u_legs, v_legs = self.state.legs_before_combination(target_node_id,
                                                            next_node_id)
        new_id = self.create_two_site_id(target_node_id, next_node_id)
        self.state.contract_nodes(target_node_id, next_node_id,
                                  new_identifier=new_id)
        h_eff = self._get_effective_two_site_hamiltonian(target_node_id,
                                                         next_node_id)
        psi = self.state.tensors[new_id]
        # Perform the time-evolution
        updated_two_sites = time_evolve(psi,
                                        h_eff,
                                        self.time_step_size * time_step_factor,
                                        forward=True,
                                        mode=self.config.time_evo_mode)
        self.state.tensors[new_id] = updated_two_sites
        # Split the two-site tensor using SVD
        self.state.split_node_svd(new_id, u_legs, v_legs,
                                  u_identifier=target_node_id,
                                  v_identifier=next_node_id,
                                  svd_params=self.svd_parameters)
        self.state.orthogonality_center_id = next_node_id
        self.update_tree_cache(target_node_id, next_node_id)

    def _single_site_backwards_update(self,
                                      node_id: str,
                                      time_step_factor: float = 1):
        """
        Performs the single-site backwards update on the given node.

        Args:
            node_id (str): The id of the node to update.
            time_step_factor (float): The factor by which to multiply the
                time step size. Default is 1.
        """
        self._update_site(node_id, -1 * time_step_factor)

    @staticmethod
    def create_two_site_id(node_id: str, next_node_id: str) -> str:
        """
        Create the identifier of a two site node obtained from contracting
        the two note with the input identifiers.
        """
        return "TwoSite_" + node_id + "_contr_" + next_node_id
