from __future__ import annotations
from typing import Dict, List, Union

import numpy as np

from ..tensor_util import (transpose_tensor_by_leg_list,
                          tensor_matricization,
                          truncated_tensor_svd)
from .time_evolution import TimeEvolution

class TEBD(TimeEvolution):
    """
    Runs the TEBD algorithm on a TTN
    """

    def __init__(self, initial_state: TreeTensorNetworkState,
                 trotter_splitting: TrotterSplitting, time_step_size: float,
                 final_time: float, operators: Union[List[Dict], Dict],
                 max_bond_dim: int = 100, rel_tol: float =1e-10,
                 total_tol: float = 1e-15):
        """
        A class that can be used to time evolve an initial state in the form
         a tree tensor network state via the TEBD algorithm.

        If no truncation is desired set max_bond_dim = inf, rel_tol = -inf,
         and total_tol = -inf.

        Args:
            initial_state (TreeTensorNetworkState)): The initial state of our
             time-evolution
            trotter_splitting (TrotterSplitting): The Trotter splitting to be
             used for time-evolution.
            time_step_size (float): The time step size to be used.
            final_time (float): The final time until which to run.
            operators (Union[List[Dict], Dict]): Operators for which expectation values
             should be determined
            max_bond_dim (int, optional): The maximum bond dimension allowed between
            nodes. Defaults to 100.
            rel_tol (float, optional): singular values s for which
             ( s / largest singular value) < rel_tol are truncated. Defaults to 1e-10.
            total_tol (float, optional): singular values s for which s < total_tol
             are truncated. Defaults to 1e-15.

        Raises:
            ValueError: If the trotter splitting and TTNS are not compatible.
        """
        super().__init__(initial_state, time_step_size, final_time, operators)
        self._trotter_splitting = trotter_splitting

        self.max_bond_dim = max_bond_dim
        self.rel_tol = rel_tol
        self.total_tol = total_tol

        if not self._trotter_splitting.is_compatible_with_ttn(self.state):
            raise ValueError(
                "State TTN and Trotter Splitting are not compatible!")

        self._exponents = self._trotter_splitting.exponentiate_splitting(self.state,
                                                              self._time_step_size)

    @property
    def exponents(self):
        return self._exponents

    @property
    def trotter_splitting(self):
        return self._trotter_splitting

    @property
    def results(self):
        return self._results

    def _apply_one_trotter_step_single_site(self, single_site_exponent):
        """
        Applies the single-site exponential operator of the Trotter splitting.

        Parameters
        ----------
        single_site_exponent: dict
            A dictionary with representing a single-site unitary operator.
            The operator is saved with key "operator" and the site to which it
            is applied is saved via node identifiers in the key "site_ids"

        Returns
        -------
        None.

        """
        operator = single_site_exponent["operator"]
        identifiers = single_site_exponent["site_ids"]

        node = self.state.nodes[identifiers[0]]

        node.absorb_tensor(operator, (1, ), (node.open_legs[0], ))

    @staticmethod
    def _find_node_for_legs_of_two_site_tensor(node1, node2):
        """
        From the leg order
        (open_legs1, open_legs2, virtual_legs1, virtual_legs2),
        where the connecting legs are not included, we want to find the
        legs belonging to each individual node.
        """
        current_max_leg_num = node1.nopen_legs()
        node1_open_leg_indices = range(0, current_max_leg_num)

        current_max_leg_num += node2.nopen_legs()
        node2_open_leg_indices = range(
            len(node1_open_leg_indices), current_max_leg_num)

        # One virtual leg was lost in the contraction
        num_virt_legs1 = node1.nvirt_legs() - 1
        temp_leg_num = current_max_leg_num
        current_max_leg_num += num_virt_legs1
        node1_virt_leg_indices = range(temp_leg_num, current_max_leg_num)

        # One virtual leg was lost in the contraction
        num_virt_legs2 = node2.nvirt_legs() - 1
        temp_leg_num = current_max_leg_num
        current_max_leg_num += num_virt_legs2
        node2_virt_leg_indices = range(temp_leg_num, current_max_leg_num)

        node1_leg_indices = list(node1_open_leg_indices)
        node1_leg_indices.extend(node1_virt_leg_indices)

        node2_leg_indices = list(node2_open_leg_indices)
        node2_leg_indices.extend(node2_virt_leg_indices)

        return node1_leg_indices, node2_leg_indices

    @staticmethod
    def _permutation_svdresult_u_to_fit_node(node):
        """
        After the SVD the two tensors associated to each site have an incorrect
        leg ordering. This function finds the permutation of legs to correct
        this. In more detail:
        The nodes currently have shape
        (virtual-legs, open-legs, contracted leg)
        and the tensor u from the svd has the leg order
        (open-legs, virtual-legs, contracted leg)

        Returns
        -------
        permutation: list of int
            Permutation to find correct legs. Compatible with numpy transpose,
            i.e. the entries are the old leg index and the position in the
            permutation is the new leg index.
        """
        num_open_legs = node.nopen_legs()
        permutation = list(
            range(num_open_legs, num_open_legs + node.nvirt_legs() - 1))
        permutation.extend(range(0, num_open_legs))
        permutation.append(node.nlegs() - 1)

        return permutation

    @staticmethod
    def _permutation_svdresult_v_to_fit_node(node):
        """
        After the SVD the two tensors associated to each site have an incorrect
        leg ordering. This function finds the permutation of legs to correct
        this. In more detail:
        The nodes currently have shape
        (virtual-legs, open-legs, contracted leg)
        and the tensor v from the svd has the leg order
        (contracted leg, open-legs, virtual-legs)

        Returns
        -------
        permutation: list of int
            Permutation to find correct legs. Compatible with numpy transpose,
            i.e. the entries are the old leg index and the position in the
            permutation is the new leg index.
        """
        num_open_legs = node.nopen_legs()
        permutation = list(
            range(num_open_legs + 1, num_open_legs + 1 + node.nvirt_legs() - 1))
        permutation.extend(range(1, num_open_legs + 1))
        permutation.append(0)

        return permutation

    def _split_two_site_tensors(self, two_site_tensor, node1, node2):
        # Split the tensor in two via svd
        # Currently the leg order is
        # (open_legs1, open_legs2, virtual_legs1, virtual_legs2)
        node1_leg_indices, node2_leg_indices = self._find_node_for_legs_of_two_site_tensor(
            node1, node2)

        node1_tensor, s, node2_tensor = truncated_tensor_svd(two_site_tensor,
                                                             node1_leg_indices,
                                                             node2_leg_indices,
                                                             max_bond_dim=self.max_bond_dim,
                                                             rel_tol=self.rel_tol,
                                                             total_tol=self.total_tol)

        permutation1 = TEBD._permutation_svdresult_u_to_fit_node(node1)
        node1_tensor = node1_tensor.transpose(permutation1)
        node1.tensor = node1_tensor

        permutation2 = TEBD._permutation_svdresult_v_to_fit_node(node2)
        node2_tensor = node2_tensor.transpose(permutation2)
        # We absorb the singular values into this second tensor
        node2_tensor = np.tensordot(node2_tensor, np.diag(s), axes=(-1, 1))

        node2.tensor = node2_tensor

    def _apply_one_trotter_step_two_site(self, two_site_exponent):
        """
        Applies the two-site exponential operator of the Trotter splitting.

        Parameters
        ----------
        two_site_exponent: dict
            A dictionary with representing a two-site unitary operator.
            The operator is saved with key "operator" and the sites to which it
            is applied are saved via node identifiers in the key "site_ids"

        Returns
        -------
        None.

        """
        operator = two_site_exponent["operator"]
        identifiers = two_site_exponent["site_ids"]

        node1 = self.state.nodes[identifiers[0]]
        node2 = self.state.nodes[identifiers[1]]

        two_site_tensor, leg_dict = contract_tensors_of_nodes(node1, node2)

        # Matricise site_tensor
        # Output legs will be the combined physical legs
        output_legs = [leg_index for leg_index in
                       leg_dict[node1.identifier + "open"]]
        output_legs.extend([leg_index for leg_index in
                            leg_dict[node2.identifier + "open"]])

        # Input legs are the combined virtual legs
        input_legs = [leg_index for leg_index in
                      leg_dict[node1.identifier + "virtual"]]
        input_legs.extend([leg_index for leg_index in
                           leg_dict[node2.identifier + "virtual"]])

        # Save original shape for later
        two_site_tensor = transpose_tensor_by_leg_list(two_site_tensor,
                                                       output_legs,
                                                       input_legs)
        orig_shape = two_site_tensor.shape

        two_site_tensor = tensor_matricization(two_site_tensor,
                                               output_legs,
                                               input_legs,
                                               correctly_ordered=True)

        # exp(-H dt) * |psi_loc>
        new_two_site_tensor = operator @ two_site_tensor

        new_two_site_tensor = new_two_site_tensor.reshape(orig_shape)

        self._split_two_site_tensors(new_two_site_tensor, node1, node2)

    def _apply_one_trotter_step(self, unitary):
        """
        Applies the exponential operator of the Trotter splitting that is
        chosen via index

        Parameters
        ----------
        unitary : dict
            A dictionary representing a time evolution operator (usually a unitary matrix), 
            where the actual operator is saved as an ndarray under the key
            `"operator"` and the sites it is applied to are saved as a list of
            strings/site identifiers under they key `"site_ids"`

        Returns
        -------
        None.

        """
        num_of_sites_acted_upon = len(unitary["site_ids"])

        if num_of_sites_acted_upon == 0:
            pass
        elif num_of_sites_acted_upon == 1:
            self._apply_one_trotter_step_single_site(unitary)
        elif num_of_sites_acted_upon == 2:
            self._apply_one_trotter_step_two_site(unitary)
        else:
            raise NotImplementedError(
                "More than two-site interactions are not yet implemented.")

    def run_one_time_step(self):
        """
        Running one time_step on the TNS according to the exponentials. The
        order in which the trotter splitting is run, is the order in which the
        time-evolution operators are saved in `self.exponents`.
        """
        for unitary in self.exponents:
            self._apply_one_trotter_step(unitary)
