import numpy as np

from ..node_contraction import contract_tensors_of_nodes
from ..tensor_util import (transpose_tensor_by_leg_list,
                          tensor_matricization,
                          truncated_tensor_svd)
from .time_evolution import TimeEvolutionAlgorithm
from ..ttn import TreeTensorNetwork

class TEBD(TimeEvolutionAlgorithm):
    """
    Runs the TEBD algorithm on a TTN
    """

    def __init__(self, state:TreeTensorNetwork, trotter_splitting, time_step_size, final_time,
                 operators=None, max_bond_dim=100, rel_tol=1e-10, total_tol=1e-15):
        """
        The state is a TreeTensorNetwork representing an initial state which is
        to be time-evolved under the `trotter_splitting` until `final_time`,
        where each time step has size time_step_size.

        If no truncation is desired set max_bond_dim = inf, rel_tol = -inf,
        and total_tol = -inf.

        Parameters
        ----------
        state : TreeTensorNetwork
            A TTN representing a quantum state.
        trotter_splitting: TrotterSplitting
            The Trotter splitting to be used for time-evolution.
        time_step_size: float
            The time difference that the state is propagated by every time step.
        final_time: float
            The final time to which the simulation is to be run.
        operators: list of dict
            A list containing dictionaries that contain node identifiers as keys and single-site
            operators as values. Each represents an operator that will be
            evaluated after every time step.
        max_bond_dim: int
            The maximum bond dimension allowed between nodes. Default is 100.
        rel_tol: float
            singular values s for which ( s / largest singular value) < rel_tol
            are truncated. Default is 0.01.
        total_tol: float
            singular values s for which s < total_tol are truncated.
            Defaults to 1e-15.

        """
        super().__init__(state, operators, time_step_size, final_time)

        self._trotter_splitting = trotter_splitting

        self.max_bond_dim = max_bond_dim
        self.rel_tol = rel_tol
        self.total_tol = total_tol

        if not self._trotter_splitting.is_compatible_with_ttn(self.state):
            raise ValueError("State TTN and Trotter Splitting are not compatible!")

        self._exponents = self._exponentiate_splitting()

    def _exponentiate_splitting(self):
        """
        Wraps the exponentiate splitting in the TrotterSplitting class.
        """
        return self._trotter_splitting.exponentiate_splitting(self.state,
                                                              self._time_step_size)

    @property
    def exponents(self):
        return self._exponents
    
    @property
    def time_step_size(self):
        return self._time_step_size
    
    @time_step_size.setter
    def time_step_size(self, new_time_step_size):
        self._time_step_size = new_time_step_size
        self._exponents = self._exponentiate_splitting()

    @property
    def trotter_splitting(self):
        return self._trotter_splitting

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
        node2_open_leg_indices = range(len(node1_open_leg_indices),current_max_leg_num)

        # One virtual leg was lost in the contraction
        num_virt_legs1 = node1.nvirt_legs() - 1
        temp_leg_num = current_max_leg_num
        current_max_leg_num  += num_virt_legs1
        node1_virt_leg_indices = range(temp_leg_num, current_max_leg_num)

        # One virtual leg was lost in the contraction
        num_virt_legs2 = node2.nvirt_legs() - 1
        temp_leg_num = current_max_leg_num
        current_max_leg_num  += num_virt_legs2
        node2_virt_leg_indices = range(temp_leg_num, current_max_leg_num)

        node1_leg_indices = list(node1_open_leg_indices)
        node1_leg_indices.extend(node1_virt_leg_indices)

        node2_leg_indices = list(node2_open_leg_indices)
        node2_leg_indices.extend(node2_virt_leg_indices)

        return node1_leg_indices, node2_leg_indices

    def _split_two_site_tensors(self, two_site_tensor, node1, node2):
            # Split the tensor in two via svd
            # Currently the leg order is
            # (open_legs1, open_legs2, virtual_legs1, virtual_legs2)
            node1_leg_indices, node2_leg_indices = self._find_node_for_legs_of_two_site_tensor(node1, node2)

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
            node2_tensor = np.tensordot(node2_tensor, np.diag(s), axes=(-1,1))

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
            raise NotImplementedError("More than two-site interactions are not yet implemented.")

    def run_one_time_step(self):
        """
        Running one time_step on the TNS according to the exponentials. The
        order in which the trotter splitting is run, is the order in which the
        time-evolution operators are saved in `self.exponents`.
        """
        for unitary in self.exponents:
            self._apply_one_trotter_step(unitary)
        
        