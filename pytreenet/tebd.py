import numpy as np

from scipy.linalg import expm

from .node_contraction import contract_tensors_of_nodes
from .tensor_util import (transpose_tensor_by_leg_list,
                          tensor_matricization, 
                          truncated_tensor_svd)

class TEBD:
    """
    Runs the TEBD algorithm on a TTN
    """
    
    def __init__(self, state, hamiltonian, time_step_size, final_time,
                 custom_splitting=None, evaluate_operator=None,
                 max_bond_dim=100, rel_tol=0.01, total_tol=1e-15):
        """
        The state is a TreeTensorNetwork representing an initial state which is
        to be time-evolved under the Hamiltonian hamiltonian until final_time,
        where each time step has size time_step_size.
        
        If no truncation is desired set max_bond_dim = inf, rel_tol = -inf,
        and total_tol = -inf.
        
        Parameters
        ----------
        state : TreeTensorNetwork
            A TTN representing a quantum state.
        time_step_size: float
            The time difference that the state is propagated by every time step.
        final_time: float
            The final time to which the simulation is to be run.
        custom_splitting: list of int
            The integers int the list should give the order in which the Hamiltonian
            terms are to be applied, i.e. the order of the Trotter-Suzuki splitting.
        evaluate_operator: dict
            A dictionary that contains node identifiers as keys and single-site
            operators as values. It represents an operator that will be
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
        
        self.state = state
        self._hamiltonian = hamiltonian
        
        self._time_step_size = time_step_size
        self.final_time = final_time
        self.num_time_steps = np.ceil(final_time / time_step_size)
        
        if custom_splitting==None:
            raise NotImplementedError()
        else:
            self.splitting = custom_splitting
            
        self.max_bond_dim = max_bond_dim
        self.rel_tol = rel_tol
        self.total_tol = total_tol
        
        self._exponents = self._exponentiate_terms()
        
    def _exponentiate_terms(self):
        """
        Exponentiates each of the terms. If we were to split the matrix into a tensor
        the site_id order should correspond to the input leg order.
        (i.e. the site with id at position n should have its leg contracted with the exponent's nth leg)
        
        If time_step_size or hamiltonian is to be changed, this function has to be rerun.
        """
        # TODO: Implement dimension checks
        
        exponents = []
        
        for interaction_operator in self.hamiltonian.terms:
            total_operator = 1
            site_ids = []
            
            for site in interaction_operator:
                total_operator = np.kron(total_operator,
                                              interaction_operator[site],)

                site_ids.append(site)
            
            exponentiated_operator = expm((-1j*self.time_step_size) * total_operator)
                    
            exponents.append({"operator": exponentiated_operator,
                                   "site_ids": site_ids})
            
        return exponents
            
    @property
    def time_step_size(self):
        return self._time_step_size
    
    @time_step_size.setter
    def time_step_size(self, new_time_step_size):
        self._time_step_size = new_time_step_size
        self._exponents = self._exponentiate_terms()
    
    @property
    def hamiltonian(self):
        return self._hamiltonian

    @hamiltonian.setter
    def hamiltonian(self, new_hamiltonian):
        self._hamiltonian = new_hamiltonian
        self._exponents = self._exponentiate_terms()
    
    @property
    def exponents(self):
        return self._exponents
    
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
        permutation = list(range(num_open_legs, num_open_legs + node.nvirt_legs() -1))
        permutation.extend(range(0,num_open_legs))
        permutation.append(node.nlegs() -1)

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
        permutation = list(range(num_open_legs + 1, num_open_legs + 1 + node.nvirt_legs() -1))
        permutation.extend(range(1, num_open_legs + 1))
        permutation.append(0)

        return permutation
    
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
        
    def _apply_one_trotter_step(self, index):
        """
        Applies the exponential operator of the Trotter splitting that is
        chosen via index

        Parameters
        ----------
        index : int
            Index in splitting that determines which interaction will be
            applied.

        Returns
        -------
        None.

        """
        assert index < len(self.exponents)
        
        two_site_exponent = self.exponents[index]
        
        operator = two_site_exponent["operator"]
        identifiers = two_site_exponent["site_ids"]
        
        # TODO: Generalise to more than two sites
        if len(identifiers) != 2:
            raise NotImplementedError
        
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
    
    def run_one_time_step(self):
        """
        Running one time_step on the TNS according to the exponentials and the
        splitting provided in the object.
        """   
        for index in self.splitting:
            self._apply_one_trotter_step(index)
            
        # Compute  expectation value
            
    def run(self):
        pass