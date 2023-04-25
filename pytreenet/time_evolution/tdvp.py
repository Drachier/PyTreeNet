from copy import deepcopy

import numpy as np

from .time_evolution import TimeEvolutionAlgorithm, time_evolve
from ..tensor_util import (tensor_qr_decomposition, tensor_matricization)
from ..ttn import TreeTensorNetwork
from ..canonical_form import _correct_ordering_of_q_legs

"""
Implements the time-dependent variational principle TDVP for tree tensor
networks.

Reference:
    D. Bauernfeind, M. Aichhorn; "Time Dependent Variational Principle for Tree
    Tensor Networks", DOI: 10.21468/SciPostPhys.8.2.024
"""


class TDVP(TimeEvolutionAlgorithm):
    def __init__(self, state: TreeTensorNetwork, hamiltonian: TreeTensorNetwork, time_step_size, final_time, operators=None, mode="1site") -> None:
        """
        Parameters
        ----------
        state : TreeTensorNetwork
            A pytreenet/ttn.py TreeTensorNetwork object.
        hamiltonian: Hamiltonian
            A pytreenet/hamiltonian.py Hamiltonian object.
        time_step_size : float
            Size of each time-step.
        final_time : float
            Total time for which TDVP should be run.
        operators: list of dict
            A list containing dictionaries that contain node identifiers as keys and single-site
            operators as values. Each represents an operator that will be
            evaluated after every time step.
        mode : str, optional
            Decides which version of the TDVP is run. The options are 1site and
            2site. The default is "1site".
        """

        assert len(state.nodes) == len(hamiltonian.nodes)
        self.hamiltonian = hamiltonian

        self.mode = mode

        super().__init__(state, operators, time_step_size, final_time)

        #     LEG ORDER: PARENT CHILDREN PHYSICAL

        self._find_tdvp_update_path()
        self._find_tdvp_orthogonalization_path()

        self.site_cache = dict()
        self._init_site_cache()

        self.partial_tree_caching = True
        self.partial_tree_cache = dict()

        # Caching for speed up
        self.neighbouring_nodes = dict()
        for node_id in self.state.nodes.keys():
            self.neighbouring_nodes[node_id] = deepcopy(self.state[node_id].neighbouring_nodes())
    
    def _init_site_cache(self):
        """
        Contracts state, hamiltonian and state.conjugate() for each site and saves
        in the self.site_cache dict.
        """
        for node_id in self.state.nodes.keys():
            self._update_site_cache(node_id)

    def _update_site_cache(self, node_id):
        """"
        Call after any tensor has been updated to refresh the site cache.
        """
        ket = self.state
        ham = self.hamiltonian

        ketham_tensor = np.tensordot(ket[node_id].tensor, ham[node_id].tensor, axes=(ket[node_id].physical_leg, ham[node_id].physical_leg_ket))
        kethambra_tensor = np.tensordot(ketham_tensor, ket[node_id].tensor.conj(), axes=(ket[node_id].tensor.ndim-1 + ham[node_id].physical_leg_bra, ket[node_id].physical_leg))

        num_cached_tensor_legs = kethambra_tensor.ndim // 3

        ordered_legs = []
        for leg_num in range(num_cached_tensor_legs):
            for j in [0,1,2]:
                ordered_legs.append(leg_num + j*num_cached_tensor_legs)
        
        kethambra_tensor = kethambra_tensor.transpose(ordered_legs)

        shape = []
        for leg_num in range(num_cached_tensor_legs):
            shape.append(np.prod([kethambra_tensor.shape[leg_num + j] for j in [0,1,2]]))

        self.site_cache[node_id] = kethambra_tensor.reshape(shape)
    
    def _find_tdvp_update_path(self):
        """
        Returns a list of all nodes - ordered so that a TDVP update along
        this path requires a minimal amount of orthogonalizations.
        """
        # Start with leaf furthest from root.
        distances_from_root = self.state.distance_to_node(self.state.root_id)
        start = max(distances_from_root, key=distances_from_root.get)

        # Move from start to root. Start might not be exactly start, but another leaf with same distance. 
        sub_path = self._find_tdvp_path_from_leaves_to_root(start)
        self.update_path = [] + sub_path + [self.state.root_id]
        
        branch_roots = [x for x in self.state[self.state.root_id].children_legs.keys() if x not in self.update_path]

        sub_paths = []
        for branch_root in branch_roots:
            sub_paths.append(self._find_tdvp_path_from_leaves_to_root(branch_root).reverse())
        
        sub_paths.sort(key=lambda x: -len(x))
        for sub_path in sub_paths:
            self.update_path = self.update_path + sub_path

    def _find_tdvp_path_from_leaves_to_root(self, any_child):
        path_from_child_to_root = self.state.find_path_to_root(any_child)
        branch_origin = path_from_child_to_root[-2]

        path = self._find_tdvp_path_for_branch(branch_origin)
        return path

    def _find_tdvp_path_for_branch(self, branch_origin, path=[]):
        node = branch_origin
        children = self.state[node].children_legs.keys()
        for child in children:
            path = self._find_tdvp_path_for_branch(child, path)
        path.append(node)
        return path

    def _find_tdvp_orthogonalization_path(self):
        self.orthogonalization_path = []
        for i in range(len(self.update_path)-1):
            sub_path = self.state.path_from_to(self.update_path[i], self.update_path[i+1])
            self.orthogonalization_path.append(sub_path[1::])
    
    def _orthogonalize_init(self):
        if self.state.orthogonality_center_id is None:
            self.state.orthogonality_center_id = self.update_path[0]
            self.state.orthogonalize(self.state.orthogonality_center_id)
        else:
            path = self.state.path_from_to(self.state.orthogonality_center_id, self.update_path[0])
            self.state.orthogonalize_sequence(path)

    def _contract_partial_tree(self, start=None, end=None, tensor_parent=None, leg_parent=None, connecting_from_root=False):
        if start is None or start == self.state.root_id:
            connecting_from_root = True
            start = self.state.root_id

        if self.partial_tree_caching and str(start)+str(end) in self.partial_tree_cache.keys():
            return self.partial_tree_cache[str(start)+str(end)]
        
        if tensor_parent is None:
            tensor = self.site_cache[start]
        else:
            leg_child = self.state[start].parent_leg[1]
            tensor = np.tensordot(tensor_parent, self.site_cache[start], axes=(leg_parent, leg_child))

        for child_id in self.state[start].children_legs.keys():
            if child_id != end:
                leg = self.state[start].children_legs[child_id] - connecting_from_root
                tensor = self._contract_partial_tree(child_id, end, tensor, leg, connecting_from_root)

        if self.partial_tree_caching:
            self.partial_tree_cache[str(start)+str(end)] = tensor
        return tensor
    
    def _contract_all_except_node(self, target_node_id):
        # Contract bra-ham-ket for each site and then contract TTN:
        #               Step 1: Contract bra-ham-ket for each site except target_node_id
        #                   (self.site_cache)
        #               Step 2: Build different trees with roots: every child of target_node_id
        #                       and self.state.root_id (cut off at target_node_id)
        #               Step 3: Connect parts via Hamiltonian at target_node_id site
        #               Step 4: Sort legs and let's go!

        # TODO imporve root setter so that parent legs and stuff makes actually sense
        
        target_node = self.state[target_node_id]
        target_hamiltonian = self.hamiltonian[target_node_id]
        tensor = target_hamiltonian.tensor
        tensor_added_legs = 0

        # Step 1: Check if node has parents and if yes build parent tree
        if target_node_id != self.state.root_id:
            parent_part = self._contract_partial_tree(start=self.state.root_id, end=target_node_id)
            parent_part = parent_part.reshape([target_node.shape()[target_node.parent_leg[1]], target_hamiltonian.shape()[target_hamiltonian.parent_leg[1]], target_node.shape()[target_node.parent_leg[1]]])
            tensor = np.tensordot(parent_part, tensor, axes=(1, target_hamiltonian.parent_leg[1]))
            tensor_added_legs += 1

        # Step 2: build children trees:
        children_trees = []
        for child_id in self.state[target_node_id].children_legs.keys():
            child_part = self._contract_partial_tree(start=child_id, end=None)
            child_part = child_part.reshape([target_node.shape()[target_node.children_legs[child_id]], target_hamiltonian.shape()[target_hamiltonian.children_legs[child_id]], target_node.shape()[target_node.children_legs[child_id]]])
            children_trees.append(child_part)

            tensor = np.tensordot(tensor, child_part, axes=(target_hamiltonian.children_legs[child_id]+tensor_added_legs, 1))
            tensor_added_legs += 1
        
        """
        leg order is:
            parent_part: ket, bra,  hamiltonian: ket, bra, child1: ket, bra, child2: ...
        """
        ordered_legs = list(range(tensor.ndim))
        ordered_legs[2:] = ordered_legs[4:] + [2, 3]
        tensor = tensor.transpose(ordered_legs)
        """
        leg order is:
            parent, children, pphysical
        """
        return tensor


    def _get_effective_site_hamiltonian(self, node_id):
        tensor = self._contract_all_except_node(node_id)
        
        ket_legs = [2*i for i in range(tensor.ndim//2)] 
        bra_legs = [2*i+1 for i in range(tensor.ndim//2)]

        return tensor_matricization(tensor, bra_legs, ket_legs)
    
    def _get_effective_link_hamiltonian(self, node_id, next_node_id):
        tensor = self._contract_all_except_node(node_id)
        
        tensor_ket_legs = [2*i for i in range(tensor.ndim//2)] 
        node_leg_of_next_node = self.neighbouring_nodes[node_id][next_node_id]
        tensorket_legs_without_next_node = [i for i in tensor_ket_legs if i//2 != node_leg_of_next_node]
        site_ket_legs_without_next_node = [i for i in range(self.state[node_id].tensor.ndim) if i != node_leg_of_next_node]

        tensor = np.tensordot(self.state[node_id].tensor, tensor, axes=(site_ket_legs_without_next_node, tensorket_legs_without_next_node))

        tensor_bra_legs_without_next_node = [i+2 for i in site_ket_legs_without_next_node]
        site_bra_legs_without_next_node = site_ket_legs_without_next_node
        tensor = np.tensordot(tensor, np.conj(self.state[node_id].tensor), axes=(tensor_bra_legs_without_next_node, site_bra_legs_without_next_node))
        
        return tensor_matricization(tensor, (2, 3), (0, 1), correctly_ordered=False)
    
    def _update_site_and_get_link(self, node_id, next_node_id):
        node = self.state[node_id]
        index_of_next_node_in_node = self.neighbouring_nodes[node_id][next_node_id]

        all_leg_indices = list(range(0, node.tensor.ndim))
        all_leg_indices.remove(index_of_next_node_in_node)

        q, r =  tensor_qr_decomposition(node.tensor, all_leg_indices, [index_of_next_node_in_node], keep_shape=True)

        reshape_order = _correct_ordering_of_q_legs(node, index_of_next_node_in_node)
        self.state[node_id].tensor = np.transpose(q, axes=reshape_order)
        self._update_site_cache(node_id)

        return r
    
    def _update(self, node_id, next_node_id):
        assert self.state.orthogonality_center_id == node_id

        hamiltonian_eff_site = self._get_effective_site_hamiltonian(node_id)

        psi = self.state[node_id].tensor
        self.state[node_id].tensor = time_evolve(psi, hamiltonian_eff_site, self.time_step_size, forward=True)
        self._update_site_cache(node_id)

        if next_node_id is None:
            return

        link_tensor = self._update_site_and_get_link(node_id, next_node_id)

        hamiltonian_eff_link = self._get_effective_link_hamiltonian(node_id, next_node_id)

        link_tensor = time_evolve(link_tensor, hamiltonian_eff_link, self.time_step_size, forward=False)

        self.state[next_node_id].absorb_tensor(link_tensor, 1, self.neighbouring_nodes[next_node_id][node_id])
        self._update_site_cache(next_node_id)

    def run_one_time_step(self):
        for i, node_id in enumerate(self.update_path):
            
            # Orthogonalize
            if i==0:
                self._orthogonalize_init()
                self.partial_tree_cache = dict()
            else:
                self.state.orthogonalize_sequence(self.orthogonalization_path[i-1])

            # Select Next Node
            if i+1 < len(self.update_path):
                next_node_id = self.update_path[i+1]
            else:
                next_node_id = None

            # Update
            self._update(node_id, next_node_id)

        # TODO consider self.mode



