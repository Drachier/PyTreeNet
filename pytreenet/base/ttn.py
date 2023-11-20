import copy

import numpy as np

from warnings import warn

from .tensornode import assert_legs_matching
from ..utils.canonical_form import canonical_form, orthogonalize, _correct_ordering_of_q_legs
from .tree_contraction import (completely_contract_tree,
                               contract_two_ttn, 
                               contract_two_ttn____,
                               single_site_operator_expectation_value,
                               operator_expectation_value,
                               scalar_product,
                               density
                               )
from ..utils.tensor_util import (tensor_qr_decomposition, set_leg_dimension)
from ..utils.cn import ContractionNetwork

class TreeTensorNetwork(object):
    """
    A tree tensor network (TTN) a tree, where each node contains a tensor,
    that is part of the network. Here a tree tensor network is a dictionary
    _nodes of tensor nodes with their identifiers as keys.

    General structure and parts of the codes are from treelib.tree
    """

    def __init__(self, original_tree = None, deep = False):
        """
        Initiates a new TreeTensorNetwork or a deep or shallow copy of a
        different one.
        """

        self._nodes = dict()

        self._root_id = None
        if original_tree is not None:
            self._root_id = original_tree.root_id

            if deep:
                for node_id in original_tree.nodes:
                    self._nodes[node_id] = copy.deepcopy(original_tree.nodes[node_id])
            else:
                    self._nodes = original_tree.nodes

    @property
    def nodes(self):
        """
        A dictionary containing the tensor trees notes via their identifiers.
        """
        return self._nodes

    @property
    def root_id(self):
        """
        The root's identifier.
        """
        return self._root_id

    @root_id.setter
    def root_id(self, new_root_id):
        """
        Sets a new root_id
        """
        self._root_id = str(new_root_id)

    def __contains__(self, identifier):
        """
        Determines if a node with identifier is in the TTN.
        """
        return identifier in self._nodes

    def __getitem__(self, key):
        """
        Return _nodes[key]
        """
        return self._nodes[key]

    def __len__(self):
        return len(self._nodes)

    def __setitem__(self, node):
        "TODO: Once the update methods are established this can be worked on."

    def check_no_nodeid_dublication(self, node_id):
        """
        Checks if node_id already exists in the TTN
        """
        return node_id not in self.nodes

    def assert_no_nodeid_dublication(self, node_id):
        """
        Asserts if node_id already exists in the TTN
        """
        assert self.check_no_nodeid_dublication(node_id), f"Tensor node with identifier {node_id} already exists in TTN"

    def assert_id_in_tree(self, node_id):
        """
        Asserts if the node with the identifier node_id is in the tree.
        """
        assert node_id in self.nodes, f"Tensor node with identifier {node_id} is not part of the TTN."

    def add_root(self, node):
        """
        Adds a root tensor node to the TTN.
        """
        assert self.root_id == None, "A TTN can't have two roots."
        self._root_id = node.identifier
        self.nodes.update({node.identifier: node})

    def add_child_to_parent(self, child, child_leg, parent_id, parent_leg):
        """
        Adds a tensor node to the TTN which is the child of the tensor node
        with identifier parent_id. The two tensors are contracted along one
        leg. The child via child_leg and the parent via parent_leg
        """
        assert parent_id in self.nodes, f"Parent with identifier {parent_id} has to be part of the TTN."

        parent = self.nodes[parent_id]
        child_id = child.identifier

        self.assert_no_nodeid_dublication(child_id)
        assert_legs_matching(child, child_leg, parent, parent_leg)

        child.open_leg_to_parent(child_leg, parent_id)
        parent.open_leg_to_child(parent_leg, child_id)
        self.nodes.update({child_id: child})

    def add_parent_to_root(self, root_leg, parent, parent_leg):
        """
        Adds the node parent as parent to the TTN's root node. The two
        are contracted. The root via root_leg and the parent via parent_leg.
        The root is updated to be the parent.
        """
        parent_id = parent.identifier
        self.assert_no_nodeid_dublication(parent_id)

        root = self.nodes[self.root_id]
        assert_legs_matching(root, root_leg, parent, parent_leg)

        root.open_leg_to_parent(root_leg, parent_id)
        parent.open_leg_to_child(parent_leg, self.root_id)
        self.nodes.update({parent_id: parent})
        self._root_id = parent_id

    def nearest_neighbours(self):
        """
        Finds all nearest neighbouring nodes in a tree.
        We basically find all parent-child pairs.

        Returns
        -------
        nn: list of tuples of strings.
            A list containing tuples that contain the two identifiers of
            nearest neighbour pairs of nodes.
        """
        nn = []
        
        for node_id in self.nodes:
            current_node = self.nodes[node_id]
            for child_id in current_node.children_legs:
                nn.append((node_id, child_id))
                
        return nn
    
    def conjugate(self):
        """
        Returns a new TTN that is a conjugated version of the current TTN

        Returns
        -------
        ttn_conj:
            A conjugated copy of the current TTN.
        
        """
        
        ttn_conj = TreeTensorNetwork(original_tree=self, deep=True)
        
        for node_id in ttn_conj.nodes:
            
            node = ttn_conj.nodes[node_id]
            node.tensor = np.conj(node.tensor)
            
        return ttn_conj

    def distance_to_node(self, center_node_id):
        """

        Parameters
        ----------
        center_node_id : str
            The identifier of the node to which the distance of all other
            nodes should be determined.

        Returns
        -------
        distance_dict : dictionary(str : int)
            A dictionary with the identifiers of the TNN's nodes as keys and
            their distance to center_node as values

        """
        distance_dict = {center_node_id: 0}
        self.distance_of_neighbours(ignore_node_id=None, distance=1, node_id=center_node_id, distance_dict=distance_dict)
        return distance_dict

    def distance_of_neighbours(self, ignore_node_id, distance, node_id, distance_dict):
        """
        Parameters
        ----------
        ignore_node_id : str
            Identifier of the node to be ignored for the recursion, i.e., the
            distance to it has already been established.
        distance : int
            The distance of the node with identifier node_id to the center_node.
        node_id : str
            The identifier of the node whose neighbours' distances are to be
            checked
        distance_dict : dictionary(str : int)
            A dictionary with the identifiers of the TNN's nodes as keys and
            their distance to center_node as values

        Returns
        -------
        None.

        """
        node = self.nodes[node_id]
        non_ignored_children_id = [child_id for child_id in node.children_legs.keys() if child_id != ignore_node_id]

        children_distance_to_center = {child_id: distance for child_id in non_ignored_children_id}
        distance_dict.update(children_distance_to_center)

        for child_id in children_distance_to_center.keys():
            self.distance_of_neighbours(ignore_node_id=node_id, distance=distance+1, node_id=child_id, distance_dict=distance_dict)

        if not node.is_root():
            parent_id = node.parent_leg[0]
            if not parent_id == ignore_node_id:
                distance_dict.update({parent_id: distance})
                self.distance_of_neighbours(ignore_node_id=node_id, distance=distance+1, node_id=parent_id, distance_dict=distance_dict)

    # TODO implement similar functions in node class.
    def rewire_only_child(self, parent_id, child_id, new_identifier):
        """
        For the node with identifier child_id the parent_leg is rewired from parent
        to a node with identifier new_identifier.

        Parameters
        ----------
        parent_id : str
            Identifier of the parent node for which one child is rewired to a new parent.
        child_id : str
            Identifier of the child which is to be rewired.
        new_identifier : str
            Identifier of the node to be rewired to.

        Returns
        -------
        None.

        """
        parent = self.nodes[parent_id]
        child = self.nodes[child_id]
        assert child_id in parent.children_legs, f"The node with identifier {child_id} is not a child of the node with identifier {parent_id}."
        assert child.parent_leg[0] == parent_id, f"The node with identifier {parent_id} is not the parent of the node with identifier {child_id}."
        child.parent_leg[0] = new_identifier

    def rewire_only_parent(self, child_id, new_identifier):
        """
        For the parent of the node child the leg connected to child is rewired to the
        tensor node with identifier new_identifier.

        Parameters
        ----------
        child_id : str
            Identifier of the node whose parent is to have one leg rewired.
        new_identifier : str
            Identifier of the tensor the parent is rewired to.

        Returns
        -------
        None.

        """
        child = self.nodes[child_id]
        if child.is_root():
            warn(f"The node with identifier {child_id} is a tree's root, so its parent cannot be rewired.")
        else:
            parent_id = child.parent_leg[0]
            parent = self.nodes[parent_id]
            leg_to_child_tensor = {new_identifier: parent.children_legs[child_id]}
            del parent.children_legs[child_id]
            parent.children_legs.update(leg_to_child_tensor)

    def set_bond_dimension(self, node_id1, node_id2, value, mode="set"):
        if node_id2 not in self[node_id1].neighboring_nodes():
            raise ValueError("Node1 and Node2 are not connected!")
        if type(value) != int:
            raise ValueError("Bond dimension must be integer!")

        leg1 = self[node_id1].neighboring_nodes()[node_id2]
        leg2 = self[node_id2].neighboring_nodes()[node_id1]
        set_leg_dimension(self[node_id1], leg1, self[node_id2], leg2, value=value, mode=mode)  

    def analyze_bond_dimension(self, node_id1, node_id2):
        if node_id2 not in self[node_id1].neighboring_nodes():
            raise ValueError("Node1 and Node2 are not connected!")
        
        leg1 = self[node_id1].neighboring_nodes()[node_id2]
        leg2 = self[node_id2].neighboring_nodes()[node_id1]
        bond_dim = self[node_id1].shape[leg1]
        return set_leg_dimension(self[node_id1], leg1, self[node_id2], leg2, mode="analyze")[:bond_dim]
    
    def analyze_all_bond_dimensions(self):
        results = []
        for node_id in self.nodes.keys():
            for child_id in self[node_id].children_legs.keys():
                results.append((node_id, child_id, self.analyze_bond_dimension(node_id, child_id)))
        return results
            
    # Functions below this are just wrappers of external functions that are
    # linked tightly to the TTN and its structure. This allows these functions
    # to be overwritten for subclasses of the TTN with more known structure.
    # The additional sturcture allows for more efficent algorithms than the
    # general case.
    
    def canonical_form(self, orthogonality_center_id):
        """
        Brings the tree_tensor_network in canonical form with

        Parameters
        ----------
        orthogonality_center_id : str
            The id of the tensor node, which sould be the orthogonality center for
            the canonical form

        Returns
        -------
        None.
        """
        canonical_form(self, orthogonality_center_id)
    
    def orthogonalize(self, orthogonality_center_id):
        """
        Brings the tree_tensor_network in orthogonal form with

        Parameters
        ----------
        orthogonality_center_id : str
            The id of the tensor node, which sould be the orthogonality center

        Returns
        -------
        None.
        """
        orthogonalize(self, orthogonality_center_id)
        
    def completely_contract_tree(self, to_copy=False):
        """
        Completely contracts the given tree_tensor_network by combining all
        nodes.
        (WARNING: Can get very costly very fast. Only use for debugging.)

        Parameters
        ----------
        to_copy: bool
            Wether or not the contraction should be perfomed on a deep copy.
            Default is False.

        Returns
        -------
        In case copy is True a deep copy of the completely contracted TTN is
        returned.

        """
        return completely_contract_tree(self, to_copy=to_copy)
        
    def contract_two_ttn(self, other):
        """
        Contracts two TTN with the same structure. Assumes both TTN use the same
        identifiers for the nodes.
        
        Parameters
        ----------
        other : TreeTensorNetwork

        Returns
        -------
        result_tensor: ndarray
            The contraction result.
            
        """
        return contract_two_ttn(self, other)
    
    def contract_two_ttn____(self, other, exclude=(), truncate=True):
        """
        # TODO docstrings
        """
        return contract_two_ttn____(self, other, exclude, re_truncate_to_ttn1=truncate)
        
    def single_site_operator_expectation_value(self, node_id, operator):
        """
        Assuming ttn represents a quantum state, this function evaluates the 
        expectation value of the operator applied to the node with identifier 
        node_id.

        Parameters
        ----------
        node_id : string
            Identifier of a node in ttn.
            Currently assumes the node has a single open leg..
        operator : ndarray
            A matrix representing the operator to be evaluated.

        Returns
        -------
        exp_value: complex
            The resulting expectation value.

        """
        return single_site_operator_expectation_value(self, node_id,
                                                         operator)
    
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
    
    def __mul__(self, other):
        """
        In-place multiply TTN with scalar value.

        Parameters
        ----------
        other: scalar
            Scalar value to multiply the TTN with

        Returns
        -------
        None

        """
        assert np.isscalar(other), "Multiplication with TTN: Value must be scalar."
        open_legs_root = self.nodes[self.root_id].open_legs
        if len(open_legs_root) > 0:
            open_dims_root = [self.nodes[self.root_id].tensor.shape[l] for l in open_legs_root]
            total_open_dim = np.prod(open_dims_root)
        else:
            total_open_dim = 1
        self.nodes[self.root_id].tensor = self.nodes[self.root_id].tensor * np.power(other, 1/total_open_dim)
    
    def __truediv__(self, other):
        """
        In-place divide TTN by scalar value.
        
        Parameters
        ----------
        other: scalar
            Scalar value to divide the TTN by
            
        Returns
        -------
        None

        """
        if other == 0:
            raise ValueError
        self * (1/other)
    
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

    def norm(self):
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
        return self.scalar_product()
    
    def normalize(self):
        """
        In-place divide the TTN by its norm.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self / self.norm()
    
    def outer_product(self):
        """
        # TODO docstrings
        """
        return self.density(exclude=())
    
    def density(self, exclude=()):
        """
        # TODO docstrings
        """
        return density(copy.deepcopy(self), exclude)

    
    def find_path_to_root(self, node_id):
        """
        Get a list of all parent nodes starting from node_id until root is reached.
        """
        path = [node_id] # Starting point

        while not self.nodes[node_id].is_root():
            path.append(self.nodes[node_id].parent_leg[0])
            node_id = self.nodes[node_id].parent_leg[0]
        return path
    
    def path_from_to(self, node1_id, node2_id):
        sub_path_current_center = self.find_path_to_root(node1_id)
        sub_path_next_center = self.find_path_to_root(node2_id)

        combined = sub_path_current_center + sub_path_next_center
        num_of_duplicates = len([j for j in combined if combined.count(j) != 1])//2

        if -num_of_duplicates+1 != 0:
            sub_path_current_center_no_duplicates = sub_path_current_center[:-num_of_duplicates+1]
        else:
            sub_path_current_center_no_duplicates = sub_path_current_center[::]

        sub_path_next_center_no_duplicates = sub_path_next_center[:-num_of_duplicates]
        sub_path_next_center_no_duplicates.reverse()

        sub_path = sub_path_current_center_no_duplicates + sub_path_next_center_no_duplicates
        
        return sub_path

    def orthogonalize_sequence(self, sequence, node_change_callback=None):
        if type(sequence) != list:
            sequence = [sequence]
        for node_id in sequence:
            if self.orthogonality_center_id != node_id:
                self._orthogonalize_to_node(self.orthogonality_center_id, node_id)
                if node_change_callback is not None:
                    node_change_callback(self.orthogonality_center_id)
                    node_change_callback(node_id)
                self.orthogonality_center_id = node_id
    
    def _orthogonalize_to_node(self, node_id_old, node_id_new):
        """
        Move orthogonalization center from node_id_old to node_id_new.
        This code is essentially just a QR decomp and some leg sorting.
        """
        assert self.orthogonality_center_id == node_id_old

        node_old_node = self[node_id_old]
        index_of_node_new_in_node_old = node_old_node.neighbouring_nodes()[node_id_new]

        all_leg_indices = list(range(0, node_old_node.tensor.ndim))
        all_leg_indices.remove(index_of_node_new_in_node_old)

        q, r =  tensor_qr_decomposition(node_old_node.tensor, all_leg_indices, [index_of_node_new_in_node_old], keep_shape=True)

        reshape_order = _correct_ordering_of_q_legs(node_old_node, index_of_node_new_in_node_old)
        node_old_node.tensor = np.transpose(q, axes=reshape_order)

        legs_of_target_node = self[node_id_new].neighbouring_nodes()
        node_index_to_contract = legs_of_target_node[node_id_old]
        self[node_id_new].absorb_tensor(r, (1,), (node_index_to_contract,))


class QuantumTTState(TreeTensorNetwork):
    def __init__(self, original_tree=None, deep=False):
        super().__init__(original_tree, deep)

        self.orthogonality_center_id = None

    def orthogonalize(self, orthogonality_center_id):
        self.orthogonality_center_id = orthogonality_center_id
        return super().orthogonalize(orthogonality_center_id)
    
    def reduced_density_matrix(self, qubit_ids):

        return self.density(exclude=qubit_ids)

        cn = ContractionNetwork()
        cn.from_state(self, do_not_connect=qubit_ids)
        nodes = cn.contract()
        keys = list(nodes.keys())

        if len(keys) == 2:
            bra_id = "bra" in keys[1]
            ket_id = 1 - bra_id

            bra_tensor = nodes[keys[bra_id]].tensor
            bra_vector = np.reshape(bra_tensor, np.prod(bra_tensor.shape))

            ket_tensor = nodes[keys[ket_id]].tensor
            ket_vector = np.reshape(ket_tensor, np.prod(ket_tensor.shape))

            return np.outer(ket_vector, bra_vector)
        elif len(keys) == 1:
            key = keys[0]

            tensor = nodes[keys[0]].tensor
            tensor = np.reshape(tensor, (4,)*len(qubit_ids))  # TODO dimension is fixed to (d=2)^2, may not always be the case?

            qubit_keys = ["open_" + id for id in qubit_ids]
            qubit_keys_sorted = list(nodes[key].connected_legs.keys())

            leg_order = [qubit_keys_sorted.index(key) for key in qubit_keys]
            tensor = np.transpose(tensor, leg_order)
            tensor = np.reshape(tensor, (2,)*2*len(qubit_ids))
            tensor = np.transpose(tensor, [i for i in range(tensor.ndim) if i%2==0] + [i for i in range(tensor.ndim) if i%2==1])
            tensor = np.reshape(tensor, (2**len(qubit_ids), 2**len(qubit_ids)))
            return tensor
        else:
            raise ValueError
        
    def apply_tto(self, tto, truncate=True):
        contract_two_ttn____(self, tto, re_truncate_to_ttn1=truncate)
            


class QuantumTTOperator(TreeTensorNetwork):
    def __init__(self, original_tree=None, deep=False):
        super().__init__(original_tree, deep)

