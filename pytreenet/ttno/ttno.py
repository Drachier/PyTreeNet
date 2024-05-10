from __future__ import annotations
from typing import Dict
from enum import Enum
import numpy as np

from ..core.ttn import TreeTensorNetwork
from ..util.tensor_splitting import (tensor_qr_decomposition,
                                     tensor_svd,
                                     truncated_tensor_svd,
                                     SVDParameters)
from ..core.node import Node
from .state_diagram import StateDiagram, TTNOFinder

class Decomposition(Enum):
    SVD = "SVD"
    QR = "QR"
    tSVD = "tSVD"

class TTNO(TreeTensorNetwork):
    """
    Represents a tree tensor network operator, that is
    an operator in tensor network form with an
    underlying tree structure. Every node in this
    TTN has zero or two open legs of equal dimension.

    Attributes
    ----------
    hamiltonian : Hamiltonian
        The Hamiltonian which is to be brought into TTNO form. Should contain
        a conversion dictionary to allow for better compression.
    reference_tree : TreeTensorNetwork
        A TTN which has the same underlying tree topology as the TTNO is
        supposed to have.

    """

    def __init__(self):
        super().__init__()

    @classmethod
    def from_hamiltonian(cls, hamiltonian: Hamiltonian,
                            reference_tree: TreeStructure,
                            method: TTNOFinder = TTNOFinder.CM ) -> TTNO:
        """
        Generates a TTNO from a Hamiltonian.

        Args:
            hamiltonian (Hamiltonian): The Hamiltonian, to which the TTNO
             should be equivalent.
            reference_tree (TreeStructure): The tree structure which the TTNO
             should respect.

        Returns:
            TTNO: The resulting TTNO.
        """
        hamiltonian = hamiltonian.pad_with_identities(reference_tree)
        state_diagram = StateDiagram.from_hamiltonian(hamiltonian,
                                                      reference_tree, method)
        ttno = TTNO()
        root_id = reference_tree.root_id
        root_shape = state_diagram.obtain_tensor_shape(root_id,
                                                       hamiltonian.conversion_dictionary)
        root_tensor = np.zeros(root_shape, dtype=complex)
        root_node = Node(identifier=root_id)
        ttno.add_root(root_node, root_tensor)
        for child_id in reference_tree.nodes[root_id].children:
            ttno._rec_zero_ttno(child_id, state_diagram,
                                hamiltonian.conversion_dictionary)
        # Now we have a TTNO filled with zero tensors of the correct shape.
        state_diagram.set_all_vertex_indices() # Fixing index_values
        for he in state_diagram.get_all_hyperedges():
            position = he.find_tensor_position(reference_tree)
            operator = hamiltonian.conversion_dictionary[he.label]
            ttno.tensors[he.corr_node_id][position] += operator
        return ttno

    def _rec_zero_ttno(self, node_id: str, state_diagram: StateDiagram,
                       conversion_dict: Dict[str, np.ndarray]):
        """
        Recursively creates an empty TTNO, i.e., one with all zeros as entries.

        Args:
            root_id (str): The identifier of the current_node.
            state_diagram (StateDiagram): A state diagram used to determine the
             dimensions and the tree topology.
            conversion_dict (Dict[str, np.ndarray]): A conversion dictionary to
             determine the physical dimensions required.
        """
        tensor_shape = state_diagram.obtain_tensor_shape(node_id,
                                                         conversion_dict)
        node_tensor = np.zeros(tensor_shape, dtype=complex)
        node = Node(identifier=node_id)
        parent_id = state_diagram.reference_tree.nodes[node_id].parent
        parent_leg = self.nodes[parent_id].nneighbours()
        self.add_child_to_parent(node, node_tensor, 0, parent_id, parent_leg)
        for child_id in state_diagram.reference_tree.nodes[node_id].children:
            self._rec_zero_ttno(child_id, state_diagram, conversion_dict)

    @classmethod
    def from_tensor(
            cls, reference_tree: TreeStructure, tensor: np.nadarray, leg_dict: dict[str, int],
            mode: Decomposition = Decomposition.QR):
        """
        Parameters
        ----------
        reference_tree : TreeStructure
            A tree used as a reference. The TTNO will have the same underlying
            tree structure and the same node_ids.
        tensor : ndarray
            A big numpy array that represents an operator on a lattice of
            quantum systems. Therefore it has to have an even number of legs.
            Assumed to be a reshape of operator, such that input and output dimensions
            are clustered as [out_1, out_2, ..., out_n, in_1, in_2, ..., in_n] and
            out_i == in_i for all dimensions.
        leg_dict : dict
            A dictionary containing node_identifiers (str) as keys and leg indices (int)
            as values. It is used to match the legs of tensor to the different
            nodes. Only the lower (smaller) half of the legs is to be put in this dict,
            the others are inferred by ordering.
        mode: str
            Can be either 'QR' or 'SVD'. Determines the decomposition to use.
            'SVD' allows for truncation of trivial singular values, while 'QR'
            is faster.

        Returns
        -------
        new_TTNO: TTNO

        """
        # Ensure that legs are even
        assert tensor.ndim % 2 == 0
        half_dim = int(tensor.ndim / 2)
        # Ensure that operator matches number of lattice sites in TTN
        assert half_dim == len(reference_tree.nodes)
        # Ensure that tensor input and output dimensions are the same
        # assert tensor.shape[:half_dim] == tensor.shape[half_dim:]

        root_id = reference_tree.root_id
        new_leg_dict = {node_id: [leg_dict[node_id], half_dim + leg_dict[node_id]]
                        for node_id in leg_dict}

        tensor_shape = cls._get_qr_decomposition_shape(reference_tree, new_leg_dict, [], root_id)

        tensor = np.transpose(tensor, axes=tensor_shape)

        root_node = Node(tensor, identifier=root_id)

        new_TTNO = TTNO()
        new_TTNO.add_root(root_node, tensor)

        new_TTNO._from_tensor_rec(
            reference_tree, root_node, mode=mode)
        return new_TTNO

    @classmethod
    def _get_qr_decomposition_shape(
            cls, reference_tree: TreeStructure, leg_dict: dict[str, list[int]],
            shape_tensor: list[int],
            current_id: str) -> list[int]:

        if reference_tree.nodes[current_id].is_leaf():
            shape_tensor = leg_dict[current_id] + shape_tensor
            # shape_tensor.extend(leg_dict[current_id])
            return shape_tensor

        for children_id in reference_tree.nodes[current_id].children:
            shape_tensor = cls._get_qr_decomposition_shape(reference_tree, leg_dict, shape_tensor, children_id)

        # shape_tensor.extend(leg_dict[current_id])
        shape_tensor = leg_dict[current_id] + shape_tensor

        return shape_tensor

    def _from_tensor_rec(
            self, reference_tree: TreeStructure, current_node: str,
            mode: Decomposition = Decomposition.QR):
        """
        Recursive part to obtain a TTNO from a tensor. For each child of the
        current node a new node is defined via a Matrix-decomposition and the
        current_node and the leg_dict are modified.

        Parameters
        ----------
        reference_tree : TreeStructure
            A tree used as a reference. The TTNO will have the same underlying
            tree structure and the same node_ids.
        current_node : Node
            The current node which we want to split via a QR-decomposition.
        leg_dict : dict
            A dictionary it contains node_identifiers as keys and leg indices
            as values. It is used to match the legs of tensor in the current
            node to the different nodes it will be split into.
        mode: str
            Can be either 'QR' or 'SVD', 'tSVD'. Determines the decomposition to use.
            'SVD' & 'tSVD' allows for truncation of trivial singular values, while 'QR'
            is faster.

        Returns
        -------
        None.

        """
        # At a leaf, we can immediately stop
        current_node_id = current_node.identifier
        if reference_tree.nodes[current_node_id].is_leaf():
            return

        current_children = reference_tree.nodes[current_node_id].children
        current_node = self.nodes[current_node.identifier]
        current_tensor = self.tensors[current_node_id]

        for child_id in current_children:
            n_recursive_children = 2*reference_tree.find_subtree_size_of_node(child_id)
            q_legs = list(range(current_tensor.ndim-n_recursive_children))
            r_legs = list(range(current_tensor.ndim-n_recursive_children, current_tensor.ndim))
            if mode == Decomposition.QR:
                Q, R = tensor_qr_decomposition(current_tensor, q_legs, r_legs)
            elif mode == Decomposition.SVD:
                Q, S, Vh = tensor_svd(current_tensor, q_legs, r_legs)
                R = np.tensordot(np.diag(S), Vh, axes=(1, 0))
            elif mode == Decomposition.tSVD:
                Q, S, Vh = truncated_tensor_svd(current_tensor, q_legs, r_legs,
                                                SVDParameters())
                R = np.tensordot(np.diag(S), Vh, axes=(1, 0))
            else:
                raise ValueError(f"{mode} is not a valid keyword for mode.")

            # Replace current_node with q_node in the TTNO
            self.nodes[current_node_id].link_tensor(Q)
            self.tensors[current_node_id] = Q

            r_node = Node(R, identifier=child_id)

            # We have to add this node as an additional child to the current node
            self.add_child_to_parent(
                r_node, R, 0, current_node.identifier, Q.ndim - 1)

            self._from_tensor_rec(reference_tree, r_node, mode=mode)

            # Prepare to repeat for next child, this transposes the tensor to the correct shape
            current_tensor = self.tensors[current_node_id]

        return

    @staticmethod
    def _prepare_legs_for_QR(reference_tree, current_node, child_id, leg_dict):
        """
        Prepares the lists of legs to be used in the QR-decomposition and
        splits the leg_dict. One of the new dictionary contains the legs
        belonging to the Q-tensor and the other contains the legs belonging to
        the R-tensor. The legs belonging to the child node are stored in the 
        R-tensor. The remaining legs are grouped into the Q-tensor.

        Parameters
        ----------
        reference_tree : TreeTensorNetwork
            A tree used as a reference. The TTNO will have the same underlying
            tree structure and the same node_ids.
        current_node : TensorNode
            The current node which we want to split via a QR-decomposition.
        child_id : string
            Identifier of the child, which we want to obtain for the TTNO by#
            splitting it of the current_node via a QR-decomposition.
        leg_dict : dict
            A dictionary it contains node_identifiers as keys and leg indices
            as values. It is used to match the legs of tensor in the current
            node to the different nodes it will be split into.

        Returns
        -------
        q_legs: list of int
            The legs of the current_tensor which are to be associated to the
            Q-tensor. (The ones that remain with the current_node)
        r_legs: list of int
            The legs of the current_tensor which are to be associated to the
            R-tensor. (The legs of the to-be created child)
        q_leg_dict, r_leg_dict: dict
            Dictionary describing to which node the open legs of the tensor in
            the Q(R)-node will belong to.
        """

        subtree_list_child = reference_tree.find_subtree_of_node(child_id)

        q_legs = []
        r_legs = []
        # Required to keep the leg ordering later as open_legs_in, open_legs_out
        q_legs_sec = []
        r_legs_sec = []

        r_leg_dict = {}
        q_leg_dict = {}

        for node_id_temp in leg_dict:
            if node_id_temp in subtree_list_child:
                r_legs.append(leg_dict[node_id_temp][0])
                r_legs_sec.append(leg_dict[node_id_temp][1])
                r_leg_dict[node_id_temp] = leg_dict[node_id_temp]
            else:
                q_legs.append(leg_dict[node_id_temp][0])
                q_legs_sec.append(leg_dict[node_id_temp][1])
                q_leg_dict[node_id_temp] = leg_dict[node_id_temp]

        q_legs.extend(q_legs_sec)
        r_legs.extend(r_legs_sec)

        if not current_node.is_root():
            q_legs.append(current_node.parent_leg[1])
        q_legs.extend(list(current_node.children_legs.values()))

        return q_legs, r_legs, q_leg_dict, r_leg_dict

    @staticmethod
    def _create_q_node(tensor, current_node, q_leg_dict):
        """
        Prepares the new TensorNode which will consist of the Q of the
        QR-decomposition and be in the place of the current_node.

        Parameters
        ----------
        tensor : ndarray
            Tensor associated to the new node.
        current_node : TensorNode
            The node that is currently in the process of being split into two.
        q_leg_dict : dict
            Dictionary describing to which node the open legs of the tensor in
            the new node will belong to.

        Returns
        -------
        q_node : TensorNode
            The new tensor node that will replace the current_node.

        """

        q_node = TensorNode(tensor, identifier=current_node.identifier)
        q_half_leg = len(q_leg_dict)
        if not current_node.is_root():
            q_node.open_leg_to_parent(2*q_half_leg, current_node.parent_leg[0])
            c = 1
        else:
            c = 0

        q_node.open_legs_to_children(list(range(2*q_half_leg+c, 2*q_half_leg+c + len(current_node.children_legs))),
                                     list(current_node.children_legs.keys()))

        return q_node

    @staticmethod
    def _find_new_leg_dict(leg_dict, c=0):
        """
        Used to find the Dictionary corresponding to the new tensor created
        in the R-part of the QR-decomposition

        Parameters
        ----------
        leg_dict : dict
            Subdictionary of leg_dict. Keys are node_ids and the values are
            leg indices of open legs of the tensor in the current node, before
            QR-decomposition
        c : int
            Shift when finding leg dict for non-root node. Set `c` to 1 to avoid adding
            leg connected to parent node to updated leg dict.

        Returns
        -------
        new_leg_dict: dict
            Keys are node_ids and the values are leg indices of open legs in
            the newly obtained tensor node.

        """

        half_leg = len(leg_dict)
        new_leg_dict = {}

        count = 0
        while len(leg_dict) > 0:
            key_w_min_index = ""
            min_index = float("inf")

            for node_id in leg_dict:
                if leg_dict[node_id][0] < min_index:
                    key_w_min_index = node_id
                    min_index = leg_dict[node_id][0]

            # Every node will have two open legs belonging to the
            # same other node are exactly half_leg apart.
            new_leg_dict[key_w_min_index] = [c + count, c + count + half_leg]

            count += 1

            del leg_dict[key_w_min_index]

        return new_leg_dict

TreeTensorNetworkOperator = TTNO
