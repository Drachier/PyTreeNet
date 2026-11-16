"""
Provides the class to represent a Tree Tensor Network Operator (TTNO).
"""
from __future__ import annotations
from typing import Dict, Tuple, List, Union
from enum import Enum
import numpy as np

from ..core.ttn import TreeTensorNetwork
from ..core.tree_structure import TreeStructure
from ..core.node import Node
from ..core.leg_specification import LegSpecification
from ..operators.hamiltonian import Hamiltonian
from ..util.tensor_splitting import (tensor_qr_decomposition,
                                     tensor_svd,
                                     truncated_tensor_svd,
                                     SVDParameters)
from .state_diagram import StateDiagram, TTNOFinder, OperatorRewrite

__all__ = ["Decomposition", "TTNO", "TreeTensorNetworkOperator"]


class Decomposition(Enum):
    """
    An enumeration to choosing decomposition method used for TTNO construction.

    Attributes:
        SVD: Singular Value Decomposition
        QR: QR-Decomposition
        tSVD: Truncated Singular Value Decomposition
    """
    SVD = "SVD"
    QR = "QR"
    tSVD = "tSVD"

class TreeTensorNetworkOperator(TreeTensorNetwork):
    """
    Represents a tree tensor network operator (TTNO).

    A TTNO is a tree tensor network equivalent to an operator. This means every
    node has two open legs (input and output). The legs can be trivial, i.e.,
    of dimension 1, but have the same dimension.
    """

    def as_matrix(self,
                  order: List[str] | None = None
                  ) -> Tuple[np.ndarray, List[str]]:
        """
        Contracts the matrix equivalent to the TTNO.

        The new order of the dimension is also returned.
        """
        tensor, order = self.completely_contract_tree(to_copy=True, order=order)
        permutation = list(range(0,tensor.ndim,2)) + list(range(1,tensor.ndim,2))
        tensor = np.transpose(tensor, permutation)
        dim = np.prod(tensor.shape[:int(tensor.ndim/2)])
        tensor = tensor.reshape(dim, dim)
        return tensor, order

    @classmethod
    def from_hamiltonian(cls,hamiltonian: Hamiltonian,
                             reference_tree: TreeStructure,
                             method: TTNOFinder = TTNOFinder.SGE,
                             dtype: np.dtype = complex,
                             rewrite: OperatorRewrite = OperatorRewrite.NONE,
                             tol: Union[float, None] = None
                             ) -> TreeTensorNetworkOperator:
        """
        Generates a TTNO from a Hamiltonian.

        Args:
            hamiltonian (Hamiltonian): The Hamiltonian, to which the TTNO
                should be equivalent.
            reference_tree (TreeStructure): The tree structure which the TTNO
                should respect.
            method (TTNOFinder): The construction method of the state diagram.
            dtype (np.dtype): The data type of the resulting tensors.
            rewrite (OperatorRewrite, optional): Whether to rewrite the terms
                over operators that are linearly independent on every node
                before the state diagram is built, which is the condition
                under which the construction reaches the operator-Schmidt
                rank. Defaults to OperatorRewrite.NONE, so the TTNO represents
                the terms as they were given.
            tol (float | None, optional): The tolerance of the dependence
                test, ignored unless the policy rewrites. Defaults to None.

        Returns:
            TreeTensorNetworkOperator: The resulting TTNO.
        """
        hamiltonian = hamiltonian.pad_with_identities(reference_tree)
        state_diagram = StateDiagram.from_hamiltonian(hamiltonian,
                                                      reference_tree, method,
                                                      rewrite=rewrite, tol=tol)
        chosen = state_diagram.hamiltonian
        return cls.from_state_diagram(state_diagram,
                                      chosen.conversion_dictionary,
                                      chosen.coeffs_mapping,
                                      dtype)

    @classmethod
    def from_state_diagram(cls,
                           state_diagram: StateDiagram,
                           conversion_dict: Dict[str, np.ndarray],
                           coeffs_mapping: Dict[str, complex],
                           dtype: np.dtype = complex
                           ) -> TreeTensorNetworkOperator:
        """
        Generates a TTNO from a state diagram.

        Args:
            state_diagram (StateDiagram): The state diagram, which the TTNO
                should represent.
            conversion_dict (Dict[str, np.ndarray]): A conversion dictionary to
                determine the physical dimensions required.
            coeffs_mapping (Dict[str, complex]): A mapping of term coefficients
                to their corresponding values.

        Returns:
            TreeTensorNetworkOperator: The resulting TTNO.
        """
        ttno = cls()
        reference_tree = state_diagram.reference_tree
        root_id = reference_tree.root_id
        root_shape = state_diagram.obtain_tensor_shape(root_id,
                                                       conversion_dict)
        root_tensor = np.zeros(root_shape, dtype=dtype)
        root_node = Node(identifier=root_id)
        ttno.add_root(root_node, root_tensor)
        for child_id in reference_tree.nodes[root_id].children:
            ttno._rec_zero_ttno(child_id, state_diagram,
                                conversion_dict, dtype)
        # Now we have a TTNO filled with zero tensors of the correct shape.
        state_diagram.set_all_vertex_indices() # Fixing index_values

        for he in state_diagram.get_all_hyperedges():
            position = he.find_tensor_position(reference_tree)
            operator = conversion_dict[he.label].astype(dtype)
            operator *= np.array(he.lambda_coeff * coeffs_mapping[he.gamma_coeff],dtype = dtype)
            ttno.tensors[he.corr_node_id][position] += operator
        return ttno

    def _rec_zero_ttno(self, node_id: str, state_diagram: StateDiagram,
                       conversion_dict: Dict[str, np.ndarray],
                       dtype: np.dtype = complex):
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
        node_tensor = np.zeros(tensor_shape, dtype=dtype)
        node = Node(identifier=node_id)
        parent_id = state_diagram.reference_tree.nodes[node_id].parent
        parent_leg = self.nodes[parent_id].nneighbours()
        self.add_child_to_parent(node, node_tensor, 0, parent_id, parent_leg)
        for child_id in state_diagram.reference_tree.nodes[node_id].children:
            self._rec_zero_ttno(child_id, state_diagram, conversion_dict, dtype)

    @classmethod
    def from_tensor(
            cls, reference_tree: TreeStructure, tensor: np.nadarray, leg_dict: dict[str, int],
            mode: Decomposition = Decomposition.QR):
        """
        Generate a TTNO from a big tensor.

        The TTNO is generated by recursively splitting the tensor into the
        desired tree shape using tensor decompositions.

        Args:
        reference_tree (TreeStructure): A tree used as a reference. The TTNO
            will have the same underlying tree structure and the same node_ids.
        tensor (np.ndarray): A big numpy array that represents an operator on a
            lattice of quantum systems. Therefore it has to have an even number
            of legs. Assumed to be a reshape of operator, such that input and
            output legs are clustered as 
            ``[out_1, out_2, ..., out_n, in_1, in_2, ..., in_n]``
            and ``dim(out_i) == dim(in_i)`` for all legs.
        leg_dict (dict): A dictionary containing node_identifiers as keys and
            leg indices as values. It is used to match the legs of tensor to the
            different nodes. Only the lower (smaller) half of the legs is to be
            put in this dict, the others are inferred by ordering.
        mode (Decomposition): Determines the decomposition to use for splitting
            the tensor. 
        
        Returns:
            TreeTensorNetworkOperator: The resulting TTNO.
        """
        # Ensure that legs are even
        assert tensor.ndim % 2 == 0
        half_dim = int(tensor.ndim / 2)
        # Ensure that operator matches number of lattice sites in TTN
        assert half_dim == len(reference_tree.nodes)
        # Ensure that tensor input and output dimensions are the same
        # assert tensor.shape[:half_dim] == tensor.shape[half_dim:]
        root_id = reference_tree.root_id
        new_leg_dict = {node_id: (leg_dict[node_id], half_dim + leg_dict[node_id])
                        for node_id in leg_dict}
        new_TTNO = cls()
        new_TTNO.add_root(Node(identifier=root_id), tensor)
        linearised = reference_tree.linearise()
        for new_node_id in linearised[:-1]: # Exclude the root node
            cur_root_new = new_TTNO.root_node()
            ref_new_node = reference_tree.nodes[new_node_id]
            ref_children = ref_new_node.children
            ref_children_set = set(ref_children) # To be more efficient in the next line
            new_root_children = [c for c in cur_root_new.children
                                 if c not in ref_children_set]
            new_node_open_legs = new_leg_dict.pop(new_node_id)
            new_root_open_legs = [legs[0] for legs in new_leg_dict.values()]
            new_root_open_legs += [legs[1] for legs in new_leg_dict.values()]
            q_legs = LegSpecification(None, new_root_children,
                                      new_root_open_legs,
                                      is_root=True)
            r_legs = LegSpecification(None, ref_children, new_node_open_legs)
            if mode == Decomposition.QR:
                new_TTNO.split_node_qr(root_id, q_legs, r_legs,
                                       q_identifier=root_id,
                                       r_identifier=new_node_id)
            elif mode == Decomposition.SVD or mode == Decomposition.tSVD:
                svd_params = SVDParameters(max_bond_dim=float('inf'), rel_tol=1e-10)
                new_TTNO.split_node_svd(root_id, q_legs, r_legs,
                                        q_identifier=root_id,
                                        r_identifier=new_node_id,
                                        svd_params=svd_params)
            else:
                raise ValueError(f"Invalid decomposition mode: {mode}")
            num_new_legs = 1 - len(ref_children)
            _update_leg_dict(new_leg_dict, tuple(new_node_open_legs), num_new_legs)

        return new_TTNO

TTNO = TreeTensorNetworkOperator


#####
# Utility for the from_tensor method
#####
class _NumSmaller(Enum):
    """
    An enumeration to determine the number of legs with a smaller index.

    Attributes:
        ZERO: No smaller legs
        ONE: One smaller leg
        TWO: Two smaller legs
    """
    ZERO = 0
    ONE = 1
    TWO = 2

    def new_leg_value(self,
                      old_value: int,
                      num_new_legs: int) -> int:
        """
        Returns the new leg value based on the number of smaller valued legs.

        Args:
            old_value (int): The old leg value.
            num_new_legs (int): The number of new legs added to the node.

        Returns:
            int: The new leg value.
        """
        # Two legs are removed from the current node, the open legs.
        # On the other hand, one virtual leg to the new node is added.
        old_value += num_new_legs
        if self == _NumSmaller.ZERO:
            return old_value
        if self == _NumSmaller.ONE:
            return old_value - 1
        if self == _NumSmaller.TWO:
            return old_value - 2
        errstr = f"Invalid value for {self.__class__.__name__}: {self}"
        raise ValueError(errstr)

def _determine_num_smaller(removed_legs: tuple[int,int],
                           leg_to_check: int
                            ) -> _NumSmaller:
    """
    Determines the number of smaller legs in a node.

    Args:
        removed_legs (tuple[int, int]): The indices of the two legs that are removed.
        leg_to_check (int): The index of the leg to check.

    Returns:
        _NumSmaller: The number of smaller legs.
    """
    if leg_to_check < removed_legs[0] and leg_to_check < removed_legs[1]:
        return _NumSmaller.ZERO
    elif leg_to_check < removed_legs[0] or leg_to_check < removed_legs[1]:
        return _NumSmaller.ONE
    else:
        return _NumSmaller.TWO

def _new_tuple_value(removed_legs: tuple[int,int],
                     old_leg_tuple: tuple[int,int],
                     num_new_legs: int
                     ) -> tuple[int,int]:
    """
    Returns a new tuple of leg values based on the removed legs.

    Args:
        removed_legs (tuple[int, int]): The indices of the two legs that are removed.
        old_leg_tuple (tuple[int, int]): The old tuple of leg values.
        num_new_legs (int): The number of new legs added to the node.

    Returns:
        tuple[int, int]: The new tuple of leg values.
    """
    out = []
    for leg in old_leg_tuple:
        num_smaller = _determine_num_smaller(removed_legs, leg)
        new_val = num_smaller.new_leg_value(leg, num_new_legs)
        out.append(new_val)
    return tuple(out)

def _update_leg_dict(leg_dict: dict[str, list[int]],
                     removed_legs: tuple[int,int],
                     num_new_legs: int
                     ) -> None:
    """
    Updates the leg dictionary based on the removed legs.

    The update is done in place.

    Args:
        leg_dict (dict[str, list[int]]): The leg dictionary to update.
        removed_legs (tuple[int, int]): The indices of the two legs that are removed.
        num_new_legs (int): The number of new legs added to the node.
    """
    for node_id, leg_pair in leg_dict.items():
        new_leg_pair = _new_tuple_value(removed_legs, tuple(leg_pair), num_new_legs)
        leg_dict[node_id] = list(new_leg_pair)
