"""
Provides the fundamental TreeTensorNetwork class.

The TreeTensorNetwork class is a subclass of the TreeStructure class and holds
the structure of a tree tensor network (TTN) in the form of Nodes, as well as
the tensors associated with these nodes. The TreeTensorNetwork class provides
methods to manipulate the TTN, such as adding nodes, contracting nodes, or
splitting nodes. It also provides methods to move the orthogonality center of
the TTN.

Example:

.. code-block:: python

    from pytreenet import TreeTensorNetwork, Node, crandn
    import numpy as np

    # Create a TreeTensorNetwork
    ttn = TreeTensorNetwork()

    # Create a root node
    root_tensor = crandn((2, 3, 4))
    root_node = Node(identifier="root)
    ttn.add_root(root_node, root_tensor)

    # Create a child node
    child_tensor = crandn((2, 2, 3))
    child_node = Node(identifier="child1")
    ttn.add_child_to_parent(child_node, child_tensor, 0, ttn.root_id, 0)

    # Create a second child node
    child_tensor2 = crandn((2, 3, 5))
    child_node2 = Node(identifier="child2")
    ttn.add_child_to_parent(child_node2, child_tensor2, 1, ttn.root_id, 1)

    # Create a grandchild node
    grandchild_tensor = crandn((2, 4))
    grandchild_node = Node(identifier="grandchild")
    ttn.add_child_to_parent(grandchild_node, grandchild_tensor, 0, "child2", 1)

    # Contract the grandchild node with the child node
    ttn.contract_nodes("child2", "grandchild")
    
For details and further usage refer to the example notebooks.
"""
from __future__ import annotations
from typing import Tuple, Callable, Union, List, Dict, Self
from copy import copy, deepcopy
from collections import UserDict
from uuid import uuid1

import numpy as np
from numpy import eye

from .tree_structure import TreeStructure
from .node import Node, relative_leg_permutation
from ..util.tensor_splitting import (tensor_qr_decomposition,
                                     contr_truncated_svd_splitting,
                                     idiots_splitting,
                                     SplitMode,
                                     SVDParameters)
from .leg_specification import LegSpecification
from .canonical_form import (canonical_form,
                             split_qr_contract_r_to_neighbour)
from ..contractions.tree_contraction import completely_contract_tree
from ..contractions.node_contraction import contract_nodes
from ..util.ttn_exceptions import NotCompatibleException


class TensorDict(UserDict):
    """
    A custom dictionary class to store the tensors of a TreeTensorNetwork.
    
    The class is connected to the nodes of the TreeTensorNetwork and ensures
    that the tensors are transposed only when they are accessed. Therefore
    tensor legs should be called using the nodes methods.
    """
    def __init__(self,
                 nodes: Dict[str,Node],
                 inpt: Union[Dict[str,np.ndarray],None] = None) -> None:
        """
        Initiates a new TensorDict.

        Args:
            nodes (Dict[str,Node]): The node dictionary to be associated with
                the TensorDict.
            inpt (Union[Dict[str,np.ndarray],None], optional): A dictionary of
                tensors to be added to the TensorDict. Defaults to None.
        """
        if inpt is None:
            inpt = {}
        super().__init__(inpt)
        self.nodes = nodes

    def __getitem__(self, node_id: str) -> np.ndarray:
        """
        Since during addition of nodes the tensors are not actually transposed,
        this has to be done when accesing them. 
        This way whenever a tensor is accessed, its leg ordering is
            (parent_leg, children_legs, open_legs)

        Args:
            node_id (str): The identifier of the tensor to be accessed.
        
        Returns:
            np.ndarray: The tensor associated with the node transposed to the
                correct leg ordering. (parent_leg, children_legs, open_legs)
        """
        tensor = super().__getitem__(node_id)
        node = self.nodes[node_id]
        transposed_tensor = node.transpose_tensor(tensor)
        super().__setitem__(node_id, transposed_tensor)
        return transposed_tensor

class TreeTensorNetwork(TreeStructure):
    """
    A Tree Tensor Network (TTN)

    A tree tensor network is a tensor network layed out as a tree.
    The data associated with each node is a tensor, stored using the same
    identifier as the node. A tensor associated to a node can have more legs
    than the node has neighbours. These legs are known as open legs.
    The TreeTensorNetwork class provides methods to manipulate the TTN, such as
    adding nodes, contracting nodes, or splitting nodes. It also provides
    methods to control the orthogonality center of the TTN.

    Attributes:
        orthogonality_center_id (Union[str,None]): The identifier of the node
            which is the orthogonality center of the TTN.
        tensors (TensorDict): A dictionary mapping the tensor tree node
            identifiers to the corresponding tensor data.
        root_id (Union[str,None]): The identifier of the root node of the TTN.
        nodes (Dict[str,Node]): A dictionary mapping the node identifiers to
            the corresponding node.
    """

    def __init__(self) -> None:
        """
        Initiates a new TreeTensorNetwork which is initially empty.
        """
        super().__init__()
        self._nodes: Dict[str,Node]
        self._tensors = TensorDict(self._nodes)
        self.orthogonality_center_id: Union[str,None] = None

    @property
    def tensors(self) -> TensorDict[str, np.ndarray]:
        """
        Returns the tensors of the TTN.

        Since during addition of nodes the tensors are not actually transposed,
        this has to be done here. This way whenever tensors are accessed, their
        leg ordering is
        
        ``(parent_leg, children_legs, open_legs)``
        
        The tensors are collected in a diciotnary, where the keys are the node
        identifiers.
        """
        return self._tensors

    @property
    def root(self) -> Tuple[Node, np.ndarray]:
        """
        Returns the root node and the associated tensor

        Returns:
            Tuple[Node, np.ndarray]: The root node and the associated tensor.

        Raises:
            KeyError: If there is no root in the TTN.
        """
        if self.root_id is None:
            errstr = "There is no root!"
            raise KeyError(errstr)
        return self[self.root_id]

    def __getitem__(self, key: str) -> Tuple[Node, np.ndarray]:
        """
        Returns the node and the tensor associated to a given identifier.

        Args:
            key (str): The identifier of the node to be accessed.

        Returns:
            Tuple[Node, np.ndarray]: The node and the tensor associated with the
                identifier. The Node is the first element of the tuple and the
                tensor is the second element.
        """
        node = super().__getitem__(key)
        tensor = self._tensors[key]
        return (node, tensor)

    def __eq__(self, other: TreeTensorNetwork) -> bool:
        """
        Provides an equality check for two TreeTensorNetworks.

        Two TreeTensorNetworks are considered equal if they have the same
        orthogonality center and the same nodes with the same connectivity and
        tensors.

        Args:
            other (TreeTensorNetwork): The TreeTensorNetwork to compare to.
        
        Returns:
            bool: If the two TreeTensorNetworks are equal.
        """
        if not self.orthogonality_center_id == other.orthogonality_center_id:
            return False
        if not len(self.nodes) == len(other.nodes):
            # Avoid the case that one is the subtree of the other.
            return False
        for node_id in self.nodes:
            if node_id in other.nodes: # Avoid KeyError
                if not self.nodes_equal(node_id, other):
                    return False
            else:
                return False # Some node_id is not the same
        return True

    def __contains__(self, identifier: str | Node) -> bool:
        """
        Checks if a node is in the TTN.

        Args:
            identifier (str | Node): The identifier of the node or the node
                itself to be checked.
        
        Returns:
            bool: If the node is in the TTN.
        
        """
        if isinstance(identifier, Node):
            identifier = identifier.identifier
        return super().__contains__(identifier) and identifier in self.tensors

    def num_nodes(self) -> int:
        """
        Returns the number of nodes in the TTN.

        Returns:
            int: The number of nodes in the TTN.
        """
        errstr = "The number of nodes and the number of tensors are not the same!"
        assert len(self.nodes) == len(self.tensors), errstr
        return len(self.nodes)

    def __len__(self) -> int:
        """
        Returns the length of the TTN.

        Returns:
            int: The number of nodes in the TTN.
        """
        return self.num_nodes()

    def deepcopy_parts(self, copy_ids: List[str]) -> Self:
        """
        Creates a new TTN with parts of the data deepcopied.

        This is useful, if one knows one will only work on some part of the TTN
        and won't change the rest of the TTN. However, if the non-listed nodes
        of the TTN are changed, the original TTN is also changed.

        Args:
            List[str]: A list of node identifiers to be deepcopied.

        Returns:
            Self: A new TTN with the specified nodes deepcopied.
        
        """
        cls = self.__class__
        new_ttn = cls.__new__(cls)
        update_dict = {key: value for key, value in self.__dict__.items()
                       if key not in {"_nodes", "_tensors"}}
        new_ttn.__dict__.update(update_dict)
        new_ttn._tensors = TensorDict(self._nodes,
                                      {node_id: self._copy_decider(node_id, tensor, copy_ids)
                                       for node_id, tensor in self.tensors.items()})
        new_ttn._nodes = {node_id: self._copy_decider(node_id, node, copy_ids)
                          for node_id, node in self._nodes.items()}
        new_ttn._tensors.nodes = new_ttn._nodes
        return new_ttn

    def _copy_decider(self,
                      node_id: str, object: Union[Node,np.ndarray],
                      copy_ids: List[str]
                      ) -> Union[Node,np.ndarray]:
        """
        Decides whether to deep copy a node or tensor and does so if necessary.
        
        Args:
            node_id (str): The identifier of the node.
            object (Union[Node,np.ndarray]): The object to be copied.
            copy_ids (List[str]): A list of node identifiers to be deepcopied.

        Returns:
            Union[Node,np.ndarray]: The copied object, if it was copied,
                otherwise the original object.

        """
        if node_id in copy_ids:
            return deepcopy(object)
        return object

    def nodes_equal(self, node_id: str, other: TreeTensorNetwork,
                    other_node_id : Union[None,str] = None) -> bool:
        """
        Compares a node in this tree with a node in a different TTN.

        Args:
            node_id (str): Identifier of a node in this TTN to use for
                comparison.
            other (TreeTensorNetwork): A different TTN to compare to.
            other_node_id (Union[None, str]), Optional: A node identifier
                for the node in the other tree. If it is `None`, the same
                identifier is used for both TTN. Defaults to None.

        Returns:
            bool: If the two nodes are equal and the associated tensors close
             to each other.
        """
        if other_node_id is None:
            test_node, test_tensor = other[node_id]
        else:
            other_node, test_tensor = other[other_node_id]
            test_node = other_node.copy_with_new_id(node_id)
        node, tensor = self[node_id]
        nodes_equal = test_node == node
        if not nodes_equal:
            # Avoids a potential numpy exception due to shape mismatch
            return False
        tensors_equal = np.allclose(test_tensor, tensor)
        return tensors_equal



    def ensure_shape_matching(self, new_tensor: np.ndarray, tensor_leg: int,
                              old_node: Node, old_leg: int,
                              new_node_id: Union[str,None] = None,
                              adapt_shape: bool = False) -> Union[np.ndarray, None]:
        """
        Ensures that the dimensions of the legs of two tensors are compatible.
        
        This function has two modes of operation based on the adapt_shape parameter:
        - If adapt_shape=False (default): Only checks compatibility and raises an 
          exception if dimensions don't match. Returns None.
        - If adapt_shape=True: Returns a modified tensor with compatible dimensions,
          padding with zeros or truncating as needed.

        Args:
            new_tensor (np.ndarray): The tensor with the new leg.
            tensor_leg (int): The leg of the new tensor to be compared.
            old_node (Node): The node with the old leg.
            old_leg (int): The leg of the old node to be compared.
            new_node_id (Union[str,None], optional): The identifier of the new
                node for error messages. Defaults to None.
            adapt_shape (bool, optional): If False, only check compatibility and 
                raise exception on mismatch. If True, return a modified tensor
                with compatible dimensions. Defaults to False.

        Returns:
            Union[np.ndarray, None]: 
                - None if adapt_shape=False (only performs compatibility check)
                - Modified np.ndarray if adapt_shape=True (tensor with compatible dimensions)
                
        Raises:
            NotCompatibleException: If adapt_shape=False and dimensions don't match.
        """
        if new_node_id is None:
           new_node_id = "the new node"

        if not adapt_shape:
            new_dimension = new_tensor.shape[tensor_leg]
            old_dimension = old_node.shape[old_leg]
            if new_dimension != old_dimension:
                errstr = f"Dimensionality of leg {tensor_leg} of {new_node_id}"
                errstr += " is not compatible with"
                errstr += f" leg {old_leg} of {old_node.identifier}"
                raise NotCompatibleException(errstr)
            return None  # Compatibility check passed, return None
        else:
            # Adapt shape mode: return modified tensor
            new_dimension = new_tensor.shape[tensor_leg]
            old_dimension = old_node.shape[old_leg]
            
            # If dimensions match, no action needed
            if new_dimension == old_dimension:
                return new_tensor
            
            # Create a new shape with the matching dimension
            new_shape = list(new_tensor.shape)
            new_shape[tensor_leg] = old_dimension
            
            # Create a new tensor with the modified shape
            modified_tensor = np.zeros(tuple(new_shape), dtype=new_tensor.dtype)
            
            # Copy data from the original tensor, up to the minimum dimension
            min_dim = min(new_dimension, old_dimension)
            
            # Prepare slices for copying data
            source_slices = []
            target_slices = []
            
            for i in new_tensor.shape:
                if i == tensor_leg:
                    source_slices.append(slice(0, min_dim))
                    target_slices.append(slice(0, min_dim))
                else:
                    source_slices.append(slice(None))
                    target_slices.append(slice(None))
            
            # Copy the data
            modified_tensor[tuple(target_slices)] = new_tensor[tuple(source_slices)]
            return modified_tensor

    def add_root(self, node: Node, tensor: np.ndarray):
        """
        Adds a root tensor node to the TTN.

        Args:
            node (Node): The root node to be added.
            tensor (np.ndarray): The tensor associated with the root node.
        """
        node.link_tensor(tensor)
        super().add_root(node)
        self.tensors[node.identifier] = tensor

    def add_child_to_parent(self, child: Node, tensor: np.ndarray,
                            child_leg: int, parent_id: str, parent_leg: int,
                            compatible: bool = True):
        """
        Adds a child node to a parent node in the TTN.

        Note: Legs of the nodes might change during this operation to fit the
        convention (parent_leg,children_legs,open_legs)

        Args:
            child (Node): The child node to be added.
            tensor (np.ndarray): The tensor associated with the child node.
            child_leg (int): The leg of the child tensor to be connected to the
                parent tensor.
            parent_id (str): The identifier of the parent node.
            parent_leg (int): The leg of the parent tensor to be connected to the
                child tensor.
            compatible (bool, optional): Controls shape compatibility behavior.
                - If True (default): Only checks dimensions match and raises exception on mismatch.
                - If False: Automatically adapts the child tensor dimensions to match the parent,
                  padding with zeros or truncating as needed. Defaults to True.
                  
        Raises:
            NotCompatibleException: If compatible=True and the dimensions of the child
                and parent legs don't match.            
        """
        self.ensure_existence(parent_id)
        parent_node = self._nodes[parent_id]
        child_id = child.identifier
        
        if not compatible:
            # Get the modified tensor with compatible dimensions
            modified_tensor = self.ensure_shape_matching(tensor, child_leg,
                                                        parent_node, parent_leg,
                                                        child_id, True)
            
            # Use the modified tensor instead of the original
            child.link_tensor(modified_tensor)
            self._add_node(child)
            child.open_leg_to_parent(parent_id, child_leg)
            parent_node.open_leg_to_child(child_id, parent_leg)
            self.tensors[child_id] = modified_tensor
        else:
            self.ensure_shape_matching(tensor, child_leg,
                                    parent_node, parent_leg,
                                    child_id)
            child.link_tensor(tensor)
            self._add_node(child)
            child.open_leg_to_parent(parent_id, child_leg)
            parent_node.open_leg_to_child(child_id, parent_leg)
            self.tensors[child_id] = tensor
       
    def add_parent_to_root(self, root_leg: int, parent: Node,
                           tensor: np.ndarray, parent_leg: int):
        """
        Adds a parent node to the root node of the TTN.

        Note: Legs of the nodes might change during this operation to fit the
        convention (parent_leg,children_legs,open_legs).

        Args:
            root_leg (int): The leg of the root tensor to be connected to the
                parent tensor.
            parent (Node): The parent node to be added.
            tensor (np.ndarray): The tensor associated with the parent node.
            parent_leg (int): The leg of the parent tensor to be connected to the
                root tensor.
                
        Raises:
            NotCompatibleException: If the dimensions of the parent and root 
                legs don't match.
        """
        self.ensure_existence(self.root_id)
        former_root_node = self.root[0]
        new_root_id = parent.identifier
        self.ensure_shape_matching(tensor, parent_leg,
                                   former_root_node, root_leg,
                                   new_root_id)
        self._add_node(parent)
        parent.open_leg_to_child(self.root_id, parent_leg)
        former_root_node.open_leg_to_parent(new_root_id, root_leg)
        self._root_id = new_root_id
        self.tensors[new_root_id] = tensor

    def conjugate(self) -> TreeTensorNetwork:
        """
        Returns a conjugated version of this TTN.

        This means for every tensor in the TTN, the complex conjugate is taken.
        """
        ttn_conj = deepcopy(self)
        for node_id, tensor in ttn_conj.tensors.items():
            ttn_conj.tensors[node_id] = tensor.conj()
        return ttn_conj

    def bond_dim(self, node_id: str,
                 neighbour_id: Union[str,None] = None) -> int:
        """
        Find the bond dimension between a node and one of its neighbours.

        Args:
            node_id (str): The identifier of the node.
            neighbour_id (Union[str,None], optional): The identifier of the
                neighbour node. If None, the parent is used. Defaults to None.
        """
        node = self.nodes[node_id]
        if neighbour_id is None:
            neighbour_id = node.parent
        return node.shape[node.neighbour_index(neighbour_id)]

    def max_bond_dim(self) -> int:
        """
        Find the maximum virtual bond dimension in this TTN.

        Returns:
            int: The maximum bond dimension of this TTN.
        """
        if self.num_nodes()<=1:
            errstr = "This TTN has no virtual bond dimension!"
            raise AssertionError(errstr)
        max_bd = 0
        for node in self.nodes.values():
            if not node.is_root():
                parent_bd = self.bond_dim(node.identifier,node.parent)
                if parent_bd > max_bd:
                    max_bd = parent_bd
        return max_bd

    def bond_dims(self) -> Dict[Tuple[str, str], int]:
        """
        Returns the bond dimensions between all neighbouring nodes.

        Returns:
            Dict[Tuple[str, str], int]: The bond dimensions between
                neighbouring nodes. The keys are the node identifiers of the
                parent and child node in that order.
        """
        bond_dims = {}
        for node in self.nodes.values():
            if not node.is_root():
                parent_bd = self.bond_dim(node.identifier,node.parent)
                bond_dims[(node.parent, node.identifier)] = parent_bd
        return bond_dims

    def avg_bond_dim(self) -> float:
        """
        Returns the average bond dimension of the TTN.

        Returns:
            float: The average bond dimension of the TTN.
        
        """
        bond_dims = self.bond_dims()
        return sum(bond_dims.values())/len(bond_dims)

    def size(self) -> int:
        """
        Returns the size of the TTN.

        The size is the total number of elements in all tensors.

        Returns:
            int: The size of the TTN.
        """
        return sum(tensor.size for tensor in self.tensors.values())

    def pad_bond_dimension(self,
                           node1_id: str,
                           node2_id: str,
                           new_bond_dim: int):
        """
        Pads the bond dimension between two nodes to a given value.

        Args:
            node1_id (str): The identifier of the first node.
            node2_id (str): The identifier of the second node.
            new_bond_dim (int): The new bond dimension to be set between the
                two nodes.
        
        """
        old_dim = self.bond_dim(node1_id, node2_id)
        parent_id, child_id = self.determine_parentage(node1_id, node2_id)
        id_id = str(uuid1())
        self.insert_identity(child_id, parent_id,
                             new_identifier=id_id)
        # Build the projector to the larger bond_dimension
        projector = np.zeros((old_dim, new_bond_dim))
        projector[:old_dim, :old_dim] = np.eye(old_dim)
        # Replace the identity node with two projectors
        proj_id_pa = str(uuid1())
        proj_id_ch = str(uuid1())
        id_node = self.nodes[id_id]
        proj_pa_legs = LegSpecification(id_node.parent,
                                        [],
                                        [])
        proj_ch_legs = LegSpecification(None,
                                        id_node.children,
                                        [])
        self.split_node_replace(id_id,
                                projector,
                                projector.T,
                                identifier_a=proj_id_pa,
                                identifier_b=proj_id_ch,
                                legs_a=proj_pa_legs,
                                legs_b=proj_ch_legs
                                )
        # Contract the projectors with the original nodes
        self.contract_nodes(parent_id, proj_id_pa, new_identifier=parent_id)
        self.contract_nodes(child_id, proj_id_ch, new_identifier=child_id)

    def pad_bond_dimensions(self,
                            new_bond_dim: int):
        """
        Pads all bond dimensions in the TTN to a given value.

        Only smaller bond dimensions are padded, larger ones are left
        unchanged.

        Args:
            new_bond_dim (int): The new bond dimension to be set between all
                neighbouring nodes.
        """
        for node_id, node in self.nodes.items():
            if not node.is_root():
                parent_id = node.parent
                if self.bond_dim(node_id, parent_id) < new_bond_dim:
                    self.pad_bond_dimension(node_id, parent_id, new_bond_dim)

    @staticmethod
    def _absorption_warning() -> str:
        errstr = "Only square Matrices can be absorbed!\n"
        errstr += "If you desire to contract a non-square matrix/a higher-degree tensor\n"
        errstr += "then add it as a new child to this TTN and contract it."
        return errstr

    def absorb_matrix(self, node_id: str, absorbed_matrix: np.ndarray,
                      this_tensors_leg_index: int,
                      absorbed_matrix_leg_index: int = 1):
        """
        Absorbs a matrix into one of this TTN's tensors at a given tensor
        leg.

        Args:
            node_id (str): Identifier of the node/tensor into which the matrix
                should be absorbed.
            absorbed_matrix (np.ndarray): Matrix to be absorbed. Has to be a
                square matrix, as otherwise the tensor shape is changed. If you
                desire to contract a non-square matrix/ a higher-degree tensor, add
                a new child to this tensor and contract it with this tensor.
            this_tensors_leg_index (int): The leg of this TTN's tensor that is to
                be contracted with the absorbed tensor.
            absorbed_tensors_leg_index (int, Optional): Leg that is to be
                contracted with this instance's tensor. Defaults to 1, as this is
                usually considered to be the input leg of a matrix.

        """
        m_shape = absorbed_matrix.shape
        if len(m_shape) != 2:
            errstr = self._absorption_warning()
            raise AssertionError(errstr)
        node_tensor = self.tensors[node_id]
        new_tensor = np.tensordot(node_tensor, absorbed_matrix,
                                  axes=(this_tensors_leg_index, absorbed_matrix_leg_index))
        this_tensors_indices = tuple(range(new_tensor.ndim))
        transpose_perm = (this_tensors_indices[0:this_tensors_leg_index]
                          + (this_tensors_indices[-1], )
                          + this_tensors_indices[this_tensors_leg_index:-1])
        self.tensors[node_id] = new_tensor.transpose(transpose_perm)

    def absorb_matrix_into_neighbour_leg(self, node_id: str, neighbour_id: str,
                                         tensor: np.ndarray, tensor_leg: int = 1):
        """
        Absorb a matrix into a node.
        
        One of the legs of the matrix will be contracted into a neighbour leg
        of the node.

        Args:
            node_id (str): The identifier of the node into which the tensor is
                absorbed
            neighbour_id (str): The identifier of the neighbour to which the leg
                points, which is to be contracted with the tensor
            tensor (np.ndarray): The tensor to be contracted
            tensor_leg (int, optional): The leg of the external tensor which is
                to be contracted. Defaults to 1, as this is usually the input
                leg of a matrix.

        """
        node = self.nodes[node_id]
        neighbour_leg = node.neighbour_index(neighbour_id)
        self.absorb_matrix(node_id, tensor, neighbour_leg, tensor_leg)

    def absorb_into_open_legs(self, node_id: str,
                              tensor: np.ndarray,
                              unitary: bool = False):
        """
        Absorb a tensor into the open legs of the tensor of a node.

        This tensor will be absorbed into all open legs and it is assumed, the
        leg order of the tensor to be absorbed is the same as the order of
        the open legs of the node.

        Since the tensor to be absorbed is considered to represent an operator
        acting on the node, it will have to have exactly twice as many legs as
        the node has open legs. The input legs, i.e. the ones contracted, are
        assumed to be the second half of the legs of the tensor.

        Args:
            node_id (str): The identifier of the node which is to be contracted
                with the tensor
            tensor (np.ndarray): The tensor to be contracted.
            unitary (bool, optional): If True, the tensor is assumed to be
                Otherwise, the orthogonality centre is removed. Defaults to
                False.
        """
        node, node_tensor = self[node_id]
        nopen_legs = node.nopen_legs()
        assert tensor.ndim == 2 * nopen_legs
        if tensor.shape[:nopen_legs] != tensor.shape[nopen_legs:]:
            errstr = self._absorption_warning()
            raise AssertionError(errstr)
        tensor_legs = [i + nopen_legs for i in range(nopen_legs)]
        new_tensor = np.tensordot(node_tensor, tensor,
                                  axes=(node.open_legs, tensor_legs))
        # The leg ordering was not changed here
        self.tensors[node_id] = new_tensor
        if not unitary and self.orthogonality_center_id != node_id:
            self.orthogonality_center_id = None

    def change_node_identifier(self, new_node_id: str, old_node_id: str):
        """
        Changes the identifier of a node in the TTN.

        This also changes all references to the node in the TTN and the tensor
        dictionary.

        Args:
            new_node_id (str): The new identifier of the node.
            old_node_id (str): The old identifier of the node.

        """
        self.tensors[new_node_id] = self._tensors.pop(old_node_id)
        super().change_node_identifier(new_node_id, old_node_id)

    def replace_tensor(self,
                       node_id: str,
                       new_tensor: np.ndarray,
                       permutation: Union[None, List[int]] = None):
        """
        Replaces the tensor associated with a node.

        Args:
            node_id (str): The identifier of the node.
            new_tensor (np.ndarray): The new tensor to be associated with the
                node.
            permutation (Union[None, List[int]], optional): A permutation to
                be applied to the new tensor to match the leg ordering of the
                node. Defaults to None.

        """
        self._nodes[node_id].replace_tensor(new_tensor,
                                            permutation=permutation)
        self._tensors[node_id] = new_tensor

    def replace_node(self,
                     new_node_id: str,
                     old_node_id: str,
                     tensor: np.ndarray):
        """
        Replaces a node with a new one.

        This mainly allows for a node to have a different number of open legs.

        Args:
            new_node_id (str): The identifier of the new node.
            old_node_id (str): The identifier of the old node.
            tensor (np.ndarray): The tensor associated to the new node.

        """
        new_node = Node(tensor=tensor,
                        identifier=new_node_id)
        old_node = self._nodes[old_node_id]
        for neighbour_id in old_node.neighbouring_nodes():
            neighbour_index = old_node.neighbour_index(neighbour_id)
            new_dim = new_node.shape[neighbour_index]
            old_dim = old_node.shape[neighbour_index]
            assert new_dim == old_dim, \
                f"The dimensions of the legs of the new node {new_node_id} are not compatible with the neighbour {neighbour_id}!"
        self.replace_node_in_neighbours(new_node_id, old_node_id,
                                        del_old_node=True)
        # Otherwise it might happen that the old tensor is still around
        del self._tensors[old_node_id]
        new_node.children = old_node.children
        new_node.parent = old_node.parent
        self._nodes[new_node_id] = new_node
        self._tensors[new_node_id] = tensor

    def insert_identity(self, child_id: str, parent_id: str,
                        new_identifier: Union[str,None] = None):
        """
        Insertes an identity tensor between two nodes.

        Args:
            child_id (str): Identifier of the child node.
            parent_id (str): Identifier of the parent node.
            new_identifier (Union[str,None], optional): An identifier for the
                new tensor. If None, a random unique identifier is used.
                Defaults to None.

        """
        assert self.is_child_of(child_id, parent_id)
        assert self.is_parent_of(parent_id, child_id)
        if new_identifier is None:
            new_identifier = str(uuid1())
        # Change node connecitivity
        child_node = self.nodes[child_id]
        child_node.replace_neighbour(parent_id, new_identifier)
        parent_node = self.nodes[parent_id]
        parent_node.replace_neighbour(child_id, new_identifier)
        # Create Identity node
        dim = child_node.parent_leg_dim()
        identity = eye(dim)
        id_node = Node(identifier=new_identifier,
                       tensor=identity)
        id_node.open_leg_to_parent(parent_id, 0)
        id_node.open_leg_to_child(child_id, 1)
        self._nodes[new_identifier] = id_node
        self._tensors[new_identifier] = identity

    def contract_nodes(self, node_id1: str, node_id2: str, new_identifier: str = ""):
        """
        Contracts two nodes.
        
        This means a new node with the contracted tensor is inserted into the
        ttn with the result of the tensor contraction of the other nodes as the
        associated tensor.
        Note that one of the nodes has to be the parent of the other.
        The resulting leg order is the following:

        ``(parent_parent_leg, node1_children_legs, node2_children_legs,
        node1_open_legs, node2_open_legs)``
    
        The resulting node will have the identifier

        ``parent_id + "contr" + child_id``

        unless an alternative is provided.

        Note that this removes the original nodes and tensors from the TTN.

        Args:
            node_id1 (str): Identifier of first tensor
            node_id2 (str): Identifier of second tensor
            new_identifier (str, optional): A potential new identifier.
            Otherwise defaults to `node_id1 + "contr" + node_id2`.
        
        """
        node1, tensor1 = self[node_id1]
        node2, tensor2 = self[node_id2]
        new_node, new_tensor = contract_nodes(node1, tensor1,
                                              node2, tensor2,
                                              new_identifier=new_identifier)
        new_identifier = new_node.identifier
        # Deal with tensor data
        ## Remove old tensors
        self.tensors.pop(node_id1)
        self.tensors.pop(node_id2)
        ## Add new tensor
        self.tensors[new_identifier] = new_tensor
        # Change connectivity. This deletes the old nodes.
        self.replace_node_in_neighbours(new_identifier, node_id1)
        self.replace_node_in_neighbours(new_identifier, node_id2)
        self._nodes[new_identifier] = new_node

    def contract_all_children(self, node_id: str,
                              new_identifier: Union[str,None] = None):
        """
        Contracts all children of a node.

        This is done by contracting the children with the parent node.

        Args:
            node_id (str): Identifier of the parent node.
            new_identifier (str, optional): An identifier for the new tensor.
                Defaults to the node identifier.
        
        """
        node = self.nodes[node_id]
        if new_identifier is None:
            new_identifier = node_id
        children = node.children
        for child_id in copy(children):
            self.contract_nodes(node_id, child_id,
                                new_identifier=new_identifier)

    def legs_before_combination(self, node1_id: str,
                                node2_id: str) -> Tuple[LegSpecification, LegSpecification]:
        """
        Records which leg corresponds to which node.

        When combining two nodes, the information about their legs is lost.
        However, sometimes one wants to split the two nodes again, as they were
        before. This function provides the required leg specification for the
        splitting.

        Args:
            node1_id (str): Identifier of the first node to be combined
            node2_id (str): Identifier of the second node to be combined

        Returns:
            Tuple[LegSpecification, LegSpecification]: The leg specifications
                containing the information to split the two nodes again, to
                have the same legs as before (assuming the open legs are not
                transposed). Since it is not needed the  LegSpecification of
                the parent node has the identifier of the child node not
                included. Same for the LegSpecification of the child node and
                the parent legs. The open legs are the index values that the
                legs would have after contracting the two nodes.
        """
        node1 = self.nodes[node1_id]
        node2 = self.nodes[node2_id]
        tot_nvirt_legs = node1.nvirt_legs() + node2.nvirt_legs() - 2
        tot_nlegs = node1.nlegs() + node2.nlegs() - 2
        open_legs1 = list(range(tot_nvirt_legs, tot_nvirt_legs + node1.nopen_legs()))
        open_legs2 = list(range(tot_nvirt_legs + node1.nopen_legs(), tot_nlegs))
        spec1 = LegSpecification(parent_leg=None,
                                 child_legs=copy(node1.children),
                                 open_legs=open_legs1,
                                 node=None)
        spec2 = LegSpecification(parent_leg=None,
                                 child_legs=copy(node2.children),
                                 open_legs=open_legs2,
                                 node=None)
        temp = [(spec1, node1), (spec2, node2)]
        if node2.is_parent_of(node1_id):
            temp.reverse()
        temp[0][0].parent_leg = temp[0][1].parent
        temp[0][0].child_legs.remove(temp[1][1].identifier)
        if node1.is_root():
            spec1.is_root = True
        elif node2.is_root():
            spec2.is_root = True
        return (spec1, spec2)

    def split_nodes(self, node_id: str,
                     out_legs: LegSpecification, in_legs: LegSpecification,
                     splitting_function: Callable,
                     out_identifier: str = "", in_identifier: str = "",
                     **kwargs):
        """
        Splits an node into two nodes using a specified function

        Args:
            node_id (str): The identifier of the node to be split.
            out_legs (LegSpecification): The legs associated to the output of the
                matricised node tensor. (The Q legs for QR and U legs for SVD)
            in_legs (LegSpecification): The legs associated to the input of the
                matricised node tensor: (The R legs for QR and the SVh legs for SVD)
            splitting_function (Callable): The function to be used for the splitting
                of the tensor. This function should take the tensor and the
                legs in the form of integers and return two tensors. The first
                tensor should have the legs in the order
                (parent_leg, children_legs, open_legs, new_leg) and the second
                tensor should have the legs in the order
                (new_leg, parent_leg, children_legs, open_legs).
            out_identifier (str, optional): An identifier for the tensor with the
            output legs. Defaults to "".
            in_identifier (str, optional): An identifier for the tensor with the input
                legs. Defaults to "".
            **kwargs: Are passed to the splitting function.
        """
        node, tensor = self[node_id]
        if out_legs.node is None:
            out_legs.node = node
        if in_legs.node is None:
            in_legs.node = node
        # Find new identifiers
        if out_identifier == "":
            out_identifier = "out_of_" + node_id
        if in_identifier == "":
            in_identifier = "in_of_" + node_id

        # Getting the numerical value of the legs
        out_legs_int = out_legs.find_leg_values()
        in_legs_int = in_legs.find_leg_values()
        out_tensor, in_tensor = splitting_function(tensor,
                                                   out_legs_int,
                                                   in_legs_int,
                                                   **kwargs)
        self._tensors[out_identifier] = out_tensor
        self._tensors[in_identifier] = in_tensor

        # New Nodes
        out_node = Node(tensor=out_tensor, identifier=out_identifier)
        in_node = Node(tensor=in_tensor, identifier=in_identifier)
        self._nodes[out_identifier] = out_node
        self._nodes[in_identifier] = in_node

        # Currently the tensors out and in have the leg ordering
        # (new_leg(for in), parent_leg, children_legs, open_legs, new_leg(for out))
        self._set_in_parent_leg_after_split(in_node,
                                            in_legs,
                                            out_identifier)
        self._set_in_children_legs_after_split(in_legs,
                                               out_legs,
                                               in_identifier,
                                               out_identifier)
        self._set_out_parent_leg_after_split(out_node,
                                             out_legs,
                                             in_identifier)
        self._set_out_children_legs_after_split(out_legs,
                                                in_legs,
                                                out_identifier,
                                                in_identifier)
        self.replace_node_in_some_neighbours(out_identifier, node_id,
                                             out_legs.find_all_neighbour_ids())
        self.replace_node_in_some_neighbours(in_identifier, node_id,
                                             in_legs.find_all_neighbour_ids())
        self._set_root_from_leg_specs(in_legs, out_legs,
                                      in_identifier, out_identifier)
        if node_id not in [out_identifier, in_identifier]:
            self._tensors.pop(node_id)
            self._nodes.pop(node_id)

    def _set_in_parent_leg_after_split(self,
                                       in_node: Node,
                                       in_legs: LegSpecification,
                                       out_identifier: str):
        """
        Sets the parent leg of the in node.
        """
        if in_legs.parent_leg is not None:
            # In this case we have for the leg ordering for in
            # (new_leg, parent_leg, children_legs, open_legs)
            in_node.open_leg_to_parent(in_legs.parent_leg,1)
        elif not in_legs.is_root:
            # In this case we have for the leg ordering for in
            # (new_leg=parent_leg, children_legs, open_legs)
            in_node.open_leg_to_parent(out_identifier,0)
        # Otherwise the in_node is the root and doesn't have a parent.

    def _set_out_parent_leg_after_split(self,
                                        out_node: Node,
                                        out_legs: LegSpecification,
                                        in_identifier: str):
        """
        Sets the parent leg of the out node.
        """
        if out_legs.parent_leg is not None:
            # In this case we have for the leg ordering for out
            # (parent_leg, children_legs, open_legs, new_leg=child_leg)
            out_node.open_leg_to_parent(out_legs.parent_leg,0)
        elif not out_legs.is_root:
            # In this case we have for the leg ordering for out
            # (children_legs, open_legs, new_leg=parent_leg)
            parent_leg = out_node.nlegs() - 1
            out_node.open_leg_to_parent(in_identifier,parent_leg)
        # Otherwise the out_node is the root and doesn't have a parent.

    def _set_in_children_legs_after_split(self,
                                         in_legs: LegSpecification,
                                         out_legs: LegSpecification,
                                         in_identifier: str,
                                         out_identifier: str):
        """
        Sets the children legs of the out node after a split
        """
        in_node = self.nodes[in_identifier]
        in_children = self._find_in_children(in_legs,
                                             out_legs,
                                             out_identifier)
        in_node.open_legs_to_children(in_children)

    def _set_out_children_legs_after_split(self,
                                           out_legs: LegSpecification,
                                           in_legs: LegSpecification,
                                           out_identifier: str,
                                           in_identifier: str):
        """
        Sets the children legs of the out node after a split
        """
        out_node = self.nodes[out_identifier]
        out_children = self._find_out_children(out_legs,
                                               in_legs,
                                               out_identifier,
                                               in_identifier)
        out_node.open_legs_to_children(out_children)

    def _set_root_from_leg_specs(self,
                                 in_legs: LegSpecification,
                                 out_legs: LegSpecification,
                                 in_identifier: str,
                                 out_identifier: str):
        """
        Sets a new root, if required after contraction.
        """
        if in_legs.is_root:
            assert not out_legs.is_root
            self._root_id = in_identifier
        elif out_legs.is_root:
            self._root_id = out_identifier

    def _find_in_children(self,
                          in_legs: LegSpecification,
                          out_legs: LegSpecification,
                          out_identifier: str) -> Dict[str, int]:
        """
        Finds the indices that correspond to the children of the in tensor.
        """
        in_setoff = 1
        in_children = {}
        if in_legs.is_root:
            # In this case we have for the leg ordering for in
            # (new_leg=child_leg, children_legs, open_legs)
            assert out_legs.parent_leg is None
            in_children[out_identifier] = 0
        elif in_legs.parent_leg is not None:
            # In this case we have for the leg ordering for in
            # (new_leg=child_leg, parent_leg, children_legs, open_legs)
            in_setoff = 2
            # The parent will be set first, moving all other legs by +one index
            in_children[out_identifier] = 1
        # All other cases have for the in leg ordering
        # (new_leg=parent_leg, children_legs, open_legs)
        in_children.update({child_id: leg_value + in_setoff
                            for leg_value, child_id in enumerate(in_legs.child_legs)})
        return in_children

    def _find_out_children(self,
                           out_legs: LegSpecification,
                           in_legs: LegSpecification,
                           out_identifier: str,
                           in_identifier: str) -> Dict[str, int]:
        """
        Finds the indices that correspond to the children of the out tensor.
        """
        out_node = self.nodes[out_identifier]
        out_children = {}
        if in_legs.is_root or in_legs.parent_leg is not None:
            # In this case we have for the leg ordering for out
            # (children_legs, open_legs, new_leg=parent_leg)
            assert out_legs.parent_leg is None
            # The parent will be set first, moving all other legs by +one index
            out_setoff = 1
        elif out_legs.is_root:
            # In this case we have for the leg ordering for out
            # (children_legs, open_legs, new_leg=child_leg)
            out_setoff = 0
            out_children[in_identifier] = out_node.nlegs() - 1
        else:
            # In this case we have for the leg ordering for out
            # (parent_leg, children_legs, open_legs, new_leg=child_leg)
            assert out_legs.parent_leg is not None
            out_setoff = 1
            out_children[in_identifier] = out_node.nlegs() - 1
        out_children.update({child_id: leg_value + out_setoff
                            for leg_value, child_id in enumerate(out_legs.child_legs)})
        return out_children

    def split_node_qr(self, node_id: str,
                      q_legs: LegSpecification, r_legs: LegSpecification,
                      q_identifier: str = "", r_identifier: str = "",
                      mode: SplitMode = SplitMode.REDUCED):
        """
        Splits a node into two nodes via QR-decomposition.

        Args:
            node_id (str): Identifier of the node to be split
            q_legs (LegSpecification): The legs which should be part of the
                Q-tensor
            r_legs (LegSpecification): The legs which should be part of the
                R-tensor
            q_identifier (str, optional): An identifier for the Q-tensor.
                Defaults to "".
            r_identifier (str, optional): An identifier for the R-tensor.
                Defaults to "".
            mode: The mode to be used for the QR decomposition. For details
                refer to `tensor_util.tensor_qr_decomposition`.
        """
        self.split_nodes(node_id, q_legs, r_legs,
                         tensor_qr_decomposition,
                         out_identifier=q_identifier,
                         in_identifier=r_identifier,
                         mode=mode)

    def split_node_svd(self, node_id: str,
                       u_legs: LegSpecification, v_legs: LegSpecification,
                       u_identifier: str = "", v_identifier: str = "",
                       svd_params: SVDParameters = SVDParameters()):
        """
        Splits a node in two using singular value decomposition.
        
        In the process the tensors are truncated as specified by truncation
        parameters. The singular values are absorbed into the v_legs.

        Args:
            node_id (str): Identifier of the nodes to be split
            u_legs (LegSpecification): The legs which should be part of the U tensor
            v_legs (LegSpecification): The legs which should be part of the V tensor
            u_identifier (str, optional): An identifier for the U-tensor.
                Defaults to ""
            v_identifier (str, optional): An identifier for the V-tensor.
                Defaults to "".
        """
        self.split_nodes(node_id, u_legs, v_legs, contr_truncated_svd_splitting,
                         out_identifier=u_identifier, in_identifier=v_identifier,
                         svd_params=svd_params)

    def split_node_replace(self, node_id: str,
                           tensor_a: np.ndarray, tensor_b: np.ndarray,
                           identifier_a: str, identifier_b: str,
                           legs_a: LegSpecification, legs_b: LegSpecification):
        """
        Replaces a node with two new nodes of compatible shape.

        Args:
            node_id (str): Identifier of the node to be replaced
            tensor_a (np.ndarray): The tensor to be associated with the first
                new node. Has to have the leg order
                (parent_leg, children_legs, open_legs, new_leg)
            tensor_b (np.ndarray): The tensor to be associated with the second
                new node. Has to have the leg order
                (new_leg, parent_leg, children_legs, open_legs)
            identifier_a (str): Identifier for the first new node.
            identifier_b (str): Identifier for the second new node.
            legs_a (LegSpecification): The legs which should be part of the
                first new node.
            legs_b (LegSpecification): The legs which should be part of the
                second new node.
        """
        self.split_nodes(node_id, legs_a, legs_b, idiots_splitting,
                         identifier_a, identifier_b,
                         a_tensor=tensor_a, b_tensor=tensor_b)


    def move_orthogonalization_center(self, new_center_id: str,
                                      split_function = split_qr_contract_r_to_neighbour,
                                      **kwargs):
        """
        Moves the orthogonalization center to a different node.

        For this to work the TTN has to be in a canonical form already, i.e.,
        there should already be an orthogonalisation center.

        Args:
            new_center_id (str): The identifier of the new orthogonalisation
                center.
            split_function: The function to use for splitting nodes (QR or SVD).
                Defaults to split_qr_contract_r_to_neighbour.
            **kwargs: Additional keyword arguments to pass to the splitting function.
                Common parameters include:
                - mode: The mode to use for QR decomposition (when using split_qr_contract_r_to_neighbour).
                - svd_params: SVD parameters for SVD-based decomposition.
        """
        if self.orthogonality_center_id is None:
            errstr = "The TTN is not in canonical form, so the orth. center cannot be moved!"
            raise AssertionError(errstr)
        self.orth_from_to(self.orthogonality_center_id, new_center_id, split_function, **kwargs)
            
    def orth_from_to(self,
                     start_id: str,
                     end_id: str,
                     split_function = split_qr_contract_r_to_neighbour,
                     **kwargs):
        """
        Perform a chain of decompositions from one node to another.

        Args:
            start_id (str): The identifier of the starting node.
            end_id (str): The identifier of the ending node.
            split_function: The function to use for splitting nodes (QR or SVD).
                Defaults to split_qr_contract_r_to_neighbour.
            **kwargs: Additional keyword arguments to pass to the splitting function.
                Common parameters include:
                - mode: The mode to use for QR decomposition (when using split_qr_contract_r_to_neighbour).
                - svd_params: SVD parameters for SVD-based decomposition.
        """        
        if start_id == end_id:
            # We are done already.
            return
        path = self.path_from_to(start_id, end_id)
        pairs = zip(path[:-1], path[1:])
        for current_id, next_id in pairs:
            split_function(self, current_id, next_id, **kwargs)
        if self.orthogonality_center_id in path:
            self.orthogonality_center_id = end_id
        elif self.orthogonality_center_id is not None:
            self.orthogonality_center_id = None


    def assert_orth_center(self, node_id: str,
                           object_name: str = "node"):
        """
        Asserts that a given node is the orthogonality center.

        Args:
            node_id (str): The identifier of the node to be checked.
            object_name (str, optional): The name of the object to be checked.
                Defaults to "node".
        
        Raises:
            AssertionError: If the node is not the orthogonality center.

        """
        if self.orthogonality_center_id != node_id:
            errstr = f"The {object_name} {node_id} is not the orthogonality center!"
            raise AssertionError(errstr)

    def ensure_orth_center(self, node_id: str,
                           mode: SplitMode = SplitMode.REDUCED
                           ) -> bool:
        """
        Ensures that a given node is the orthogonality center.

        If the node is not the orthogonality center, the center is moved to
        the node, if there is no orthogonality center yet, the TTN is brought
        into canonical form.

        Args:
            node_id (str): The identifier of the node to be checked.
            mode: The mode to be used for the QR decomposition. For details refer to
                `tensor_util.tensor_qr_decomposition`.
        
        Returns:
            bool: Wether the node was the orthogonality center or not.

        """
        self.ensure_existence(node_id)
        if self.orthogonality_center_id is None:
            self.canonical_form(node_id, mode = mode)
            return False
        if self.orthogonality_center_id != node_id:
            self.move_orthogonalization_center(node_id, mode=mode)
            return False
        return True

    def ensure_root_orth_center(self,
                               mode: SplitMode = SplitMode.REDUCED
                               ) -> bool:
        """
        Ensures that the root node is the orthogonality center.

        If the root node is not the orthogonality center, the center is moved
        to the root node. If there is no orthogonality center yet, the TTN is
        brought into canonical form.

        Args:
            mode: The mode to be used for the QR decomposition. For details refer to
                `tensor_util.tensor_qr_decomposition`.

        Returns:
            bool: Wether the root node was the orthogonality center or not.

        """
        assert self.root_id is not None, "The TTN has no root node!"
        return self.ensure_orth_center(self.root_id, mode=mode)

    # Functions below this are just wrappers of external functions that are
    # linked tightly to the TTN and its structure. This allows these functions
    # to be overwritten for subclasses of the TTN with more known structure.
    # The additional structure allows for more efficent algorithms than the    # general case.
    
    def canonical_form(self, orthogonality_center_id: str, 
                       split_function: callable = split_qr_contract_r_to_neighbour,
                       **kwargs):
        """
        Brings the TTN in canonical form with respect to a given orthogonality
         center.

        Only moves the orthogonality center if there is one already. To reset
        the canonical form for sure, use the function
        `canonical_form.canonical_form`.

        Args:
            orthogonality_center_id (str): The new orthogonality center of the
                TTN.
            split_function: The function to use for splitting nodes (QR or SVD).
                Defaults to split_qr_contract_r_to_neighbour.
            **kwargs: Additional keyword arguments to pass to the splitting function.
                Common parameters include:
                - mode: The mode to use for QR decomposition (when using split_qr_contract_r_to_neighbour).
                - svd_params: SVD parameters for SVD-based decomposition.
        """
        if self.orthogonality_center_id is None:
            canonical_form(self, orthogonality_center_id, split_function, **kwargs)
        else:
            self.move_orthogonalization_center(orthogonality_center_id, split_function, **kwargs)

    def orthogonalize(self, orthogonality_center_id: str,
                      split_function: callable = split_qr_contract_r_to_neighbour,
                      **kwargs):
        """
        Wrapper of canonical form.

        Args:
            orthogonality_center_id (str): The new orthogonality center of the
                TTN.
            split_function: The function to use for splitting nodes (QR or SVD).
                Defaults to split_qr_contract_r_to_neighbour.
            **kwargs: Additional keyword arguments to pass to the splitting function.
                Common parameters include:
                - mode: The mode to use for QR decomposition (when using split_qr_contract_r_to_neighbour).
                - svd_params: SVD parameters for SVD-based decomposition.
        """
        self.canonical_form(orthogonality_center_id, split_function, **kwargs)

    def completely_contract_tree(self,
                                 to_copy: bool=False) -> Tuple[np.ndarray, List[str]]:
        """
        Completely contracts the given TTN by combining all nodes.

        (WARNING: Can get very costly very fast. Only use for debugging.)

        Args:
            ttn (TreeTensorNetwork): The TTN to be contracted.
            to_copy (bool): Wether or not the contraction should be perfomed on
                a deep copy. Default is False.

        Returns:
            Tuple[np.ndarray, List[str]]: The contracted TTN and the list of
                the identifiers of the contracted nodes in the order they were
                contracted. The latter is very useful for debugging.
        """
        return completely_contract_tree(self, to_copy=to_copy)

TTN = TreeTensorNetwork

def pull_tensor_from_different_ttn(old_ttn: TreeTensorNetwork,
                                   new_ttn: TreeTensorNetwork,
                                   node_id: str,
                                   mod_fct: Union[Callable, None] = None):
    """
    Pulls a tensor from a different TTN into the current TTN.

    Args:
        old_ttn (TreeTensorNetwork): The TTN from which the tensor is pulled.
        new_ttn (TreeTensorNetwork): The TTN into which the tensor is pulled.
        node_id (str): The identifier of the node in the old TTN.
        mod_fct (Union[Callable, None], optional): A function to modify the
            children identifiers of the node in the new TTN to match the
            identifiers in the old TTN. Defaults to None.
    
    """
    old_node = old_ttn.nodes[node_id]
    new_node = new_ttn.nodes[node_id]
    # Find a potential permutation of the neighbours
    # The children in the new state are the basis change tensors
    perm = relative_leg_permutation(old_node, new_node,
                                    modify_function=mod_fct)
    old_tensor = old_ttn.tensors[node_id]

    new_node._shape = old_tensor.shape 
    new_node._leg_permutation = perm

    new_ttn.replace_tensor(node_id, deepcopy(old_tensor), perm)


def get_tensor_from_different_ttn(old_ttn: TreeTensorNetwork,
                                  new_ttn: TreeTensorNetwork,
                                  node_id: str,
                                  mod_fct: Union[Callable, None] = None
                                  ) -> np.ndarray:
    """
    Gets a tensor from one TTN and transforms it to fit with the leg order of
    another TTN.

    To directly insert into the other TTN use `pull_tensor_from_different_ttn`.

    Args:
        old_ttn (TreeTensorNetwork): The TTN from which the tensor is pulled.
        new_ttn (TreeTensorNetwork): The TTN into which the tensor is pulled.
        node_id (str): The identifier of the node in the old TTN.
        mod_fct (Union[Callable, None], optional): A function to modify the
            children identifiers of the node in the new TTN to match the
            identifiers in the old TTN. Defaults to None.
    
    Returns:
        np.ndarray: The tensor from the old TTN in the leg order of the new
            TTN. Might be a view of the original tensor.
    """
    old_node, old_tensor = old_ttn[node_id]
    new_node = new_ttn.nodes[node_id]
    # Find a potential permutation of the neighbours
    perm = relative_leg_permutation(new_node, old_node,
                                    modify_function=mod_fct)
    return old_tensor.transpose(perm)
