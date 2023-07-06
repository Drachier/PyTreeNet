from __future__ import annotations
from typing import List, Dict
from copy import copy

from numpy import ndarray

from pytreenet.util import crandn

from .graph_node import GraphNode


class Node(GraphNode):
    """ 
    The Node contains a permutation that is responsible for everything
    that has to do with legs. 
    The superclass AbstractNode contains all that has to do with tree connectivity.

    The attribute `leg_permutation` is a list of integers with the same length
    as the associated tensor has dimensions. The associated permutation is such
    that the associated tensor transposed with it has the leg ordering:
        `(parent, child0, ..., childN-1, open_leg0, ..., open_legM-1)`
    In general legs values will be returned according to this ordering and not according
    to the actual tensor legs.
    Is compatible with `np.transpose`.
    So in the permutation we have the format
        `[leg of tensor corr. to parent, leg of tensor corr. to child0, ...]`
    The children legs are in the same order as the children node identifiers in 
    the superclass.
    """

    def __init__(self, tensor=None, tag=None, identifier=None):
        super().__init__(tag, identifier)
        if tensor is not None:
            self._leg_permutation = list(range(tensor.ndim))
            self._shape = tensor.shape
        else:
            self._leg_permutation = None
            self._shape = None

    @classmethod
    def from_node(cls, tensor: ndarray, node: GraphNode) -> Node:
        """
        Generates a `Node` object from a `GraphNode` and a tensor.

        Args:
            tensor (ndarray): The tensor associated to this node
            node (Node): A node with the same identifier as this new instance

        Returns:
            LegNode:
        """
        leg_node = cls(tensor=tensor, tag=node.tag, identifier=node.identifier)
        leg_node.parent = node.parent
        leg_node.children = copy(node.children)

        return leg_node

    @property
    def leg_permutation(self):
        """
        Get the leg permutation, cf. class docstring.
        """
        return self._leg_permutation

    def set_leg_permutation(self, leg_dict: Dict[str, int]):
        """
        Set a new leg permutation. This might be neccessary, if the tensor
        corresponding to this node is changed externally and then placed back
        into the ttn.

        All neighbours are assigned a new leg value, while the open legs are
         assinged the remaining values. This assumes that the order of open legs
         was not changed.

        Args:
            leg_dict (Dict[str, int]): A dictionary containing all the neighbouring
                nodes as keys and the actual tensor leg value as value.
        """
        self._reset_permutation()
        neighbours = self.neighbouring_nodes()
        assert len(neighbours) == len(leg_dict)
        for node_id, leg_value in leg_dict.items():
            if node_id in neighbours:
                new_position = self.get_neighbour_leg(node_id)
                self._leg_permutation.pop(leg_value)
                self._leg_permutation.insert(new_position, leg_value)
            else:
                errstr = f"Nodes {self.identifier} and {node_id} aren't neighbours!"
                raise ValueError(errstr)

    def link_tensor(self, tensor: ndarray):
        self._leg_permutation = list(range(tensor.ndim))
        self._shape = tensor.shape

    def _reset_permutation(self):
        """
        Resets the permutation to the standard.
        Always call this, when the associated tensor is transposed
            according to the permutation. This ensures, the legs still match.
        """
        self._leg_permutation = list(range(len(self._leg_permutation)))

    @property
    def parent_leg(self) -> int:
        """
        Returns parent_leg according to original implementation.
        """
        return 0

    @property
    def children_legs(self) -> list[int]:
        """ 
        Returns the children_legs according to original implementation.

        Returns:
            Dict[str, int]: The children_legs according to original implementation. The keys are the
                children identifiers and the values are the indices in the permutation list (NOT the
                actual tensor legs)
        """
        return list(range(self.nparents(), self.nparents() + self.nchildren()))

    @property
    def open_legs(self) -> List[int]:
        """
        Returns the indices of the open legs.
        """
        return list(range(self.nvirt_legs(), self.nlegs()))

    def open_leg_to_parent(self, parent_id: str, open_leg: int):
        """
        Changes an open leg into the leg towards a parent.

        Args:
            parent_id (str): The identifier of the to be parent node
            open_leg (int): The index of the tensor leg
        """
        if self.nopen_legs() == 0:
            errstr = f"Node with identifier {self.identifier} has no open legs!"
            raise ValueError(errstr)

        # Move value open_leg to front of list
        actual_value = self._leg_permutation.pop(open_leg)
        self._leg_permutation.insert(0, actual_value)

        self.add_parent(parent_id)

    def open_leg_to_child(self, child_id: str, open_leg: int):
        """
        Changes an open leg into the leg towards a child.
        Children legs will be sorted in the same way as their ids are in the superclass.

        Args:
            child_id (str): The identifier of the to be child node
            open_leg (int): The index of the tensor leg
        """
        if self.nopen_legs() < 1:
            errstr = f"Node with identifier {self.identifier} has no open legs!"
            raise ValueError(errstr)

        actual_value = self._leg_permutation.pop(open_leg)
        new_position = self.nparents() + self.nchildren()
        self._leg_permutation.insert(new_position, actual_value)

        self.add_child(child_id)

    def open_legs_to_children(self, child_dict: Dict[str, int]):
        """
        Changes multiple open legs to be legs towards children.

        Args:
            child_dict (Dict[str, int]): A dictionary that contains the identifiers of
             the to be children nodes as keys and the open leg that they should contract
             to as values.
        """
        for child_id, open_leg in child_dict.items():
            self.open_leg_to_child(child_id, open_leg)

    def parent_leg_to_open_leg(self):
        """
        Changes the parent leg to be an open leg, if it exists.

        Moves it to the back.
        """
        if not self.is_root():
            leg_index = self._leg_permutation.pop(0)
            self._leg_permutation.append(leg_index)

        self.remove_parent()

    def child_leg_to_open_leg(self, child_id: str):
        """
        Changes a leg towards a child Node into an open_leg

        Args:
            child_id (str): The identifier of the child_nodem to be disconnected.
        """
        index = self.nparents() + self.child_index(child_id)
        leg_index = self._leg_permutation.pop(index)
        self._leg_permutation.append(leg_index)

        self.remove_child(child_id)

    def children_legs_to_open_legs(self, children_id_list: List[str]):
        """
        Changes multiple child legs into open legs.

        Args:
            children_id_list (List[str]): A list of the identifiers of the child
                Nodes which are to be turned into open legs. 
        """
        for child_id in children_id_list:
            self.child_leg_to_open_leg(child_id)

    def last_leg_to_parent_leg(self):
        """
        Sometimes the leg of the parent is not set properly, when something
        has been done to the tensor. This method takes care of this.
        Use with care!
        """
        value = self._leg_permutation.pop(-1)
        self._leg_permutation.insert(0, value)

    def leg_to_last_child_leg(self, leg_value: int):
        """
        Sometimes the leg of the last child is not set properly, when something
        has been done to the tensor. This method takes care of this.
        Use with care!
        """
        self._leg_permutation.remove(leg_value)
        new_position = self.nparents() + self.nchildren() - 1
        self._leg_permutation.insert(new_position, leg_value)

    def get_child_leg(self, child_id: str) -> int:
        """
        Obtains the leg value of a given child of this node.

        This is the leg of the tensor corresponding to this child after
        transposing the tensor accordingly.
        """
        return self.nparents() + self.child_index(child_id)

    def get_neighbour_leg(self, node_id: str) -> int:
        """
        Returns the leg value of a given neighbour.

        This is the leg of the tensor corresponding to this neighbour after
        transposing the tensor accordingly.
        """
        if self.is_child_of(node_id):
            return 0
        if self.is_parent_of(node_id):
            return self.get_child_leg(node_id)
        errstr = f"Node {self.identifier} is not connected to {node_id}!"
        raise ValueError(errstr)

    def swap_two_child_legs(self, child_id1: str, child_id2: str):
        """
        Swaps the index position of two children.
        """
        if child_id1 == child_id2:
            return

        child1_index = self.child_index(child_id1)
        child2_index = self.child_index(child_id2)
        # Swap children identifiers
        self.children[child1_index], self.children[child2_index] =\
            self.children[child2_index], self.children[child1_index]
        # Swap their leg value
        c1_index = child1_index + self.nparents()
        c2_index = child2_index + self.nparents()
        self._leg_permutation[c1_index], self._leg_permutation[c2_index] =\
            self._leg_permutation[c2_index], self._leg_permutation[c1_index]

    def swap_with_first_child(self, child_id: str):
        """
        Makes the leg of the given child the first of all children legs
        """
        self.swap_two_child_legs(child_id, self.children[0])

    def nlegs(self) -> int:
        """
        Returns the total number of legs of this node.
        """
        return len(self._leg_permutation)

    def nchild_legs(self) -> int:
        """
        Returns the number of legs connected to child Nodes.
        """
        return super().nchildren()

    def nvirt_legs(self):
        """
        Returns the total number of legs to other nodes, i.e. parent + children.
        """
        return super().nneighbours()

    def nopen_legs(self):
        """
        Returns the number of open legs of this node.
        """
        return self.nlegs() - self.nvirt_legs()


def random_tensor_node(shape, tag: str = "", identifier: str = ""):
    """
    Creates a tensor node with an a random associated tensor with shape=shape.
    """
    rand_tensor = crandn(shape)
    return (Node(tensor=rand_tensor, tag=tag, identifier=identifier), rand_tensor)
