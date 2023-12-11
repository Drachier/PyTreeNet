from __future__ import annotations
from typing import List, Dict, Union
from copy import copy
from functools import reduce

from numpy import ndarray

from .util import crandn
from .graph_node import GraphNode
from .ttn_exceptions import NotCompatibleException


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

    def __init__(self, tensor=None, identifier=None):
        super().__init__(identifier)
        if tensor is not None:
            self._leg_permutation = list(range(tensor.ndim))
            self._shape = tensor.shape
        else:
            self._leg_permutation = None
            self._shape = None

    def __eq__(self, other: Node) -> bool:
        """
        Two nodes are equal, if they have the same identifier, children in the right order and
         the same parent.
        """
        identifier_eq = self.identifier == other.identifier
        children_eq = self.children == other.children
        # Needed to avoid string to None comparison
        if self.is_root() and other.is_root():
            parent_eq = True
        elif self.is_root() or other.is_root():
            parent_eq = False
        else:
            parent_eq = self.parent == other.parent
        return identifier_eq and children_eq and parent_eq

    @property
    def leg_permutation(self):
        """
        Get the leg permutation, cf. class docstring.
        """
        return self._leg_permutation

    @property
    def shape(self) -> Union[tuple, None]:
        """
        Returns the shape as it would be for the tranposed tensor.
        E.g. the dimension of the parent leg is always output[0].
        """
        if self._shape is None:
            return None
        return tuple([self._shape[i] for i in self._leg_permutation])

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
    
    def link_tensor(self, tensor: ndarray):
        """
        Links this node to a tensor, by saving its shape and dimension.

        Args:
            tensor (ndarray): The tensor to be linked with this node.
        """
        self._leg_permutation = list(range(tensor.ndim))
        self._shape = tensor.shape

    def _reset_permutation(self):
        """
        Resets the permutation to the standard.
        Always call this, when the associated tensor is transposed
            according to the permutation. This ensures, the legs still match.
        """
        self._shape = self.shape
        self._leg_permutation = list(range(len(self._leg_permutation)))

    def open_leg_to_parent(self, parent_id: [str, None], open_leg: Union[int,None]):
        """
        Changes an open leg into the leg towards a parent.

        Args:
            parent_id (str): The identifier of the to be parent node
            open_leg (int): The index of the tensor leg
        """
        if not self.is_root():
            errstr = f"Node with identifier {self.identifier} already has a parent!"
            raise NotCompatibleException(errstr)
        if open_leg is None:
            return
        if parent_id is None:
            errstr = "None is not a legitimate parent identifier!"
            raise ValueError(errstr)
        if self.nopen_legs() == 0:
            errstr = f"Node with identifier {self.identifier} has no open legs!"
            raise ValueError(errstr)
        if open_leg < self.nneighbours():
            errstr = f"The leg with index {open_leg} of {self.identifier} is not open to connect to {parent_id}!"
            raise NotCompatibleException(errstr)

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
        if self.nopen_legs() == 0:
            errstr = f"Node with identifier {self.identifier} has no open legs!"
            raise ValueError(errstr)
        if open_leg < self.nneighbours():
            errstr = f"The leg with index {open_leg} of {self.identifier} is not open to connect to {child_id}!"
            raise NotCompatibleException(errstr)

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
        actual_value = {child_id: self._leg_permutation[open_leg]
                        for child_id, open_leg in child_dict.items()}
        original_nneighbours = self.nneighbours()
        for child_id, value in actual_value.items():
            new_position = self.nvirt_legs()
            if child_dict[child_id] < original_nneighbours:
                errstr = f"The leg with index {child_dict[child_id]} of {self.identifier} is not open to connect to {child_id}!"
                raise NotCompatibleException(errstr)
            self._leg_permutation.remove(value)
            self._leg_permutation.insert(new_position, value)
            self.add_child(child_id)

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

    def exchange_open_leg_ranges(self, open_1: range, open_2: range):
        """
        Exchanges two continuous batches of open legs, with one another.

        Args:
            open_1, open_2 (range): Each is one batch of open legs.
        """
        assert open_1.step == 1
        assert open_2.step == 1
        if open_2.start < open_1.start:
            open_1, open_2 = open_2, open_1
        assert open_1.stop <= open_2.start

        values2 = [self._leg_permutation.pop(open_2.start)
                   for _ in open_2]
        values1 = [self._leg_permutation.pop(open_1.start)
                   for _ in open_1]
        self._leg_permutation[open_1.start:open_1.start] = values2
        difference = open_2.start - open_1.stop
        new_position = open_1.start + len(open_2) + difference
        self._leg_permutation[new_position:new_position] = values1

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

    def open_dimension(self) -> int:
        """
        Returns the total open dimension of this leg.
        """
        open_dim = [self.shape[leg]
                    for leg in self.open_legs]
        if open_dim == []:
            return 0
        return reduce(lambda x, y: x * y, open_dim)

    def parent_leg_dim(self) -> int:
        """
        Returns the dimension associated to the parent leg.
        """
        if self.is_root():
            errstr = f"Node {self.identifier} is root, thus does not have a parent!"
            raise NotCompatibleException(errstr)
        return self._shape[self._leg_permutation[0]]

def random_tensor_node(shape, identifier: str = ""):
    """
    Creates a tensor node with an a random associated tensor with shape=shape.
    """
    rand_tensor = crandn(shape)
    return (Node(tensor=rand_tensor, identifier=identifier), rand_tensor)
