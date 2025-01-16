from __future__ import annotations
from typing import List, Dict, Union, Tuple, Callable
from functools import reduce

from numpy import ndarray

from .graph_node import (GraphNode,
                         find_child_permutation_neighbour_index)
from ..util.std_utils import permute_tuple
from ..util.ttn_exceptions import NotCompatibleException


class Node(GraphNode):
    """ 
    The node class is responsible for the control of the legs.

    This means, it keeps track of the leg order or more precisely the
    permutation of the legs compared to the data tensor in order to keep the
    leg convention

    ``(parent, child0, ..., childN-1, open_leg0, ..., open_legM-1)``
    
    A Node can be created independently from a tensor. It can then be linked to
    this tensor afterwards.
    
    Attributes:
        identifier (str): The identifier of the node.
        leg_permutation (List[int]): The permutation of the legs compared to
            the data tensor. The values are the indices of the tensor legs at
            the positions they need to be permuted to. In general leg indices
            are returned according to the positions in this list and not according
            to the actual tensor legs.
        shape (Tuple): The shape of the tensor associated with this node. This
            is the shape of the tensor after the permutation of the legs and
            allows for more convenient checks and sanity checks.
    """

    def __init__(self,
                 tensor: Union[ndarray,None] = None,
                 identifier: str = ""):
        super().__init__(identifier)
        if tensor is not None:
            self._leg_permutation = list(range(tensor.ndim))
            self._shape = tensor.shape
        else:
            self._leg_permutation = None
            self._shape = None

    def __eq__(self, other: Node) -> bool:
        """
        Provides the equality check for two nodes.

        Checks equality of two nodes. Two nodes are considered equal if they 
        have the same identifier, children in the correct order, the same 
        parent, and the same external shape. The internal `_shape` can differ.

        Note: The permutation and the associated tensor are not checked, as 
        the tensor is stored separately.
        """
        if self.shape != other.shape:
            return False
        return super().__eq__(other)

    @property
    def leg_permutation(self):
        """
        Get the leg permutation, cf. class docstring.
        """
        return self._leg_permutation

    @property
    def shape(self) -> Union[Tuple, None]:
        """
        Returns the shape as it would be for the tranposed tensor.

        E.g. the dimension of the parent leg is always output[0].
        If no tensor is linked to this node, None is returned.
        """
        if self._shape is None:
            return None
        return permute_tuple(self._shape, self._leg_permutation)

    @property
    def parent_leg(self) -> int:
        """
        Returns the parent_leg as index.

        Returns:
            int: The parent_leg where the value is the position in the permutation
                list. If there is no parent, None is returned.
        """
        if self.is_root():
            return None
        return 0

    @property
    def children_legs(self) -> List[int]:
        """ 
        Returns the children_legs as index list.

        Returns:
            List[int]: The children_legs where the values are the positions in
             the permutation list.
        """
        return list(range(self.nparents(), self.nparents() + self.nchildren()))

    @property
    def open_legs(self) -> List[int]:
        """
        Returns the indices of the open legs.
        """
        return list(range(self.nvirt_legs(), self.nlegs()))

    def __str__(self) -> str:
        """
        Returns a string representation of the node.
        """
        string = f"Node {self.identifier}\n"
        string += f"Parent: {self.parent}\n"
        string += f"Children: {self.children}\n"
        string += f"Open legs: {self.open_legs}\n"
        string += f"Shape: {self.shape}\n"
        return string

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

    def replace_tensor(self,
                       tensor: ndarray,
                       permutation: Union[None,Tuple[int]] = None):
        """
        Replaces the tensor associated to this node.

        Args:
            tensor (ndarray): The tensor to be linked with this node.
            permutation (Union[None,Tuple[int]]): The permutation of the legs
                compared to the current node legs. If None, the permutation is
                not changed.
        """
        if permutation is None and self.shape == tensor.shape:
            # In this case the tensor fits without changes needed.
            self._reset_permutation()
        elif permutation is not None and permute_tuple(tensor.shape, permutation) == self.shape:
            # The tensor fits with the permutation.
            self._leg_permutation = permutation
            self._shape = tensor.shape
        else:
            errstr = "Shapes of the tensor and the node do not match!"
            raise NotCompatibleException(errstr)

    def _open_leg_checks(self, open_leg: int,
                         other_id: Union[str, None] = None):
        """
        Checks if the given leg is an open leg.

        Args:
            open_leg (int): The index of the leg to be checked.
            other_id (Union[str, None]): The identifier of a different node to
                appear in the error message.
        """
        if self.nopen_legs() == 0:
            errstr = f"Node with identifier {self.identifier} has no open legs!"
            raise ValueError(errstr)
        if open_leg < self.nneighbours():
            errstr = f"The leg with index {open_leg} of {self.identifier} is not open"
            if other_id is None:
                errstr = errstr + " to be connected!"
            else:
                errstr = errstr + f" to connect to {other_id}"
            raise NotCompatibleException(errstr)

    def open_leg_to_parent(self,
                           parent_id: str,
                           open_leg: Union[int, None]):
        """
        Changes an open leg into the leg towards a parent.

        Args:
            parent_id (str): The identifier of the to be parent node
            open_leg (int): The index of the tensor leg
        """
        if not self.is_root():
            errstr = f"Node {self.identifier} already has a parent!"
            raise NotCompatibleException(errstr)
        if open_leg is None:
            return
        if parent_id is None:
            errstr = "None is not a legitimate parent identifier!"
            raise ValueError(errstr)
        self._open_leg_checks(open_leg, parent_id)
        # Move value open_leg to front of list
        actual_value = self._leg_permutation.pop(open_leg)
        self._leg_permutation.insert(0, actual_value)
        self.add_parent(parent_id)

    def open_leg_to_child(self, child_id: str, open_leg: int):
        """
        Changes an open leg into the leg towards a child.

        Children legs will be sorted in the same way as their ids are in the
        superclass.

        Args:
            child_id (str): The identifier of the to be child node
            open_leg (int): The index of the tensor leg
        """
        self._open_leg_checks(open_leg, child_id)
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

        The newly opened leg will be the last leg of the node.
        """
        if not self.is_root():
            leg_index = self._leg_permutation.pop(0)
            self._leg_permutation.append(leg_index)
        self.remove_parent()

    def child_leg_to_open_leg(self, child_id: str):
        """
        Changes a leg towards a child node into an open leg.

        The newly opened leg will be the last leg of the node.

        Args:
            child_id (str): The identifier of the child_nodem to be disconnected.
        """
        self._check_child_existence(child_id)
        index = self.neighbour_index(child_id)
        leg_index = self._leg_permutation.pop(index)
        self._leg_permutation.append(leg_index)
        self.remove_child(child_id)

    def children_legs_to_open_legs(self, children_id_list: List[str]):
        """
        Changes multiple child legs into open legs.

        The new legs will be the last legs of the node but in the same order
        as provided in the list.

        Args:
            children_id_list (List[str]): A list of the identifiers of the child
                nodes which are to be turned into open legs. 
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

    def swap_two_child_legs(self, child_id1: str, child_id2: str):
        """
        Swaps the index position of two children.
        """
        self._check_child_existence(child_id1)
        self._check_child_existence(child_id2)
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
            return 1
        return reduce(lambda x, y: x * y, open_dim)

    def parent_leg_dim(self) -> int:
        """
        Returns the dimension associated to the parent leg.
        """
        if self.is_root():
            errstr = f"Node {self.identifier} is root, thus does not have a parent!"
            raise NotCompatibleException(errstr)
        return self._shape[self._leg_permutation[0]]

    def neighbour_dim(self, neighbour_id: str) -> int:
        """
        Returns the dimension associated to the neighbour leg.

        Args:
            neighbour_id (str): The identifier of the neighbour node.
        """
        neighbour_index = self.neighbour_index(neighbour_id)
        return self.shape[neighbour_index]

# ---------------------------- Usefull functions using nodes ----------------------------
def relative_leg_permutation(old_node: Node,
                             new_node: Node,
                             modify_function: Union[Callable,None] = None,
                             modified_parent: bool = False) -> List[int]:
    """
    Calculates the relative permutation between two nodes.

    This is the permutation required to transform the legs of the old node
    to the legs of the new node. Thus applying the permutation to the tensor
    of the old node, will make it fit the leg order of the new node.

    Args:
        old_node (Node): The node to be transformed.
        new_node (Node): The node to be transformed to.
        modify_function (Union[Callable,None]): A function by which the 
            neighbour identifiers of the old node differ from the neighbour
            identifiers of the new node. If None, the neighbour identifiers
            are assumed to be the same.
        modified_parent (bool): If True, the parent leg can also be different.
    
    Returns:
        List[int]: The relative permutation.
    """
    if modified_parent:
        raise NotImplementedError
    else:
        child_perm = find_child_permutation_neighbour_index(old_node,
                                                            new_node,
                                                            modify_function)
        if new_node.is_root():
            perm = []
        else:
            perm = [new_node.parent_leg]
        perm += child_perm
        perm += new_node.open_legs
        return perm
