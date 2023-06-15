from __future__ import annotations
from typing import List, Dict

from .node import Node


class LegNode(Node):
    """ 
    The leg node contains a permutation that is responsible for everything
    that has to do with legs. On the other hand the superclass Node contains
    all that has to do with tree connectivity.

    The attribute `leg_permutation` is a list of integers with the same length
    as the associated tensor has dimensions. The associated permutation is such
    that the associated tensor transposed with it has the leg ordering:
        `(parent, child0, ..., childN-1, open_leg0, ..., open_legM-1)`
    Is compatible with `np.transpose`.
    So in the permutation we have the format
        `[leg of tensor corr. to parent, leg of tensor corr. to child0, ...]`
    The children legs are in the same order as the children node identifiers in 
    the superclass.
    """

    def __init__(self, tensor: ndarray, tag=None, identifier=None):
        super().__init__(tag, identifier)

        self._leg_permutation = list(range(tensor.ndim))

    @property
    def leg_permutation(self):
        """
        Get the leg permutation, cf. class docstring.
        """
        return self._leg_permutation
    
    @property
    def parent_leg(self) -> List[str, int]:
        """
        Returns parent_leg according to original implementation
        """
        if super().is_root():
            errstring = f"Node with identifier {super().identifier} has no parent!"
            raise ValueError(errstring)
        return {super().parent: 0}
    
    @property
    def children_legs(self) -> Dict[str, int]:
        """ 
        Returns the children_legs according to original implementation.

        Returns:
            Dict[str, int]: The children_legs according to original implementation. The keys are the
                children identifiers and the values are the indices in the permutation list (NOT the
                actual tensor legs)
        """
        return {child_id: index + (not super().is_root) for index, child_id in enumerate(super().children)}
    
    @property
    def open_legs(self) -> List[int]:
        """
        Returns the indices of the open legs in the permutation list (NOT the actual tensor legs).
        """
        return list(range(self.nvirt_legs(), self.nlegs()))

    def open_leg_to_parent(self, open_leg: int, parent_id: str):
        """
        Changes an open leg into the leg towards a parent.

        Args:
            open_leg (int): The index of the actual tensor leg
            parent_id (str): The identifier of the to be parent node
        """
        # Move value open_leg to front of list
        self._leg_permutation.remove(open_leg)
        self._leg_permutation.insert(0, open_leg)
        super().add_parent(parent_id)

    def open_leg_to_child(self, open_leg: int, child_id: str):
        """
        Changes an open leg into the leg towards a child.
        Children legs will be assorted in the same way as their ids are in the superclass.

        Args:
            open_leg (int): The index of the actual tensor leg
            child_id (str): The identifier of the to be child node
        """
        self._leg_permutation.remove(open_leg)
        new_position = super().is_root() + super().nchildren()
        self._leg_permutation.insert(new_position, open_leg)
        super().add_child(child_id)

    def open_legs_to_children(self, open_leg_list: List[int], identifier_list: List[str]):
        """
        Changes multiple open legs to be legs towards children.

        Args:
            open_leg_list (List[int]): List of actual tensor leg indices
            identifier_list (List[str]): List of the to be children nodes
        """
        for open_leg, child_id in zip(open_leg_list, identifier_list):
            self.open_leg_to_child(open_leg, child_id)

    def parent_leg_to_open_leg(self):
        """
        Changes the parent leg to be an open leg, if it exists.
        """
        if not super().is_root:
            leg_index = self._leg_permutation.pop(0)
            self._leg_permutation.append(leg_index)
            super().remove_parent()

    def child_leg_to_open_leg(self, child_id: str):
        """
        Changes a leg towards a child Node into an open_leg

        Args:
            child_id (str): The identifier of the child_nodem to be disconnected.
        """
        index = (not super().is_root()) + super().child_index(child_id)
        leg_index = self._leg_permutation.pop(index)
        self._leg_permutation.append(leg_index)
        super().remove_child(child_id)

    def children_legs_to_open_legs(self, children_id_list: List[str]):
        """
        Changes multiple child legs into open legs.

        Args:
            children_id_list (List[str]): A list of the identifiers of the child
                Nodes which are to be turned into open legs. 
        """
        for child_id in children_id_list:
            self.child_leg_to_open_leg(child_id)

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
