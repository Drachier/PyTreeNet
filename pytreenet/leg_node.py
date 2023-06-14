from __future__ import annotations
from typing import List

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
