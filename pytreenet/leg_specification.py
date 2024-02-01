"""
Intendend for internal use to specify legs before splitting tensors.
"""
from __future__ import annotations
from typing import Union, List

from .node import Node

class LegSpecification():
    """
    Contains useful function and all required data to specify which legs
     are part of which tensor in case of a tensor splitting.
    """

    def __init__(self, parent_leg: Union[str, None], child_legs: List[str],
                 open_legs: List[int], node: Union[Node, None]=None):
        self.parent_leg = parent_leg
        if child_legs is None:
            self.child_legs = []
        else:
            self.child_legs = child_legs
        if open_legs is None:
            self.open_legs = []
        else:
            self.open_legs = open_legs
        self.node = node
        self.is_root = False

    def __eq__(self, other: LegSpecification) -> bool:
        """
        Two Leg specifications are equal, if they contain the same legs and
         correspond to the same node by identifier.
        """
        if self.parent_leg is None:
            parents_eq = other.parent_leg is None
        elif other.parent_leg is None:
            return False
        else:
            parents_eq = self.parent_leg == other.parent_leg
        children_eq = self.child_legs == other.child_legs
        open_eq = self.open_legs == other.open_legs
        if self.node is None:
            node_eq = other.node is None
        elif other.node is None:
            return False
        else:
            node_eq = self.node.identifier == other.node.identifier
        return parents_eq and children_eq and open_eq and node_eq

    def __str__(self):
        string =  f"parent_leg: {self.parent_leg}, "
        string += f"child_legs: {self.child_legs}, "
        string += f"open_legs: {self.open_legs}, "
        if self.node is None:
            string += f"node_id: {self.node}, "
        else:
            string += f"node_id: {self.node.identifier}, "
        string += f"is_root: {self.is_root}"
        return string

    def find_leg_values(self) -> List[str]:
        """
        Finds the index values of the tensor legs specified in this class
         based on the legs in node

        Returns:
            List[str]: The leg values specified in the order
                `[parent_leg, child_legs, open_legs]`
        """
        assert self.node is not None
        if self.parent_leg is not None:
            leg_vals = [0]
        else:
            leg_vals = []
        leg_vals.extend([self.node.get_neighbour_leg(child_leg)
                         for child_leg in self.child_legs])
        leg_vals.extend(self.open_legs)
        return leg_vals

    def find_all_neighbour_ids(self) -> List[str]:
        """
        Returns all identifiers of neighbours of the node specified in this instance.

        Returns:
            List[str]: All identifiers of neighbours of node, the parent is the first.
        """
        if self.parent_leg is not None:
            n_ids = [self.parent_leg]
        else:
            n_ids = []
        n_ids.extend(self.child_legs)
        return n_ids
