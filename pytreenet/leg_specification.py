"""
Intendend for internal use to specify legs before splitting tensors.
"""
from __future__ import annotations
from typing import Union, List

class LegSpecification():
    """
    Contains usefull function and all required data to specify which legs
     are part of which tensor in case of a tensor splitting.
    """
    def __init__(self, parent_leg: Union[str, None], child_legs: List[str],
                 open_legs: List[str], node: LegNode):
        self.parent_leg = parent_leg
        self.child_legs = child_legs
        self.open_legs = open_legs
        self.node = node

    @classmethod
    def from_dict(cls, dictionary, node) -> LegSpecification:
        """
        Creates an instance from a dictionary.

        Args:
            dictionary (Dict): Has to have the keywords "parent_legs", "child_legs",
             and "open_legs"
            node (LegNode): The related node
        """
        return cls(dictionary["parent_leg"], dictionary["child_legs"],
                   dictionary["open_legs"], node)

    def find_leg_values(self) -> List[str]:
        """
        Finds the index values of the tensor legs specified in this class
         based on the legs in node

        Returns:
            List[str]: The leg values specified in the order
                `[parent_leg, child_legs, open_legs]`
        """
        if self.parent_leg is not None:
            leg_vals = [0]
        else:
            leg_vals = []
        leg_vals.extend([self.node.get_neighbour_leg(self.child_legs[i])
                         for i in range(len(self.child_legs))])
        leg_vals.extend(self.open_legs)
        return leg_vals

    def find_all_neighbour_ids(self) -> List[str]:
        """
        Returns all identifiers of neighbours of node specified in this instance.

        Returns:
            List[str]: All identifiers of neighbours of node, the parent is the first.
        """
        if self.parent_leg is not None:
            n_ids = [self.parent_leg]
        else:
            n_ids = []
        n_ids.extend(self.child_legs)
        return n_ids
    