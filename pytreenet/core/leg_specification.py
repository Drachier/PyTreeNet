"""
This module is used to unambigously define tensor legs.

It contains the LegSpecification class. This class allows the definition of
legs with regards to legs towards neighbours and open legs. It is commonly
used to define the which legs belong to which resulting node yielded by the
splitting of a tensor. It allows the use of node identifiers rather than leg
indices, avoiding potential complications.

Example:
    Assume we want to perform the following split

                |open1  |open 0                             |open1          |open0
             ___|_______|___                             ___|___         ___|___
     child1 |               |   parent           child1 |       |  new  |       |   parent
        ----|       C       |-----          -->     ----|   A   |-------|   B   |----
            |_______________|                           |_______|       |_______|
                |       |                                   |               |
          child2|       |child3                       child2|         child3|
    
    the corresponding code would be:
    ```python
        # Creating the mother node
        ctensor = crandn((2,2,2,2,2,2))
        C = Node(identifier="C",tensor=ctensor)
        C.open_leg_to_parent("parent",0)
        children = {"child1":1,"child2":2,"child3":3}
        C.open_leg_to_children()

        # Defining the neighbouring legs
        parent_legA = None
        parent_legB = "parent"
        children_legsA = ["child1","child2"]
        children_legsB = ["child3"]

        # Defining the open legs
        open_legsA = [5]
        open_legsB = [4]

        # The LegSpecification
        legsA = LegSpecification(parent_legA, children_legsA, open_legsA
                                 node=C)
        legsB = LegSpecification(parent_legB, children_legsB, open_legsB
                                 node=C)
    ```
"""
from __future__ import annotations
from typing import Union, List

from .node import Node

class LegSpecification():
    """
    Contains data to describe a split-off node's legs fully.

    In case a splitting of node occurs, a LegSpecification contains all the
    data required to define one of the new nodes. Can also be used in other
    cases where a convenient storage of node legs is required.

    Attributes:
        parent_leg (Union[str,None]): The identifier of a potential parent.
            If it is None, the node is either a root or the parent node is
            created after the LegSpecification and the identifier is not yet
            known.
        children_legs (List[str]): The identifiers of the children that should
            be associated with this node.
        open_legs (List[int]): A list of open leg indices. The indices are
            usually the indices of the tensor before a split or similar
            operation. They are usually not the indices of the open legs of
            newly created nodes.
        node (Union[Node,None], optional): A node that is required to
            translate the   stored identifiers into indices. Can be None, as
            the node needing it might be created at a later point. Defaults to
            None.
        is_root (bool): Determines, if the node associated before a potential
            split is a root and the node this specification is for should
            suceed it in that. Not having a parent leg is insufficient, as this
            might be created at a later date.
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
