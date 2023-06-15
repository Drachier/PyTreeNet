from __future__ import annotations
import numpy as np
import uuid

from copy import deepcopy

from .util import crandn, copy_object


class Node(object):
    """
    A node in a tree tensor network and which legs are contracted to which other tensors.

    General structure and parts of the code from treelib.node

    """

    def __init__(self, tag=None, identifier=None):

        if identifier == None:
            self._identifier = str(uuid.uuid1())
        else:
            self._identifier = str(identifier)
        if tag == None:
            self._tag = self.identifier
        else:
            self._tag = tag

    @property
    def identifier(self):
        """
        A string that is unique to this node.
        """
        return self._identifier

    @identifier.setter
    def identifier(self, new_identifier):
        if new_identifier is None or new_identifier == "":
            self._identifier = str(uuid.uuid1())
        else:
            self._identifier = str(new_identifier)

    @property
    def tag(self):
        """
        A human readable tag for this node.
        """
        return self._tag

    @tag.setter
    def tag(self, new_tag):
        if new_tag is None or new_tag == "":
            self._tag = self.identifier
        else:
            self._tag = new_tag

    def neighbouring_nodes(self, with_legs=True):
        """
        Finds the neighbouring tensor nodes of this node with varying
        additional information.

        Parameters
        ----------
        with_legs : boolean, optional
            If True the legs of neighbours are also returned. The default is True.

        Returns
        -------
        neighbour_legs: dict
            Is returned, if with_legs=True. A dictionary that contains all the
            identifiers of tensor nodes that are contracted with this node and
            the leg they are attached to.
        neighbour_ids: list of str
            Is returned if with_legs=False. A list containing the identifiers
            of all the tensor nodes this node is contracted with.
        """

        if with_legs:
            parent_dict = self.parent_leg_dict()
            neighbour_legs = deepcopy(self.children_legs)
            neighbour_legs.update(parent_dict)
            return neighbour_legs

        else:
            neighbour_ids = list(self.children_legs.keys())
            if not self.is_root():
                neighbour_ids.append(self.parent_leg[0])
            return neighbour_ids

    def is_root(self):
        """
        Determines if this node is a root node, i.e., a node without a parent.
        """
        if len(self.parent_leg) == 0:
            return True
        else:
            return False

    def has_x_children(self, x: int):
        """
        Determines if the node has at least x-many children
        """
        assert x > 0, "The number of children will be at least zero. Choose a bigger number."

        if len(self._children_legs) >= x:
            return True
        else:
            return False

    def is_leaf(self):
        """
        Determines if the node is a leaf, i.e., has at least one child.
        """
        return not self.has_x_children(x=1)

    def has_open_leg(self):
        """
        Determines if the node has any open legs.
        """
        return len(self.open_legs) > 0

    def is_child_of(self, other_node_id):
        """
        Determines if this instance is the child of the node with identifier
        other_node_id
        """
        return other_node_id in self.parent_leg

    def is_parent_of(self, other_node_id):
        """
        Determines if this instance is the parent of the node with identifier
        other_node_id
        """
        return other_node_id in self.children_legs
    

    ######## Newly Added Stuff ########
    def add_parent(self, parent_id: str):
        """
        Adds `parent_id` as the new parent.
        """
        self.parent = parent_id

    def remove_parent(self):
        """
        Removes parent and replaces it by None
        """
        self.parent = None

    def add_child(self, child_id: str):
        """
        Adds `child_id` as a new child.
        """
        self.children.append(child_id)

    def remove_child(self, child_id: str):
        """
        Removes a children identifier from this node's children.
        """
        self.children.remove(child_id)

    def child_index(self, child_id: str):
        """
        Returns the index of identifier child_id in this Node's children list.
        """
        return self.children.index(child_id)

    def nchildren(self) -> int:
        """
        The number of children of this node
        """
        return len(self.children)

    def nneighbours(self) -> int:
        """
        Returns the number of neighbours of this node.
        """
        return self.nchildren + (not self.is_root())

def random_tensor_node(shape, tag=None, identifier=None):
    """
    Creates a tensor node with an a random associated tensor with shape=shape.
    """
    rand_tensor = crandn(shape)
    return Node(tensor=rand_tensor, tag=tag, identifier=identifier)


def assert_legs_matching(node1, leg1_index, node2, leg2_index):
    """
    Asserts if the dimensions of leg1 of node1 and leg2 of node2 match.
    """
    leg1_dimension = node1.shape[leg1_index]
    leg2_dimension = node2.shape[leg2_index]
    assert leg1_dimension == leg2_dimension


def conjugate_node(node, deep_copy=True, conj_neighbours=False):
    """
    Returns a copy of the same node but with all entries complex conjugated
    and with a new identifier and tag. If conj_neighbours all the identifiers
    in node's legs will have ann added "conj_" in front.
    """
    conj_node = copy_object(node, deep=deep_copy)
    new_identifier = "conj_" + conj_node.identifier
    new_tag = "conj_" + conj_node.tag

    conj_node.identifier = new_identifier
    conj_node.tag = new_tag

    if conj_neighbours:
        if not node.is_root():
            conj_node.parent_leg[0] = "conj_" + conj_node.parent_leg[0]
        children_legs = conj_node.children_legs
        conj_children_legs = {("conj_" + child_id): children_legs[child_id]
                              for child_id in children_legs}
        conj_node.children_legs = conj_children_legs

    conj_node.tensor = np.conj(conj_node.tensor)
    return conj_node


def random_tensor_node(shape, tag=None, identifier=None):
    """
    Creates a tensor node with an a random associated tensor with shape=shape.
    """
    rand_tensor = crandn(shape)
    return (Node(tensor=rand_tensor, tag=tag, identifier=identifier), rand_tensor)
