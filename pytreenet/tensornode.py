import numpy as np
import uuid

class TensorNode(object):
    """
    A node in a tree tensor network that contains a tensor and which legs
    are contracted to which other tensors.

    General structure an parts of the code from treelib.node
    """

    def __init__(self, tensor, tag=None, identifier=None):

        self._tensor = tensor
        self._identifier = identifier
        self._tag = tag

        self._open_legs = list(np.arange(tensor.ndim))
        self._parent_leg = []
        self._children_legs = dict()

    @property
    def tensor(self):
        """
        The tensor in form of a numpy array associated to a tensornode.
        """
        return self._tensor

    @tensor.setter
    def tensor(self, new_tensor):
        """
        Set value of tensor. Can be used to update the tensor of a node.
        """
        assert self.tensor.shape == new_tensor.shape, "Tensors in the same position should have the same dimensions."
        self._tensor = new_tensor

    @property
    def identifier(self):
        """
        An identifier that is unique to this node.
        """
        return self._identifier

    @identifier.setter
    def identifier(self, new_identifier):
        if new_identifier == None:
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
        if new_tag == None:
            self._tag = self.identifier
        else:
            self._tag = new_tag

    @property
    def open_legs(self):
        """
        A list of tensor legs that are not contracted with other tensors
        """
        return self._open_legs

    @property
    def parent_leg(self):
        """
        A one or zero element dictionary, that potentially contains the parent's
        identifier as key and the tensor leg that is contracted with it.
        """
        return self._parent_leg

    @property
    def children_legs(self):
        """
        The legs contracted with the children's tensors.
        The dictionary contains the children's identifier as key and the
        corresponding contracted leg as value.
        """
        return self._children_legs

    def open_leg_to_parent(self, open_leg_index, parent_id):
        """
        Change an open leg to be a leg contracted with the parent node.
        """
        assert len(self.open_legs) > 0, "There are no remaining open legs."
        
        self._parent_leg.append(parent_id)
        self._parent_leg.append(open_leg_index)        
        self._open_legs.remove(open_leg_index)

    def open_legs_to_children(self, open_leg_list, identifier_list):
        """
        Change a list of open legs to legs contracted with children.
        """
        open_leg_list = list(open_leg_list)
        identifier_list = list(identifier_list)
        
        assert len(self.open_legs) > 0, "There are no remaining open legs."
        assert len(open_leg_list) == len(identifier_list), "Children and identifier list should be the same length"

        new_children_legs = dict(zip(identifier_list, open_leg_list))
        self._children_legs.update(new_children_legs)
        self._open_legs = [open_leg for open_leg in self._open_legs if open_leg not in open_leg_list]

    def open_leg_to_child(self, open_leg, child_id):
        """
        Only changes a single open leg to be contracted with a child
        """
        self.open_legs_to_children([open_leg], [child_id])
        
    def parent_leg_to_open_leg(self):
        """
        If existant, changes the leg contracted with a parent node, to an
        open leg. (Note: this will remove any relation of this node to the parent)
        """
        self.open_legs.append(self.parent_leg[1])
        self._parent_leg = []
        
    def children_legs_to_open_legs(self, children_identifier_list):
        """
        Makes legs contracted with children identified in children_identifier_list
        into open legs.
        """
        children_identifier_list = list(children_identifier_list)

        assert all(identifiers in self.children_legs for identifiers in children_identifier_list), "All identifiers must correspond a child of the node."

        children_legs_list = [self.children_legs[identifier] for identifier in children_identifier_list]
        self._open_legs.extend(children_legs_list)

        for identifier in children_identifier_list:
            del self._children_legs[identifier]

    def child_leg_to_open_leg(self, child_identifier):
        """
        Makes a leg contracted with the child identified by child_identifier
        into an open leg.
        """
        assert child_identifier in self.children_legs, "Identifier should belong to a child of this node."

        self.children_legs_to_open_legs(child_identifier)

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

