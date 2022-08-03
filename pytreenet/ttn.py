import copy
import uuid
import numpy as np

from .tensornode import assert_legs_matching, TensorNode
from .tnn_exceptions import NoConnectionException

class TreeTensorNetwork(object):
    """
    A tree tensor network (TTN) a tree, where each node contains a tensor,
    that is part of the network. Here a tree tensor network is a dictionary
    _nodes of tensor nodes with their identifiers as keys.

    General structure and parts of the codes are from treelib.tree
    """

    def __init__(self, original_tree = None, deep = False):
        """
        Initiates a new TreeTensorNetwork or a deep or shallow copy of a
        different one.
        """

        self._nodes = dict()

        self._root_id = None
        if original_tree is not None:
            self._root_id = original_tree.root_id

            if deep:
                for node_id in original_tree.nodes:
                    self._nodes[node_id] = copy.deepcopy(original_tree.nodes[node_id])
                else:
                    self._nodes = original_tree.nodes

    @property
    def nodes(self):
        """
        A dictionary containing the tensor trees notes via their identifiers.
        """
        return self._nodes

    @property
    def root_id(self):
        """
        The root's identifier.'
        """
        return self._root_id

    def __contains__(self, identifier):
        """
        Determines if a node with identifier is in the TTN.
        """
        return identifier in self._nodes

    def __getitem__(self, key):
        """
        Return _nodes[key]
        """
        return self._nodes[key]

    def __len__(self):
        return len(self._nodes)

    def __setitem__(self, node):
        "TODO: Once the update methods are established this can be worked on."

    def check_no_nodeid_dublication(self, node_id):
        """
        Checks if node_id already exists in the TTN
        """
        return node_id not in self.nodes

    def assert_no_nodeid_dublication(self, node_id):
        """
        Asserts if node_id already exists in the TTN
        """
        assert self.check_no_nodeid_dublication(node_id), f"Tensor node with identifier {node_id} already exists in TTN"

    def assert_id_in_tree(self, node_id):
        """
        Asserts if the node with the identifier node_id is in the tree.
        """
        assert node_id in self.nodes, f"Tensor node with identifier {node_id} is not part of the TTN."

    def add_root(self, node):
        """
        Adds a root tensor node to the TTN.
        """
        assert self.root_id == None, "A TTN can't have two roots."
        self._root_id = node.identifier
        self.nodes.update({node.identifier: node})

    def add_child_to_parent(self, child, child_leg, parent_id, parent_leg):
        """
        Adds a tensor node to the TTN which is the child of the tensor node
        with identifier parent_id. The two tensors are contracted along one
        leg. The child via child_leg and the parent via parent_leg
        """
        assert parent_id in self.nodes, f"Parent with identifier {parent_id} has to be part of the TTN."

        parent = self.nodes[parent_id]
        child_id = child.identifier

        self.assert_no_nodeid_dublication(child_id)
        assert_legs_matching(child, child_leg, parent, parent_leg)

        child.open_leg_to_parent(child_leg, parent_id)
        parent.open_leg_to_child(parent_leg, child_id)
        self.nodes.update({child_id: child})

    def add_parent_to_root(self, root_leg, parent, parent_leg):
        """
        Adds the node parent as parent to the TTN's root node. The two
        are contracted. The root via root_leg and the parent via parent_leg.
        The root is updated to be the parent.
        """
        parent_id = parent.identifier
        self.assert_no_nodeid_dublication(parent_id)

        root = self.nodes[self.root_id]
        assert_legs_matching(root, root_leg, parent, parent_leg)

        root.open_leg_to_parent(root_leg, parent_id)
        parent.open_leg_to_child(parent_leg, self.root_id)
        self.nodes.update({parent_id: parent})
        self._root_id = parent_id

    def combine_nodes(self, node1_id, node2_id, new_tag=None, new_identifier=None):
        """
        Combines the two neighbouring nodes with the identifiers node1_id and
        node2_id. The the nodes' tensors are contracted along the connecting
        leg, other legs are distributed accordingly.
        If new_idenifier is None, the new identifier will be the identifiers
        of both nodes connected by "_". If that identifier is already in use,
        a random identifier is assigned. The tag is handled in the same way,
        except for the uniqueness
        """
        self.assert_id_in_tree(node1_id)
        self.assert_id_in_tree(node2_id)

        node1 = self.node[node1_id]
        node2 = self.node[node2_id]

        # one has to be the parent of the other
        if node1.is_parent_of(node2_id):
            parent = node1
            child = node2
            mode = "node1_parent"
        elif node2.is_parent_of(node1_id):
            parent = node2
            child = node1
            mode = "node2_parent"
        else:
            raise NoConnectionException(f"The tensors with identifiers {node1_id} and {node2_id} are not connected!")

        parent_id = parent.identifier
        child_id = child.identifier

        leg_parent_to_child = parent.children_legs[child_id]
        leg_child_to_parent = child.parent_leg[0]
        parent_tensor = parent.tensor
        child_tensor = child.tensor
        new_tensor = np.tensordot(parent_tensor, child_tensor, axes=(leg_parent_to_child,leg_child_to_parent))

        num_uncontracted_legs_parent = parent_tensor.ndim - 1

        # Not actually needed, but might help in testing.
        # parent_open_legs = [leg
        #                     for leg in parent.open_legs
        #                     if leg < leg_parent_to_child]
        # parent_open_legs.extend([leg - 1
        #                          for leg in parent.open_legs
        #                          if leg > leg_parent_to_child])
        # child_open_legs = [leg + num_uncontracted_legs_parent
        #                    for leg in child.open_legs
        #                    if leg < leg_child_to_parent]
        # child_open_legs.extend([leg + num_uncontracted_legs_parent -1
        #                         for leg in child.open_legs
        #                         if leg > leg_child_to_parent])
        # parent_open_legs.extend(child_open_legs)
        # new_open_legs = parent_open_legs

        parent_children_legs = {identifier: parent.children_legs[identifier]
                                for identifier in parent.children_legs
                                if parent.children_legs[identifier] < leg_parent_to_child}
        parent_children_legs.update({identifier: parent.children_legs[identifier] -1
                                     for identifier in parent.children_legs
                                     if parent.children_legs[identifier] < leg_parent_to_child})
        child_children_legs = {identifier: child.children_legs[identifier] + num_uncontracted_legs_parent
                                for identifier in child.children_legs
                                if child.children_legs[identifier] < leg_child_to_parent}
        child_children_legs.update({identifier: child.children_legs[identifier] + num_uncontracted_legs_parent -1
                                for identifier in child.children_legs
                                if child.children_legs[identifier] > leg_child_to_parent})
        parent_children_legs.update(child_children_legs)
        new_children_legs = parent_children_legs

        if new_identifier == None:
            new_identifier = node1_id + node2_id
            if not self.check_no_nodeid_dublication(new_identifier):
                new_identifier = str(uuid.uuid1())

        else:
            new_identifier = str(new_identifier)

        if new_tag == None:
            new_tag = node1.tag + node2.tag
        else:
            new_tag = str(new_tag)

        if parent.is_root():
            self.root = new_identifier
            new_parent_leg = []
        elif parent.parent_leg[1] < leg_parent_to_child:
            new_parent_leg = parent.parent_leg
        elif parent.parent_leg[1] > leg_parent_to_child:
            new_parent_leg = [parent.parent_leg[0], parent.parent_leg[1]]

        new_tensor_node = TensorNode(tensor=new_tensor, tag=new_tag, identifier=new_identifier)
        new_tensor_node.open_leg_to_parent(new_parent_leg[1], new_parent_leg[0])
        new_tensor_node.open_legs_to_children(new_children_legs.values(), new_children_legs.keys())

        del self.nodes[node1_id]
        del self.nodes[node2_id]

        self.nodes.update({new_tensor_node.identifier: new_tensor_node})


    def distance_to_node(self, center_node_id):
        """

        Parameters
        ----------
        center_node_id : str
            The identifier of the node to which the distance of all other
            nodes should be determined.

        Returns
        -------
        distance_dict : dictionary(str : int)
            A dictionary with the identifiers of the TNN's nodes as keys and
            their distance to center_node as values

        """
        distance_dict = {center_node_id: 0}
        self.distance_of_neighbours(ignore_node_id=None, distance=1, node_id=center_node_id, distance_dict=distance_dict)
        return distance_dict

    def distance_of_neighbours(self, ignore_node_id, distance, node_id, distance_dict):
        """

        Parameters
        ----------
        ignore_node_id : str
            Identifier of the node to be ignored for the recursion, i.e., the
            distance to it has already been established.
        distance : int
            The distance of the node with identifier node_id to the center_node.
        node_id : str
            The identifier of the node whose neighbours' distances are to be
            checked
        distance_dict : dictionary(str : int)
            A dictionary with the identifiers of the TNN's nodes as keys and
            their distance to center_node as values

        Returns
        -------
        None.

        """
        node = self.nodes[node_id]
        non_ignored_children_id = [child_id for child_id in node.children_legs.keys() if child_id != ignore_node_id]

        children_distance_to_center = {child_id: distance for child_id in non_ignored_children_id}
        distance_dict.update(children_distance_to_center)

        for child_id in children_distance_to_center.keys():
            self.distance_of_neighbours(ignore_node_id=node_id, distance=distance+1, node_id=child_id, distance_dict=distance_dict)

        if not node.is_root():
            parent_id = node.parent_leg[0]
            if not parent_id == ignore_node_id:
                distance_dict.update({parent_id: distance})
                self.distance_of_neighbours(ignore_node_id=node_id, distance=distance+1, node_id=parent_id, distance_dict=distance_dict)





