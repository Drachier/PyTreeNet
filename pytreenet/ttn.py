import copy

import numpy as np

from warnings import warn

from .tensornode import assert_legs_matching

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
        The root's identifier.
        """
        return self._root_id

    @root_id.setter
    def root_id(self, new_root_id):
        """
        Sets a new root_id
        """
        self._root_id = str(new_root_id)

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

    def nearest_neighbours(self):
        """
        Finds all nearest neighbouring nodes in a tree.
        We basically find all parent-child pairs.

        Returns
        -------
        nn: list of tuples of strings.
            A list containing tuples that contain the two identifiers of
            nearest neighbour pairs of nodes.
        """
        nn = []
        
        for node_id in self.nodes:
            current_node = self.nodes[node_id]
            for child_id in current_node.children_legs:
                nn.append((node_id, child_id))
                
        return nn
    
    def conjugate(self):
        """
        Returns a new TTN that is a conjugated version of the current TTN

        Returns
        -------
        ttn_conj:
            A conjugated copy of the current TTN.
        
        """
        
        ttn_conj = TreeTensorNetwork(original_tree=self, deep=True)
        
        for node_id in ttn_conj.nodes:
            
            node = ttn_conj.nodes[node_id]
            node.tensor = np.conj(node.tensor)
            
        return ttn_conj

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

    # TODO implement similar functions in node class.
    def rewire_only_child(self, parent_id, child_id, new_identifier):
        """
        For the node with identifier child_id the parent_leg is rewired from parent
        to a node with identifier new_identifier.

        Parameters
        ----------
        parent_id : str
            Identifier of the parent node for which one child is rewired to a new parent.
        child_id : str
            Identifier of the child which is to be rewired.
        new_identifier : str
            Identifier of the node to be rewired to.

        Returns
        -------
        None.

        """
        parent = self.nodes[parent_id]
        child = self.nodes[child_id]
        assert child_id in parent.children_legs, f"The node with identifier {child_id} is not a child of the node with identifier {parent_id}."
        assert child.parent_leg[0] == parent_id, f"The node with identifier {parent_id} is not the parent of the node with identifier {child_id}."
        child.parent_leg[0] = new_identifier

    def rewire_only_parent(self, child_id, new_identifier):
        """
        For the parent of the node child the leg connected to child is rewired to the
        tensor node with identifier new_identifier.

        Parameters
        ----------
        child_id : str
            Identifier of the node whose parent is to have one leg rewired.
        new_identifier : str
            Identifier of the tensor the parent is rewired to.

        Returns
        -------
        None.

        """
        child = self.nodes[child_id]
        if child.is_root():
            warn(f"The node with identifier {child_id} is a tree's root, so its parent cannot be rewired.")
        else:
            parent_id = child.parent_leg[0]
            parent = self.nodes[parent_id]
            leg_to_child_tensor = {new_identifier: parent.children_legs[child_id]}
            del parent.children_legs[child_id]
            parent.children_legs.update(leg_to_child_tensor)