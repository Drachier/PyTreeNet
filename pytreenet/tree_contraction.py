import uuid
import numpy as np

from .tensornode import TensorNode
from .tnn_exceptions import NoConnectionException

def combine_nodes(tree_tensor_network, node1_id, node2_id, new_tag=None, new_identifier=None):
    """
    Combines the two neighbouring nodes with the identifiers node1_id and
    node2_id. The the nodes' tensors are contracted along the connecting
    leg, other legs are distributed accordingly.
    If new_idenifier is None, the new identifier will be the identifiers
    of both nodes connected by "_". If that identifier is already in use,
    a random identifier is assigned. The tag is handled in the same way,
    except for the uniqueness
    """
    tree_tensor_network.assert_id_in_tree(node1_id)
    tree_tensor_network.assert_id_in_tree(node2_id)

    node1 = tree_tensor_network.nodes[node1_id]
    node2 = tree_tensor_network.nodes[node2_id]

    new_identifier = construct_contracted_identifier(tree_tensor_network,
                                                     node1_id, node2_id,
                                                     new_identifier)
    new_tag = construct_new_tag(node1.tag, node2.tag, new_tag)

    tensor1 = node1.tensor
    tensor2 = node2.tensor

    num_uncontracted_legs_node1 = tensor1.ndim - 1

    # one has to be the parent of the other
    if node1.is_parent_of(node2_id):
        leg_node1_to_node2, leg_node2_to_node1 = find_connecting_legs(node1, node2)
        new_parent_leg = _find_total_parent_leg(tree_tensor_network, node1, 0, leg_node1_to_node2, new_identifier)
        _rewire_neighbours(tree_tensor_network, node1, node2, new_identifier)
    elif node2.is_parent_of(node1_id):
        leg_node2_to_node1, leg_node1_to_node2 = find_connecting_legs(node2, node1)
        new_parent_leg = _find_total_parent_leg(tree_tensor_network, node1, num_uncontracted_legs_node1, leg_node2_to_node1, new_identifier)
        _rewire_neighbours(tree_tensor_network, node2, node1, new_identifier)
    else:
        raise NoConnectionException(f"The tensors with identifiers {node1_id} and {node2_id} are not connected!")

    new_tensor = np.tensordot(tensor1, tensor2, axes=(leg_node1_to_node2, leg_node2_to_node1))
    new_children_legs = find_new_children_legs(node1, node2,
                                               leg_node1_to_node2, leg_node2_to_node1,
                                               num_uncontracted_legs_node1)

    new_tensor_node = TensorNode(tensor=new_tensor, tag=new_tag, identifier=new_identifier)
    new_tensor_node.open_leg_to_parent(new_parent_leg[1], new_parent_leg[0])
    new_tensor_node.open_legs_to_children(new_children_legs.values(), new_children_legs.keys())

    del tree_tensor_network.nodes[node1_id]
    del tree_tensor_network.nodes[node2_id]

    tree_tensor_network.nodes.update({new_tensor_node.identifier: new_tensor_node})

def construct_contracted_identifier(tree_tensor_network, node1_id, node2_id, new_identifier):
    if new_identifier == None:
        new_identifier = node1_id + "_contr_" + node2_id
        if not tree_tensor_network.check_no_nodeid_dublication(new_identifier):
            new_identifier = str(uuid.uuid1())
    else:
        new_identifier = str(new_identifier)

    return new_identifier

def construct_new_tag(node1_tag, node2_tag, new_tag):
    if new_tag == None:
        new_tag = node1_tag + "_contr_" + node2_tag
    else:
        new_tag = str(new_tag)

    return new_tag

def find_connecting_legs(parent, child):
    child_id = child.identifier

    leg_parent_to_child = parent.children_legs[child_id]
    leg_child_to_parent = child.parent_leg[0]
    return leg_parent_to_child, leg_child_to_parent

def _find_total_parent_leg(tree_tensor_network, parent, offset, contracted_leg, new_identifier):
    total_parent_id = parent.parent_leg[0]
    old_parent_leg = parent.parent_leg[1]

    if parent.is_root():
        tree_tensor_network.root = new_identifier
        new_parent_leg = []
    elif old_parent_leg < contracted_leg:
        new_parent_leg = [total_parent_id, old_parent_leg + offset]
    elif old_parent_leg > contracted_leg:
        new_parent_leg = [total_parent_id, old_parent_leg + offset -1]

    return new_parent_leg

def _rewire_neighbours(tree_tensor_network, parent, contracted_child, new_identifier):
    parent_id = parent.identifier
    contracted_child_id = contracted_child.identifier

    parent_children_ids = parent.children_legs.keys()
    for child_id in parent_children_ids:
        if child_id != contracted_child_id:
            tree_tensor_network.rewire_only_child(parent_id, child_id, new_identifier)

    if not parent.is_root():
        tree_tensor_network.rewire_only_parent(parent_id, new_identifier)

    child_children_ids = contracted_child.children_legs.keys()
    for child_id in child_children_ids:
        tree_tensor_network.rewire_only_child(contracted_child_id, child_id, new_identifier)


def find_new_children_legs(node1, node2, leg_node1_to_node2, leg_node2_to_node1, num_uncontracted_legs_node1):
    node1_children_legs = {identifier: node1.children_legs[identifier]
                            for identifier in node1.children_legs
                            if node1.children_legs[identifier] < leg_node1_to_node2}
    node1_children_legs.update({identifier: node1.children_legs[identifier] - 1
                                 for identifier in node1.children_legs
                                 if node1.children_legs[identifier] > leg_node1_to_node2})
    node2_children_legs = {identifier: node2.children_legs[identifier] + num_uncontracted_legs_node1
                            for identifier in node2.children_legs
                            if node2.children_legs[identifier] < leg_node2_to_node1}
    node2_children_legs.update({identifier: node2.children_legs[identifier] + num_uncontracted_legs_node1 -1
                            for identifier in node2.children_legs
                            if node2.children_legs[identifier] > leg_node2_to_node1})
    node1_children_legs.update(node2_children_legs)
    return node1_children_legs
