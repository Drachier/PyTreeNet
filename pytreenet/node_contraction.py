"""
Contains the functions to contract TensorNodes with one another as well as some
useful contractions.
"""
import numpy as np

from .tensornode import TensorNode
from .ttn_exceptions import NoConnectionException

def _construct_contracted_identifier(node1_id, node2_id, new_identifier=None):
    if new_identifier == None:
        new_identifier = node1_id + "_contr_" + node2_id
    else:
        new_identifier = str(new_identifier)

    return new_identifier

def _construct_contracted_tag(node1_tag, node2_tag, new_tag):
    if new_tag == None:
        new_tag = node1_tag + "_contr_" + node2_tag
    else:
        new_tag = str(new_tag)

    return new_tag

def _find_connecting_legs_parent_child(parent, child):
    parent_id = parent.identifier
    assert parent_id in child.parent_leg
    child_id = child.identifier

    leg_parent_to_child = parent.children_legs[child_id]
    leg_child_to_parent = child.parent_leg[1]
    return leg_parent_to_child, leg_child_to_parent

def _find_total_parent_leg(parent, offset, contracted_leg):
    if parent.is_root():
        new_parent_leg = []
    else:
        total_parent_id = parent.parent_leg[0]
        old_parent_leg = parent.parent_leg[1]
        if old_parent_leg < contracted_leg:
            new_parent_leg = [total_parent_id, old_parent_leg + offset]
        elif old_parent_leg > contracted_leg:
            new_parent_leg = [total_parent_id, old_parent_leg + offset -1]
    return new_parent_leg

def _find_new_children_legs(node1, node2, leg_node1_to_node2, leg_node2_to_node1, num_uncontracted_legs_node1):
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

def contract_nodes(node1, node2, new_tag=None, new_identifier=None):
    """
    Contracts the two TensorNodes node1 and node2 by contracting their tensors
    along the leg connecting the nodes. The result will be a new TensorNode
    with the new tag new_tag and new identifier new_identifier. If either is
    None None the resulting string will be the concatination of the nodes'
    property with "_contr_" in-between.
    The resulting TensorNode will have the leg-ordering
    (legs of node1 without contracted leg) - (legs of node2 without contracted leg)

    Parameters
    ----------
    node1 : TensorNode
        First node to be contracted.
    node2 : TensorNode
        Second node to be contracted.
    new_tag : str, optional
        Tag given to new TensorNode. The default is None.
    new_identifier : str, optional
        Identifier given to the new TensorNode. The default is None.

    Returns
    -------
    new_tensor_node : TensorNode
        The TensorNode resulting from the contraction.

    """
    node1_id = node1.identifier
    node2_id = node2.identifier
    tensor1 = node1.tensor
    tensor2 = node2.tensor

    num_uncontracted_legs_node1 = tensor1.ndim - 1

    new_identifier = _construct_contracted_identifier(node1_id=node1_id, node2_id=node2_id, new_identifier=new_identifier)
    new_tag = _construct_contracted_tag(node1.tag, node2.tag, new_tag)

    # one has to be the parent of the other
    if node1.is_parent_of(node2_id):
        leg_node1_to_node2, leg_node2_to_node1 = _find_connecting_legs_parent_child(node1, node2)
        new_parent_leg = _find_total_parent_leg(node1, 0, leg_node1_to_node2)
    elif node2.is_parent_of(node1_id):
        leg_node2_to_node1, leg_node1_to_node2 = _find_connecting_legs_parent_child(node2, node1)
        new_parent_leg = _find_total_parent_leg(node2, num_uncontracted_legs_node1, leg_node2_to_node1)
    else:
        raise NoConnectionException(f"The nodes with identifiers {node1_id} and {node2_id} are not connected!")

    new_tensor = np.tensordot(tensor1, tensor2,
                              axes=(leg_node1_to_node2, leg_node2_to_node1))
    new_children_legs = _find_new_children_legs(node1, node2,
                                               leg_node1_to_node2, leg_node2_to_node1,
                                               num_uncontracted_legs_node1)

    new_tensor_node = TensorNode(tensor=new_tensor, tag=new_tag, identifier=new_identifier)
    if len(new_parent_leg) != 0:
        new_tensor_node.open_leg_to_parent(new_parent_leg[1], new_parent_leg[0])
    new_tensor_node.open_legs_to_children(new_children_legs.values(), new_children_legs.keys())

    return new_tensor_node

def determine_parentage(node1, node2):
    """
    Returns: (parent_node, child_node)
    """
    
    node1_id = node1.identifier
    node2_id = node2.identifier
    
    if node1.is_parent_of(node2_id):
        return (node1, node2)
    elif node2.is_parent_of(node1_id):
        return (node2, node1)
    else:
        raise NoConnectionException(f"Nodes with identifiers {node1_id} and {node2_id} are no neigbours.")

def find_connecting_legs(node1, node2):
    """
    
    Parameters
    ----------
    node1 : TensorNode
    node2 : TensorNode

    Raises
    ------
    NoConnectionException
        If the two nodes not directly connected.

    Returns
    -------
    leg_1_to_2 : int
        Leg of node1 that is connected to node2.
    leg_2_to_1 : int
        Leg of node2 that is connected to node1.

    """
    
    neighbours = node1.neighbouring_nodes()
    node1_id = node1.identifier
    node2_id = node2.identifier    
    
    if node2_id in neighbours:
        leg_1_to_2 = neighbours[node2_id]
        
        neighbours = node2.neighbouring_nodes
        leg_2_to_1 = neighbours[node2_id]
        
        return (leg_1_to_2, leg_2_to_1) 
    else:
        raise NoConnectionException(f"Nodes with identifiers {node1_id} and {node2_id} are no neigbours.")
        
def contract_tensors_of_nodes(node1, node2):
    """
    Contracts the tensors of associated to two nodes.

    Parameters
    ----------
    node1 : TensorNode
    node2 : TensorNode

    Returns
    -------
    contracted_tensor : ndarray
        The result of contracing the tensors of both nodes with leg order
        (remaining legs of tensor of node1, remaining legs of tensor of node2)

    """
    tensor1 = node1.tensor
    tensor2 = node2.tensor
    leg_1_to_2, leg_2_to_1 = find_connecting_legs(node1, node2)
    
    contracted_tensor = np.tensordot(tensor1, tensor2,
                                     axe=([leg_1_to_2],[leg_2_to_1]))
    
    return contracted_tensor