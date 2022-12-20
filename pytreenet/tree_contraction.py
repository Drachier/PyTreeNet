import uuid

from .ttn import TreeTensorNetwork
from .node_contraction import contract_nodes
from .ttn_exceptions import NoConnectionException

def check_contracted_identifier(tree_tensor_network, new_identifier):
    if not tree_tensor_network.check_no_nodeid_dublication(new_identifier):
        new_identifier = str(uuid.uuid1())
    return new_identifier
    
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

def contract_nodes_in_tree(tree_tensor_network, node1_id, node2_id, new_tag=None, new_identifier=None):
    """
    Combines the two neighbouring nodes with the identifiers node1_id and
    node2_id. The the nodes' tensors are contracted along the connecting
    leg, other legs are distributed accordingly.
    If new_identifier is None, the new identifier will be the identifiers
    of both nodes connected by "_". If that identifier is already in use,
    a random identifier is assigned. The tag is handled in the same way,
    except for the uniqueness.
    Neighbours of the two TensorNodes are also rewired appropriately to the new
    tensor.
    New leg order:
    (legs of node1 without contr_leg) - (legs of node2 without contracted leg)
    """
    tree_tensor_network.assert_id_in_tree(node1_id)
    tree_tensor_network.assert_id_in_tree(node2_id)

    node1 = tree_tensor_network.nodes[node1_id]
    node2 = tree_tensor_network.nodes[node2_id]
    
    new_tensor_node = contract_nodes(node1, node2, new_tag=new_tag, new_identifier=new_identifier)
    new_identifier = new_tensor_node.identifier
    new_tensor_node.identifier = check_contracted_identifier(tree_tensor_network, new_identifier=new_identifier)
    tree_tensor_network.root_id = new_identifier
    
    # one has to be the parent of the other
    if node1.is_parent_of(node2_id):
        _rewire_neighbours(tree_tensor_network, node1, node2, new_identifier)
    elif node2.is_parent_of(node1_id):
        _rewire_neighbours(tree_tensor_network, node2, node1, new_identifier)
    else:
        raise NoConnectionException(f"Nodes with identifiers {node1_id} and {node2_id} are no neigbours.")
        
    del tree_tensor_network.nodes[node1_id]
    del tree_tensor_network.nodes[node2_id]    
    
    tree_tensor_network.nodes.update({new_tensor_node.identifier: new_tensor_node})

def completely_contract_tree(tree_tensor_network, to_copy=False):
    """
    Completely contracts the given tree_tensor_network by combining all nodes.
    (Naive implementation. Only use for debugging.)

    Parameters
    ----------
    tree_tensor_network : TreeTensorNetwork
        The TTN to be contracted.
    to_copy: bool
        Wether or not the contraction should be perfomed on a deep copy.

    Returns
    -------
    In case copy is True a deep copy of the completely contracted TTN is returned.

    """
    if to_copy:
        work_ttn = TreeTensorNetwork(original_tree=tree_tensor_network, deep=True)
    else:
        work_ttn = tree_tensor_network

    distance_to_root = work_ttn.distance_to_node(work_ttn.root_id)

    for distance in range(1, max(distance_to_root.values())+1):
            node_id_with_distance = [node_id for node_id in distance_to_root
                                     if distance_to_root[node_id] == distance]
            for node_id in node_id_with_distance:
                contract_nodes_in_tree(work_ttn, work_ttn.root_id, node_id)
                
    if to_copy:
        return work_ttn
