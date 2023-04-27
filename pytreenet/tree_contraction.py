import uuid

import numpy as np
from copy import deepcopy

from .node_contraction import contract_nodes, operator_expectation_value_on_node
from .ttn_exceptions import NoConnectionException
from .canonical_form import canonical_form, orthogonalize
from .util import copy_object, sort_dictionary

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
    (WARNING: Can get very costly very fast. Only use for debugging.)

    Parameters
    ----------
    tree_tensor_network : TreeTensorNetwork
        The TTN to be contracted.
    to_copy: bool
        Wether or not the contraction should be perfomed on a deep copy.
        Default is False.

    Returns
    -------
    In case copy is True a deep copy of the completely contracted TTN is
    returned.

    """
    work_ttn = copy_object(tree_tensor_network, deep=to_copy)

    distance_to_root = work_ttn.distance_to_node(work_ttn.root_id)

    for distance in range(1, max(distance_to_root.values())+1):
            node_id_with_distance = [node_id for node_id in distance_to_root
                                     if distance_to_root[node_id] == distance]
            for node_id in node_id_with_distance:
                contract_nodes_in_tree(work_ttn, work_ttn.root_id, node_id)
                
    if to_copy:
        return work_ttn

# TODO: Check functions below

def _contract_same_structure_nodes(node1, node2, ttn1, ttn2):
    """
    Contracts two nodes with the same structure.

    Parameters
    ----------
    node1 : TensorNode
    node2 : TensorNode
    ttn1 : TreeTensorNetwork
        TTN containing node1
    ttn2 : TreeTensorNetwork
        TTN containing node2

    Returns
    -------
    resulting_tensor : ndarray
        resulting tensor

    """
    
    if node1.is_leaf():
        open_legs = node1.open_legs
        
        resulting_tensor = np.tensordot(node1.tensor, node2.tensor,
                                        axes=(open_legs, open_legs))
        
        return resulting_tensor
    
    else:
        children_legs = node1.children_legs
        
        result_tensors = dict()
        for child_id in children_legs:
            
            child1 = ttn1.nodes[child_id]
            child2 = ttn2.nodes[child_id]
            
            # This tensor will have exactly two legs.
            # Leg 0 is contracted with node1 and leg 1 with node2.
            child_tensor = _contract_same_structure_nodes(child1, child2, 
                                                          ttn1, ttn2)
            
            result_tensors[child_id] = child_tensor
        
        # Make children the first legs
        node1.order_legs()
        node2.order_legs()
        
        # To call children in increasing order of leg index
        sorted_children = sort_dictionary(node1.children_legs)
        
        open_legs = node1.open_legs
        
        resulting_tensor = np.tensordot(node1.tensor, node2.tensor,
                                        axes=(open_legs, open_legs))
        
        for child_id in sorted_children:
            child_result = result_tensors[child_id]
            
            leg2 = int(resulting_tensor.ndim / 2)

            resulting_tensor = np.tensordot(resulting_tensor, child_result,
                                         axes=([0,leg2], [0,1]))
            
        return resulting_tensor

def contract_two_ttn(ttn1, ttn2):
    """
    Contracts two TTN with the same structure. Assumes both TTN use the same
    identifiers for the nodes.
    
    Parameters
    ----------
    ttn1 : TreeTensorNetwork
    ttn2 : TreeTensorNetwork

    Returns
    -------
    result_tensor: ndarray
        The contraction result.
        
    """
    root_id = ttn1.root_id
    
    root1 = ttn1.nodes[root_id]
    root2 = ttn2.nodes[root_id]
    
    result_tensor = _contract_same_structure_nodes(root1, root2, ttn1, ttn2)
    
    return result_tensor
    
def single_site_operator_expectation_value(ttn_in, node_id, operator):
    """
    Assuming ttn represents a quantum state, this function evaluates the 
    expectation value of the operator applied to the node with identifier 
    node_id.

    Parameters
    ----------
    ttn_in : TreeTensoNetwork
        A TTN representing a quantum state.
    node_id : string
        Identifier of a node in ttn.
        Currently assumes the node has a single open leg..
    operator : ndarray
        A matrix representing the operator to be evaluated.

    Returns
    -------
    exp_value: complex
        The resulting expectation value.

    """
    # Canonical form makes the evaluation very simple.
    # TODO DISCUSS THIS WITH RICHARD
    ttn = deepcopy(ttn_in)
    orthogonalize(ttn, node_id) 
    node = ttn[node_id]
    
    if len(node.open_legs) != 1:
        raise NotImplementedError(f"Not implemented for nodes with more than one physical leg. Node with id {node_id} has more than one open leg.")
    
    # Make use of the single-site nature
    exp_value = operator_expectation_value_on_node(node, operator)
    
    return exp_value

def operator_expectation_value(ttn, operator_dict):
    """
    Assuming ttn represents a quantum state, this function evaluates the 
    expectation value of the operator.

    Parameters
    ----------
    ttn : TreeTensorNetwork
        A TTN representing a quantum state. Currently assumes each node has a
        single open leg.
    operator_dict : dict
        A dictionary representing an operator applied to a quantum state.
        The keys are node identifiers to which the value, a matrix, is applied.

    Returns
    -------
    exp_value: complex
        The resulting expectation value.

    """
    
    if len(operator_dict )==1:
        node_id = list(operator_dict.keys())[0]
        operator = operator_dict[node_id]
        
        # Single-site is special due to canonical forms
        return ttn.single_site_operator_expectation_value(node_id, operator)
    else:
        ttn_copy = copy_object(ttn)
        
        ttn_conj = ttn_copy.conjugate()

        for node_id in operator_dict:

            node = ttn_copy.nodes[node_id]
            operator = operator_dict[node_id]
            
            node.absorb_tensor(operator, (1, ), (node.open_legs[0], ))
            
            exp_value = ttn_copy.contract_two_ttn(ttn_conj).flatten()
            
            assert len(exp_value) == 1
            
        return exp_value[0]

def scalar_product(ttn):
    """
    Computes the scalar product for a state_like TTN, i.e. one where the open
    legs represent a quantum state.

    Parameters
    ----------
    ttn : TreeTensorNetwork
        A TTN representing a quantum state. Currently assumes each node has a
        single open leg.

    Returns
    -------
    sc_prod: complex
        The resulting scalar product.

    """

    ttn_copy = copy_object(ttn)
        
    ttn_conj = ttn_copy.conjugate()

    sc_prod = ttn_copy.contract_two_ttn(ttn_conj).flatten()
    
    assert len(sc_prod) == 1
    
    return sc_prod[0]
        