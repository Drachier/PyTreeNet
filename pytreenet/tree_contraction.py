from __future__ import annotations

from copy import copy

import numpy as np

from .node_contraction import operator_expectation_value_on_node
from .canonical_form import canonical_form
from .util import copy_object, sort_dictionary

def completely_contract_tree(ttn: TreeTensorNetwork, to_copy: bool=False) -> TreeTensorNetwork:
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
    work_ttn (TreeTensorNetwork): A ttn with a single node containing the contracted tensor.
    """
    work_ttn = copy_object(ttn, deep=to_copy)

    root_id = work_ttn.root_id
    _completely_contract_tree_rec(work_ttn, root_id)

    return work_ttn

def _completely_contract_tree_rec(work_ttn: TreeTensorNetwork, current_node_id: str):
    """
    Recursively runs through the tree contracting it from leaf to root.

    Args:
        work_ttn (TreeTensorNetwork): The TTN to be contracted
        current_node_id (str): The node into which we want to contract the subtree.
    """
    current_node = work_ttn.nodes[current_node_id]
    children = copy(current_node.children)
    for child_id in children:
        # Contracts the complete subtree into this child
        _completely_contract_tree_rec(work_ttn, child_id)
        work_ttn.contract_nodes(current_node_id, child_id, new_identifier=current_node_id)

def _contract_same_structure_nodes(node_id: str, ttn1: TreeTensorNetwork, ttn2: TreeTensorNetwork) -> np.ndarray:
    """
    Contracts the two nodes in the tensor networks that correspond to the same identifier.

    Args:
        node_id (str): The identifier giving the position in the TTNs.
        ttn1 (TreeTensorNetwork): One TTN to be contracted
        ttn2 (TreeTensorNetwork): Second TTN to be contracted

    Returns:
        (np.ndarray) : The contraction result
    """
    node1 = ttn1.nodes[node_id]
    if node1.is_leaf():
        open_legs = node1.open_legs
        resulting_tensor = np.tensordot(ttn1.tensors[node_id], ttn2.tensors[node_id],
                                        axes=(open_legs, open_legs))
        return resulting_tensor

    children = node1.children
    result_tensors = {}
    for child_id in children:
        # This tensor will have exactly two legs.
        # Leg 0 is contracted with node1 and leg 1 with node2.
        child_tensor = _contract_same_structure_nodes(child_id, ttn1, ttn2)
        result_tensors[child_id] = child_tensor

    open_legs = node1.open_legs
    for child_id in children:
        result_tensor = result_tensors[child_id]
        ttn1.absorb_tensor_into_neighbour_leg(node_id, child_id, result_tensor, 0)
    # Now the tensor of node1 contains all the children results
    n_legs = node1.nlegs()
    if node1.is_root():
        contracting_legs = list(range(n_legs))
    else:
        contracting_legs = list(range(1, n_legs))
    resulting_tensor = np.tensordot(ttn1.tensors[node_id], ttn2.tensors[node_id],
                                    axes=(contracting_legs, contracting_legs))
    return resulting_tensor

def contract_two_ttn(ttn1: TreeTensorNetwork, ttn2: TreeTensorNetwork) -> complex:
    """
    Contracts two TTN with the same structure.
    Assumes both TTN use the same identifiers for the nodes.

    Args:
        ttn1 (TreeTensorNetwork): One TTN to be contracted
        ttn2 (TreeTensorNetwork): Second TTN to be contracted

    Returns:
        complex: The contraction result.
    """
    return complex(_contract_same_structure_nodes(ttn1.root_id, ttn1, ttn2))

# TODO: Do the functions below really fit to the TTN class, or should they go into a TTNS class
def single_site_operator_expectation_value(ttn, node_id, operator):
    """
    Assuming ttn represents a quantum state, this function evaluates the 
    expectation value of the operator applied to the node with identifier 
    node_id.

    Parameters
    ----------
    ttn : TreeTensoNetwork
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
    canonical_form(ttn, node_id)
    node = ttn.nodes[node_id]

    if len(node.open_legs) != 1:
        raise NotImplementedError(
            f"Not implemented for nodes with more than one physical leg. Node with id {node_id} has more than one open leg.")

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

    if len(operator_dict) == 1:
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
