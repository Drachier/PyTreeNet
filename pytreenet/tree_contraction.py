from __future__ import annotations
import uuid

import numpy as np

from .node_contraction import contract_nodes, operator_expectation_value_on_node
from .ttn_exceptions import NoConnectionException
from .canonical_form import canonical_form
from .util import copy_object, sort_dictionary

def completely_contract_tree(ttn, to_copy=False):
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
    work_ttn = copy_object(ttn, deep=to_copy)

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
                                            axes=([0, leg2], [0, 1]))

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
