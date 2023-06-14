"""
Contains the functions to contract TensorNodes with one another as well as some
useful contractions.
"""
from __future__ import annotations
import numpy as np

from .node import Node, conjugate_node
from .ttn_exceptions import NoConnectionException
from .util import copy_object


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
            new_parent_leg = [total_parent_id, old_parent_leg + offset - 1]
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
    node2_children_legs.update({identifier: node2.children_legs[identifier] + num_uncontracted_legs_node1 - 1
                                for identifier in node2.children_legs
                                if node2.children_legs[identifier] > leg_node2_to_node1})
    node1_children_legs.update(node2_children_legs)
    return node1_children_legs


def contract_nodes(ttn: TreeTensorNetwork, node1: Node, node2: Node, new_tag=None, new_identifier=None):
    """
    Contracts the two Nodes node1 and node2 by contracting their tensors
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
    tensor1 = ttn.tensors[node1.identifier]
    tensor2 = ttn.tensors[node2.identifier]

    num_uncontracted_legs_node1 = tensor1.ndim - 1

    new_identifier = _construct_contracted_identifier(
        node1_id=node1_id, node2_id=node2_id, new_identifier=new_identifier)
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

    new_tensor_node = Node(tensor=new_tensor, tag=new_tag, identifier=new_identifier)
    if len(new_parent_leg) != 0:
        new_tensor_node.open_leg_to_parent(new_parent_leg[1], new_parent_leg[0])
    new_tensor_node.open_legs_to_children(new_children_legs.values(), new_children_legs.keys())

    return (new_tensor_node, new_tensor)


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

        neighbours = node2.neighbouring_nodes()
        leg_2_to_1 = neighbours[node1_id]

        return (leg_1_to_2, leg_2_to_1)
    else:
        raise NoConnectionException(f"Nodes with identifiers {node1_id} and {node2_id} are no neigbours.")


def _create_leg_dict(node: Node, tensor: ndarray, connecting_leg_index, offset=0, key_virtual=None, key_open=None):
    """
    Will return a dictionary with all legs, but the connecting_leg_index.
    WARNING: This function will cange the leg ordering in the node.

    Parameters
    ----------
    node : TensorNode
    connecting_leg_index : int
    offset: int
        A constant to add to every index
    key_virtual, key_open: string, string
        A custom key can be given to each entry, if none they default to
        node.identifier + "vitual"/"open"

    Returns
    -------
    leg_dict: dict
    A dictionary that contains two entries. Both have a key starting with the node`s identifier
    and end in virtual or open. The one with "virtual" key contains all virtual legs, apart from
    the connecting leg and the value of the "open" key contains all open legs.

    """
    node.order_legs(last_leg_index=connecting_leg_index)

    virtual_leg_indices = [node.children_legs[child_id] + offset
                           for child_id in node.children_legs]

    if not node.is_root():
        virtual_leg_indices.append(node.parent_leg[1] + offset)

    # We know the contracted leg has to be a virtual one
    # and it will be the highest index
    new_connecting_index = node.tensor.ndim - 1 + offset
    assert new_connecting_index in virtual_leg_indices
    virtual_leg_indices.remove(new_connecting_index)

    open_leg_indices = [open_leg_index + offset
                        for open_leg_index in node.open_legs]

    # Prepare keys
    if key_virtual == None:
        key_virtual = node.identifier + "virtual"
    if key_open == None:
        key_open = node.identifier + "open"

    dictionary = {key_virtual: virtual_leg_indices,
                  key_open: open_leg_indices}

    return dictionary


def contract_tensors_of_nodes(node1, tensor1, node2, tensor2):
    """
    Contracts the tensors of associated to two nodes and returns it
    to work mostly outside of the tree picture.
    WARNING: Using this function will change the node's leg ordering

    Parameters
    ----------
    node1 : TensorNode
    node2 : TensorNode     

    Returns
    -------
    contracted_tensor : ndarray
        The result of contracing the tensors of both nodes with leg order
        (remaining legs of tensor of node1, remaining legs of tensor of node2)
    leg_dictionary: dict
        A dictionar that contains information on which legs are open/virtual
        and belong to which of the two nodes.

    """

    leg_1_to_2, leg_2_to_1 = find_connecting_legs(node1, node2)

    dict_node1 = _create_leg_dict(node1, tensor1, leg_1_to_2)

    # In the contracted tensor the legs of the second tensor start after
    # the last leg of the first tensor.
    offset = node1.tensor.ndim - 1
    dict_node2 = _create_leg_dict(node2, tensor2, leg_2_to_1, offset=offset)

    leg_dictionary = dict()
    leg_dictionary.update(dict_node1)
    leg_dictionary.update(dict_node2)

    leg_1_to_2 = offset
    leg_2_to_1 = node2.tensor.ndim - 1

    contracted_tensor = np.tensordot(node1.tensor, node2.tensor,
                                     axes=([leg_1_to_2], [leg_2_to_1]))

    return contracted_tensor, leg_dictionary


def operator_expectation_value_on_node(node, operator):
    """
    This function evaluates the expectation value of the operator applied to
    the node, while tracing out the remaining legs.

    Parameters
    ----------
    node_id : string
        Currently assumes the node has a single open leg.
    operator : ndarray
        A matrix representing the operator to be evaluated.

    Returns
    -------
    exp_value: complex
        The resulting expectation value.

    """
    node = copy_object(node, deep=True)

    if len(node.open_legs) == 0:
        raise ValueError("A node with no open leg cannot have an operator applied.")
    # TODO: Dimensional checks

    node_conjugate = conjugate_node(node)

    node.absorb_tensor(operator, (1,), (node.open_legs[0],))

    all_axes = range(node.tensor.ndim)

    exp_value = complex(np.tensordot(node.tensor, node_conjugate.tensor,
                                     axes=(all_axes, all_axes)))

    return exp_value
