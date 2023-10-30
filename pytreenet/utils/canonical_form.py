import numpy as np

from .tensor_util import tensor_qr_decomposition


def canonical_form(tree_tensor_network, orthogonality_center_id):
    """
    Brings the tree_tensor_network in canonical form with

    Parameters
    ----------
    tree_tensor_network : TreeTensorNetwork
        The TTN for which to find the canonical form
    orthogonality_center_id : str
        The id of the tensor node, which sould be the orthogonality center for
        the canonical form

    Returns
    -------
    None.

    """
    orthogonalize(tree_tensor_network, orthogonality_center_id, keep_shape=False)


def _find_smallest_distance_neighbour(node, distance_dict):
    """
    Finds identifier of the neighbour of node with the minimal distance in
    distance dict, i.e. minimum distance to the orthogonality center.

    Parameters
    ----------
    node : TensorNode
        TensorNode for which to search.
    distance_dict : dict
        Dictionary with the distance of every node to the orthogonality center.

    Returns
    -------
    minimum_distance_neighbour_id : str
        Identifier of the neighbour of node with minimum distance to in
        distance_dict.

    """
    neighbour_ids = node.neighbouring_nodes(with_legs=False)
    neighbour_distance_dict = {neighbour_id: distance_dict[neighbour_id]
                                   for neighbour_id in neighbour_ids}
    minimum_distance_neighbour_id = min(neighbour_distance_dict, key=neighbour_distance_dict.get)
    return minimum_distance_neighbour_id

def _find_smalles_distance_neighbour_index(node, minimum_distance_neighbour_id):
    """

    Parameters
    ----------
    node : TensorNode
        Node for which to find the leg.
    minimum_distance_neighbour_id : str
        Identifier of the neighbour of node with minimum distance in
        distance_dict

    Returns
    -------
     : str
     The index of leg attached to the neighbour with smallest distance in distance_dict.

    """
    neighbour_index = node.neighbouring_nodes()
    return neighbour_index[minimum_distance_neighbour_id]

def _correct_ordering_of_q_legs(node, minimum_distance_neighbour_leg):
    """
    Finds the correct ordering of the legs of the q-tensor after perfomring
    QR-decomposition on the tensor of node.

    """
    number_legs = node.tensor.ndim
    first_part = tuple(range(0,minimum_distance_neighbour_leg))
    last_part = tuple(range(minimum_distance_neighbour_leg,number_legs-1))
    reshape_order = first_part + (number_legs-1,) + last_part
    return reshape_order

def orthogonalize(tree_tensor_network, orthogonality_center_id, keep_shape=True):
    """
    Brings the tree_tensor_network in orthogonal form with

    Parameters
    ----------
    tree_tensor_network : TreeTensorNetwork
        The TTN for which to find the canonical form
    orthogonality_center_id : str
        The id of the tensor node, which sould be the orthogonality center 
    keep_shape : bool
        True: virtual bond dimensions are not changed. False: Creates_canonical form.

    Returns
    -------
    None.

    """
    distance_dict = tree_tensor_network.distance_to_node(orthogonality_center_id)

    maximum_distance = max(distance_dict.values())

    # Perform QR-decomposition on all TensorNodes but the orthogonality center
    for distance in reversed(range(1, maximum_distance+1)):
        node_id_with_distance = [node_id for node_id in distance_dict.keys()
                                 if distance_dict[node_id] == distance]

        for node_id in node_id_with_distance:
            node = tree_tensor_network.nodes[node_id]
            minimum_distance_neighbour_id = _find_smallest_distance_neighbour(node, distance_dict)
            minimum_distance_neighbour_index = _find_smalles_distance_neighbour_index(node, minimum_distance_neighbour_id)
            all_leg_indices = list(range(0,node.tensor.ndim))
            all_leg_indices.remove(minimum_distance_neighbour_index)

            q, r = tensor_qr_decomposition(node.tensor, all_leg_indices, [minimum_distance_neighbour_index], keep_shape=keep_shape)

            reshape_order = _correct_ordering_of_q_legs(node, minimum_distance_neighbour_index)
            node.tensor = np.transpose(q, axes=reshape_order)

            neighbour_tensor = tree_tensor_network.nodes[minimum_distance_neighbour_id]
            legs_to_neighbours_neighbours = neighbour_tensor.neighbouring_nodes()
            neighbour_index_to_contract = legs_to_neighbours_neighbours[node_id]
            neighbour_tensor.absorb_tensor(r, (1,), (neighbour_index_to_contract,))











