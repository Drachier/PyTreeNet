from __future__ import annotations

from copy import copy


def canonical_form(ttn: TreeTensorNetwork, orthogonality_center_id: str):
    """
    Modifies the TreeTensorNetwork (ttn) into canonical form with the center of orthogonality at
    node with node_id `orthogonality_center_id`.

    Parameters
    ----------
    tree_tensor_network : TreeTensorNetwork
        The TTN for which to find the canonical form
    orthogonality_center_id : str
        The id of the tensor node which is the orthogonality center for
        the canonical form

    Returns
    -------
    None.

    """
    distance_dict = ttn.distance_to_node(
        orthogonality_center_id)

    maximum_distance = max(distance_dict.values())

    # Perform QR-decomposition on all TensorNodes but the orthogonality center
    for distance in reversed(range(1, maximum_distance+1)):
        # Perform QR on nodes furthest away first.
        node_id_with_distance = [node_id for node_id in distance_dict.keys()
                                 if distance_dict[node_id] == distance]

        for node_id in node_id_with_distance:
            node = ttn.nodes[node_id]
            minimum_distance_neighbour_id = _find_smallest_distance_neighbour(
                node, distance_dict)

            min_neighbour_node = ttn.nodes[minimum_distance_neighbour_id]

            q_legs = {"parent_leg": None,
                      "child_legs": copy(node.children),
                      "open_legs": node.open_legs}
            if node.is_child_of(minimum_distance_neighbour_id):
                r_legs = {"parent_leg": minimum_distance_neighbour_id,
                          "child_legs": [],
                          "open_legs": []}
            else:
                q_legs["parent_leg"] = node.parent
                q_legs["child_legs"].remove(minimum_distance_neighbour_id)
                r_legs = {"parent_leg": None,
                          "child_legs": [minimum_distance_neighbour_id],
                          "open_legs": []}

            ttn.split_node_qr(node_id, q_legs, r_legs,
                              q_identifier=node_id, r_identifier="R_tensor")

            if node.is_child_of(minimum_distance_neighbour_id):
                leg_index = min_neighbour_node.get_child_leg(node_id) - min_neighbour_node.nparents()
                min_neighbour_node.children[leg_index] = "R_tensor"
            else:
                min_neighbour_node.remove_parent()
                min_neighbour_node.add_parent("R_tensor")

            ttn.contract_nodes(minimum_distance_neighbour_id, "R_tensor",
                               new_identifier=minimum_distance_neighbour_id)


def _find_smallest_distance_neighbour(node: Node, distance_dict: dict[str, int]) -> str:
    """
    Finds identifier of the neighbour of node with the minimal distance in
    distance dict, i.e. minimum distance to the orthogonality center.

    Parameters
    ----------
    node : Node
        Node for which to search.
    distance_dict : dict
        Dictionary with the distance of every node to the orthogonality center.

    Returns
    -------
    minimum_distance_neighbour_id : str
        Identifier of the neighbour of node with minimum distance to in
        distance_dict.

    """
    neighbour_ids = node.neighbouring_nodes()
    neighbour_distance_dict = {neighbour_id: distance_dict[neighbour_id]
                               for neighbour_id in neighbour_ids}
    minimum_distance_neighbour_id = min(neighbour_distance_dict,
                                        key=neighbour_distance_dict.get)
    return minimum_distance_neighbour_id
