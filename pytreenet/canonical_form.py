from __future__ import annotations
from typing import Tuple

from copy import copy

from .leg_specification import LegSpecification
from .node import Node
from .tensor_util import SplitMode

def canonical_form(ttn: TreeTensorNetwork, orthogonality_center_id: str,
                   mode: SplitMode = SplitMode.REDUCED):
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
    mode: The mode to be used for the QR decomposition. For details refer to
     `tensor_util.tensor_qr_decomposition`.

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
            minimum_distance_neighbour_id = _find_smallest_distance_neighbour(node,
                                                                              distance_dict)
            q_legs, r_legs = _build_qr_leg_specs(node,
                                                 minimum_distance_neighbour_id)
            ttn.split_node_qr(node_id, q_legs, r_legs,
                              q_identifier=node_id,
                              r_identifier="R_tensor",
                              mode=mode)
            ttn.contract_nodes(minimum_distance_neighbour_id, "R_tensor",
                               new_identifier=minimum_distance_neighbour_id)
    ttn.orthogonality_center_id = orthogonality_center_id

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

def _build_qr_leg_specs(node: Node,
                        min_neighbour_id: str) -> Tuple[LegSpecification,LegSpecification]:
    """
    Construct the leg specifications required for the qr decompositions during
     canonicalisation.

    Args:
        node (Node): The node which is to be split.
        min_neighbour_id (str): The identifier of the neighbour of the node
         which is closest to the orthogonality center.

    Returns:
        Tuple[LegSpecification,LegSpecification]: 
         The leg specifications for the legs of the Q-tensor, i.e. what
          remains as the node, and the R-tensor, i.e. what will be absorbed
          into the node defined by `min_neighbour_id`.
    """
    q_legs = LegSpecification(None, copy(node.children), node.open_legs)
    if node.is_child_of(min_neighbour_id):
        r_legs = LegSpecification(min_neighbour_id, [], [])
    else:
        q_legs.parent_leg = node.parent
        q_legs.child_legs.remove(min_neighbour_id)
        r_legs = LegSpecification(None, [min_neighbour_id], [])
    return q_legs, r_legs
