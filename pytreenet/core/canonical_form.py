from __future__ import annotations
from typing import Tuple
from uuid import uuid1

from copy import copy
import numpy as np
from .leg_specification import LegSpecification
from .node import Node
from ..util.tensor_util import SplitMode
from ..util.tensor_util import tensor_qr_decomposition
import pytreenet as ptn

def canonical_form(ttn: ptn.TreeTensorNetwork, orthogonality_center_id: str,
                   **truncation_param):
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
            split_qr_contract_r_to_neighbour(ttn,
                                             node_id,
                                             minimum_distance_neighbour_id,
                                             **truncation_param)
    ttn.orthogonality_center_id = orthogonality_center_id

def complete_canonical_form(ttn: ptn.TreeTensorNetwork,
                            **truncation_param):
    """
    Modifies the TreeTensorNetwork (ttn) into complete canonical form. 
    first transforms the TTN into canonical form with the center of orthogonality at root site.
    Then, normilized root tensor.
    So the final tree tensor is left-normalized with respect to the root site.

    Args:
    ttn : TreeTensorNetwork
        The TTN for which to find the canonical form
    max_bond_dim: int, optional
        The maximum bond dimension of the reduced density matrices.
        Default is np.inf.
    rel_tol: float, optional
        The relative tolerance for the truncation of the singular values.
        Default is -np.inf.
    total_tol: float, optional
        The total tolerance for the truncation of the singular values.
        Default is -np.inf.

    Returns:
    norm : float
        The norm of the TTN after the canonical form is found.

    """
    orthogonality_center_id = ttn.root_id
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
            split_qr_contract_r_to_neighbour(ttn,
                                             node_id,
                                             minimum_distance_neighbour_id,
                                             **truncation_param)
    ttn.orthogonality_center_id = orthogonality_center_id
    q_legs = tuple(range(ttn.nodes[orthogonality_center_id].nchildren()))
    r_legs = (ttn.tensors[orthogonality_center_id].ndim-1,)
    ttn.tensors[orthogonality_center_id] , R = tensor_qr_decomposition(ttn.tensors[orthogonality_center_id], q_legs, r_legs)
    ttn.tensors[orthogonality_center_id] = ttn.tensors[orthogonality_center_id] / np.sqrt(ptn.contract_two_ttns(ttn,ttn.conjugate()))
    norm = ptn.compute_transfer_tensor(R, (0,1))
    return norm

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

def split_qr_contract_r_to_neighbour(ttn: ptn.TreeTensorNetwork,
                                     node_id: str,
                                     neighbour_id: str,
                                     **truncation_param):
    """
    Takes a node an splits of the virtual leg to a neighbours via QR
     decomposition. The resulting R tensor is contracted with the neighbour.

         __|__      __|__        __|__      __      __|__
      __|  N1 |____|  N2 | ---> | N1' |____|__|____|  N2 |
        |_____|    |_____|      |_____|            |_____|

                __|__      __|__ 
      --->   __| N1' |____| N2' |
               |_____|    |_____|

    Args:
        ttn (TreeTensorNetwork): The tree tensor network in which to perform
         this action.
        node_id (str): The identifier of the node to be split.
        neighbour_id (str): The identifier of the neigbour to which to split.
        mode: The mode to be used for the QR decomposition. For details refer to
        `tensor_util.tensor_qr_decomposition`.
    """
    node = ttn.nodes[node_id]
    u_legs, v_legs = _build_qr_leg_specs(node, neighbour_id)
    r_tensor_id = str(uuid1()) # Avoid identifier duplication
    ttn.split_node_svd(node_id , u_legs, v_legs,
                       u_identifier = node_id, v_identifier = r_tensor_id,
                      **truncation_param)
    ttn.contract_nodes(neighbour_id, r_tensor_id,
                        new_identifier=neighbour_id)        
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
    if node.is_root():
        q_legs.is_root = True
    return q_legs, r_legs
