"""
This module is concerned with the canonical form of a TreeTensorNetwork.

The canonical form of a TreeTensorNetwork is a specific choice of the gauge
freedom causing all tensors apart from the orthogonality center to be a
equivalent to an isometry. This is achieved by performing a series of QR
decompositions on the tensors of the network.
"""
from __future__ import annotations
from typing import Tuple
from uuid import uuid1

from copy import copy

from .leg_specification import LegSpecification
from .node import Node
from ..util.tensor_splitting import SplitMode
from ..util.tensor_splitting import SVDParameters
from typing import TYPE_CHECKING
from copy import deepcopy

if TYPE_CHECKING:
    from ..ttns import TreeTensorNetwork


def _find_smallest_distance_neighbour(node: Node,
                                      distance_dict: dict[str, int]) -> str:
    """
    Finds the neighbour of a node with the smallest distance to the center node.
    
    Args:
        node (Node): The node for which to search the neighbours.
        distance_dict (dict[str,int]): A dictionary with the distance of every
            node to the orthogonality center.

    Returns:
        str: The identifier of the neighbour of the node with the smallest
            distance to the orthogonality center.
    """
    neighbour_ids = node.neighbouring_nodes()
    neighbour_distance_dict = {neighbour_id: distance_dict[neighbour_id]
                               for neighbour_id in neighbour_ids}
    minimum_distance_neighbour_id = min(neighbour_distance_dict,
                                        key=neighbour_distance_dict.get)
    return minimum_distance_neighbour_id


def _build_leg_specs(node: Node,
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


def split_qr_contract_r_to_neighbour(ttn: TreeTensorNetwork,
                                     node_id: str,
                                     neighbour_id: str,
                                     mode: SplitMode = SplitMode.REDUCED):
    """
    Takes a node an splits of the virtual leg to a neighbours via QR
     decomposition. The resulting R tensor is contracted with the neighbour.::

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
    children_dict = {node_id      : deepcopy(ttn.nodes[node_id].children)}
    children_dict[neighbour_id] = deepcopy(ttn.nodes[neighbour_id].children)

    node = ttn.nodes[node_id]
    q_legs, r_legs = _build_leg_specs(node, neighbour_id)
    r_tensor_id = str(uuid1()) # Avoid identifier duplication
    ttn.split_node_qr(node_id, q_legs, r_legs,
                        q_identifier=node_id,
                        r_identifier=r_tensor_id,
                        mode=mode)
    ttn.contract_nodes(neighbour_id, r_tensor_id,
                        new_identifier=neighbour_id)

    ttn.update_children_and_leg_permutation(node_id, children_dict[node_id])
    ttn.update_children_and_leg_permutation(neighbour_id, children_dict[neighbour_id]) 

def split_svd_contract_sv_to_neighbour(ttn: TreeTensorNetwork,
                                     node_id: str,
                                     neighbour_id: str,
                                     svd_params: SVDParameters):

    children_dict = {node_id      : deepcopy(ttn.nodes[node_id].children)}
    children_dict[neighbour_id] = deepcopy(ttn.nodes[neighbour_id].children)

    node = ttn.nodes[node_id]
    u_legs, v_legs = _build_leg_specs(node, neighbour_id)
    r_tensor_id = str(uuid1()) # Avoid identifier duplication
    ttn.split_node_svd(node_id ,
                        u_legs = u_legs,
                        v_legs = v_legs,
                       u_identifier = node_id, 
                       v_identifier = r_tensor_id,
                       svd_params = svd_params)
    ttn.contract_nodes(neighbour_id, r_tensor_id,
                        new_identifier=neighbour_id)

    ttn.update_children_and_leg_permutation(node_id, children_dict[node_id])
    ttn.update_children_and_leg_permutation(neighbour_id, children_dict[neighbour_id]) 

def canonical_form(ttn: TreeTensorNetwork,
                   orthogonality_center_id: str,
                   split_function = split_qr_contract_r_to_neighbour,
                   nodes_to_process: list = None,
                   **kwargs):
    """
    Modifies a TreeTensorNetwork into canonical form.

    Args:
        ttn (TreeTensorNetwork): The TTN for which to be transformed into
            canonical form.
        orthogonality_center_id (str): The identifier of the tensor node which
            is the orthogonality center for the canonical form.
        split_function: The function to use for splitting nodes (QR or SVD).
            Defaults to split_qr_contract_r_to_neighbour.
        nodes_to_process: Optional list of node IDs to normalize. If provided,
            only these nodes will be normalized.
        **kwargs: Additional keyword arguments that are passed to the split function.
            Common parameters include:
            - mode: The mode to use for QR decomposition (when using split_qr_contract_r_to_neighbour).
            - svd_params: SVD parameters for SVD-based decomposition (when using split_svd_contract_sv_to_neighbour).
    """
    # Get full distance dictionary for all nodes
    full_distance_dict = ttn.distance_to_node(orthogonality_center_id)
    
    # Determine which nodes to process
    nodes_to_process = nodes_to_process if nodes_to_process is not None else full_distance_dict.keys()
    
    # Get maximum distance only for nodes we want to process
    process_distances = [full_distance_dict[node_id] for node_id in nodes_to_process]
    maximum_distance = max(process_distances) if process_distances else 0
    
    # Perform decomposition on selected nodes
    for distance in reversed(range(1, maximum_distance + 1)):
        # Only process nodes that are in our explicit list (or all nodes if no list provided)
        node_id_with_distance = [node_id for node_id in nodes_to_process
                               if full_distance_dict[node_id] == distance]
        
        for node_id in node_id_with_distance:
            node = ttn.nodes[node_id]
            minimum_distance_neighbour_id = _find_smallest_distance_neighbour(node,
                                                                           full_distance_dict)
            
            split_function(ttn, node_id, minimum_distance_neighbour_id, *kwargs)

    ttn.orthogonality_center_id = orthogonality_center_id

