"""
This module contains direct addition operations for TTN.
"""
from __future__ import annotations
from copy import deepcopy

import numpy as np

from ..ttn import TreeTensorNetwork
from ..node import Node, relative_leg_permutation
from ..truncation.truncating import (TruncationMethod,
                                     truncate_ttns)

def direct_addition_and_truncation(ttns: list[TreeTensorNetwork],
                                   truncation_methods: TruncationMethod,
                                   *args,
                                   **kwargs
                                   ) -> TreeTensorNetwork:
    """
    Perform direct addition of multiple Tree Tensor Networks (TTNs)
    followed by truncation.

    Args:
        ttns (list[TreeTensorNetwork]): The list of TTNs to add.
        truncation_methods (TruncationMethod): The truncation method to use.
        *args: Additional positional arguments for the truncation function.
        **kwargs: Additional keyword arguments for the truncation function.

    Returns:
        TreeTensorNetwork: The resulting TTN after addition and truncation.
    """
    added_ttn = direct_addition(ttns)
    truncated_ttn = truncate_ttns(added_ttn,
                                  truncation_methods,
                                  *args,
                                  **kwargs)
    return truncated_ttn

def direct_addition(ttns: list[TreeTensorNetwork],
                    ) -> TreeTensorNetwork:
    """
    Perform direct addition of multiple Tree Tensor Networks (TTNs).
    """
    if not ttns:
        errstr = "The list of TTNs to add cannot be empty!"
        raise ValueError(errstr)
    if len(ttns) == 1:
        return deepcopy(ttns[0])
    ref_ttn = ttns[0]
    new_tensors: dict[str, np.ndarray] = {}
    for node_id in ref_ttn.nodes.keys():
        new_tensor = _create_new_tensor(node_id, ttns)
        new_tensors[node_id] = new_tensor
    new_ttn = TreeTensorNetwork.from_tensors(ref_ttn,
                                             new_tensors)
    return new_ttn

def _create_new_tensor(node_id: str,
                       ttns: list[TreeTensorNetwork]
                       ) -> np.ndarray:
    """
    Create a new tensor for the added TTN node by stacking tensors
    from the corresponding nodes of the input TTNs.

    Args:
        node_id (str): The identifier of the node.
        ttns (list[TreeTensorNetwork]): The list of TTNs to add.

    Returns:
        np.ndarray: The new tensor for the added TTN node.
    """
    node_tensors = [ttn[node_id] for ttn in ttns]
    new_shape = _new_tensor_shape([nt[0] for nt in node_tensors])
    ref_node, ref_tensor = node_tensors[0]
    new_tensor = np.zeros(new_shape, dtype=ref_tensor.dtype)
    current_lower_bounds = {neigh_id: 0
                            for neigh_id in ref_node.neighbouring_nodes()}
    num_open_legs = ref_node.nopen_legs()
    open_slices = [slice(None)
                   for _ in range(num_open_legs)]
    for node, tensor in node_tensors:
        perm = relative_leg_permutation(ref_node, node)
        permuted_tensor = np.transpose(tensor, axes=perm)
        insert_slices = []
        for neigh_id in ref_node.neighbouring_nodes():
            dim = node.neighbour_dim(neigh_id)
            lower_bound = current_lower_bounds[neigh_id]
            upper_bound = lower_bound + dim
            insert_slices.append(slice(lower_bound, upper_bound))
            current_lower_bounds[neigh_id] = upper_bound
        print(node_id, insert_slices, open_slices)
        new_tensor[tuple(insert_slices + open_slices)] = permuted_tensor
    return new_tensor

def _new_tensor_shape(nodes: list[Node]) -> list[int]:
    """
    Determine the shape of the new tensor for the added TTN node.

    Args:
        nodes (list[Node]): The list of corresponding nodes from the input TTNs.

    Returns:
        list[int]: The shape of the new tensor.
    """
    ref_node = nodes[0]
    shapes: dict[str, list[int]] = {neigh_id: []
                              for neigh_id in ref_node.neighbouring_nodes()}
    for node in nodes:
        for neighbour_id in node.neighbouring_nodes():
            dim = node.neighbour_dim(neighbour_id)
            shapes[neighbour_id].append(dim)
    new_shape = [sum(shapes[neigh_id])
                 for neigh_id in ref_node.neighbouring_nodes()]
    open_shape = ref_node.open_dimensions()
    new_shape.extend(open_shape)
    return new_shape
        