"""
Implements the variational fitting algorithm for TTNS.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from ...time_evolution.time_evo_util.update_path import (TDVPUpdatePathFinder,
                                                         find_orthogonalisation_path)
from ...contractions.tree_cach_dict import PartialTreeCachDict
from ...contractions.state_state_contraction import contract_any
from ...contractions.contraction_util import contract_all_neighbour_blocks_to_ket

if TYPE_CHECKING:
    from ...ttns.ttns import TTNS
    from ..tree_structure import TreeStructure

__all__ = ["single_site_fitting"]

def single_site_fitting(init_ttns: TTNS,
                        target_ttns: TTNS,
                        num_sweeps: int,
                        record_sweep_errors: bool = False
                        ) -> list[float]:
    """
    Truncate a TTNS to a given TTNS bond dimension structure.

    Args:
        init_ttns (TTNS): The initial TTNS to be fitted. It will be modified
            in place and also returned.
        target_ttns (TTNS): The target TTNS to fit to.
        num_sweeps (int): The number of sweeps to perform. One sweep consists
            of a forward and a backward sweep.
        record_sweep_errors (bool): Whether to record the fitting error after
            each sweep. This will slow down the algorithm, as it requires
            a full contraction of the tree after each sweep. Note, this only
            works for normalised states. Defaults to `False`.

    Returns:
        list[float]: The fitting errors after each sweep, if
            `record_sweep_errors` is `True`. Otherwise, an empty list.    
    """
    sweep_path, orth_paths, back_sweep_path, back_orth_paths = find_update_paths(init_ttns)
    init_ttns.canonical_form(sweep_path[0])
    if record_sweep_errors:
        infid = init_ttns.infidelity(target_ttns)
        error = [infid]
    else:
        error = []
    subtree_tensors = PartialTreeCachDict()
    # Initialise the subtree tensors
    cache_path = find_cache_path(init_ttns, sweep_path[0])
    for node_id, next_node_id in cache_path:
        # Note that the state to be optimised is conjugated
        tensor = contract_any(node_id, next_node_id,
                              target_ttns,
                              init_ttns,
                              subtree_tensors,
                              state2_conj=True)
        subtree_tensors.add_entry(node_id, next_node_id,
                                  tensor)
    for _ in range(num_sweeps):
        # Forward sweep
        sweep(init_ttns, target_ttns, sweep_path, orth_paths, subtree_tensors)
        # Backward sweep
        sweep(init_ttns, target_ttns, back_sweep_path, back_orth_paths, subtree_tensors)
        if record_sweep_errors:
            infid = init_ttns.infidelity(target_ttns)
            error.append(infid)
    return error

def sweep(init_ttns: TTNS,
                  target_ttns: TTNS,
                  sweep_path: list[str],
                  orth_paths: list[list[str]],
                  subtree_tensors: PartialTreeCachDict
                  ) -> None:
    """
    Perform a sweep of the variational fitting algorithm along the given path.

    Args:
        init_ttns (TTNS): The initial TTNS to be fitted.
        target_ttns (TTNS): The target TTNS to fit to.
        sweep_path (list[str]): The path to sweep through the tree.
        orth_paths (list[list[str]]): The path to orthogonalise the tree.
        subtree_tensors (PartialTreeCacheDict): The dictionary containing the
            already contracted subtrees.
    """
    for index, node_id in enumerate(sweep_path):
        assert init_ttns.orthogonality_center_id == node_id
        node, tensor = target_ttns[node_id]
        # We need the legs to be in the order to fit with the ket node.
        init_node = init_ttns.nodes[node_id]
        order = init_node.neighbouring_nodes()
        new_tensor = contract_all_neighbour_blocks_to_ket(tensor,
                                                          node,
                                                          subtree_tensors,
                                                          order=order,
                                                          output=True)
        init_ttns.replace_tensor(node_id, new_tensor)
        if index != len(sweep_path) - 1:
            # Otherwise we are at the end of the sweep and do not need to
            # move the orthogonality center
            orth_path = orth_paths[index]
            init_ttns.move_orthogonalization_center(orth_path[-1])
            # Update the cache
            _update_cache(orth_path, target_ttns, init_ttns, subtree_tensors)

def find_update_paths(tree: TreeStructure
                      ) -> tuple[list[str], list[list[str]], list[str], list[list[str]]]:
    """
    Find the update paths for a given tree structure.

    Args:
        tree (TreeStructure): The tree structure to find the update paths for.

    Returns:
        tuple[list[str], list[list[str]], list[str], list[list[str]]]: A tuple
            containing the forward sweep path, the forward orthogonalisation
            paths, the backward sweep path and the backward orthogonalisation
            paths.
    """
    sweep_path = TDVPUpdatePathFinder(tree).find_path()
    orth_paths = find_orthogonalisation_path(sweep_path, tree,
                                             include_start=True)
    back_sweep_path = list(reversed(sweep_path))
    back_orth_paths = find_orthogonalisation_path(back_sweep_path, tree,
                                                  include_start=True)
    return sweep_path, orth_paths, back_sweep_path, back_orth_paths

def find_cache_path(tree: TreeStructure,
                    left_out_id: str
                    ) -> list[tuple[str,str]]:
    """
    Find the caching path for the tree structure.

    Args:
        tree (TreeStructure): The tree structure to find the caching path for.
        left_out_id (str): The node identifier to not cache for.

    Returns:
        list[tuple[str,str]]: The caching path as a list of tuples. The first
            element of the tuple is the `node_id` on which the contraction
            should happen and the second element is the `next_node_id` to
            which the open legs of the resulting subtree tensor should point.
            The path is ordered such that the first element is also the first
            contraction to be performed.
    """
    path = []
    node_ids_to_visit = [left_out_id]
    visited_node_ids = set()
    for node_id in node_ids_to_visit:
        node = tree.nodes[node_id]
        for neigh_id in node.neighbouring_nodes():
            if neigh_id not in visited_node_ids:
                path.append((neigh_id, node_id))
                # This modification is intended
                node_ids_to_visit.append(neigh_id)
        visited_node_ids.add(node_id)
    # We need to reverse it, to ensure all nodes neighbours are visited and
    # thus the subtree tensor exists for the contraction
    path.reverse()
    return path

def _update_cache(orth_path: list[str],
                  target_ttns: TTNS,
                  init_ttns: TTNS,
                  subtree_tensors: PartialTreeCachDict
                  ) -> None:
    """
    Update the cache of subtree tensors after moving the orthogonality center.

    Args:
        orth_path (list[str]): The path to orthogonalise the tree.
        target_ttns (TTNS): The target TTNS to fit to.
        init_ttns (TTNS): The initial TTNS to be fitted.
        subtree_tensors (PartialTreeCacheDict): The dictionary containing the
            already contracted subtrees.
    """
    for i, node_id in enumerate(orth_path[:-1]):
        next_node_id = orth_path[i + 1]
        # Note that the state to be optimised is conjugated
        cache_tensor = contract_any(node_id, next_node_id,
                                    target_ttns,
                                    init_ttns,
                                    subtree_tensors,
                                    state2_conj=True)
        subtree_tensors.add_entry(node_id, next_node_id,
                                  cache_tensor)
