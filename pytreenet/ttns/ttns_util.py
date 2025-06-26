"""
This module provides utility functions useful in conjunction with TTNS.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from numpy import eye

from ..operators.tensorproduct import TensorProduct
from ..util.tensor_splitting import SplitMode
from ..contractions.tree_cach_dict import PartialTreeCachDict
from ..contractions.state_state_contraction import contract_any_nodes
from ..contractions.state_operator_contraction import contract_single_site_operator_env

if TYPE_CHECKING:
    from ..ttns import TreeTensorNetworkState

def _operator_valid(operator: TensorProduct):
    """
    Raises an error if the operator is not a single-site operator.
    """
    if operator.num_sites() != 1:
        raise ValueError("Operator must be a single-site operator!")

def prepare_single_site_operators(
    operators: TensorProduct | list[TensorProduct] | dict[str, TensorProduct | list[TensorProduct]]
    ) -> dict[str, list[TensorProduct]]:
    """
    Prepares the single-site operators for the expectation value calculation.

    Args:
        operators (TensorProduct | list[TensorProduct] | dict[str, TensorProduct | list[TensorProduct]]):
            The single-site operators to be prepared. This can be a single
            operator, a list of operators, or a dictionary mapping node IDs to
            operators. If a single operator is provided, it will be converted to
            multiple single-site operators.
    
    Returns:
        dict[str, list[TensorProduct]]: A dictionary mapping node IDs to the
            corresponding single-site operators.
    """
    if isinstance(operators, TensorProduct):
        return {node_id: [TensorProduct({node_id: operator})]
               for node_id, operator in operators.items()}
    if isinstance(operators, list):
        out = {}
        for operator in operators:
            _operator_valid(operator)
            node_id = operator.node_id
            if node_id in out:
                out[node_id].append(operator)
            else:
                out[node_id] = [operator]
        return out
    if isinstance(operators, dict):
        for node_id, operator in operators.items():
            if isinstance(operator, list):
                for op in operator:
                    _operator_valid(op)
            else:
                _operator_valid(operator)
                operators[node_id] = [operator]
        return operators
    raise TypeError("Invalid input type of operators!")

def _trivial_env_dict(
    ttns: TreeTensorNetworkState
    ) -> PartialTreeCachDict:
    """
    Generates a trivial environment dictionary for the TTNS.

    A trivial environment is simply the identity acting as the contracted
    subtree environment.

    This assumes that the root is the orthogonality center and that the TTNS
    is in valid canonical form.

    Args:
        ttns (TreeTensorNetworkState): The TTNS object.
    
    Returns:
        PartialTreeCachDict: A dictionary mapping node ID pairs to the trivial
            environment tensors.
    """
    out = PartialTreeCachDict()
    root_id = ttns.root_id
    for node_id, node in ttns.nodes.items():
        if node_id != root_id:
            dim = node.parent_leg_dim()
            ident = eye(dim, dtype=complex)
            parent_id = node.parent
            out.add_entry(node_id, parent_id, ident)
    return out

def multi_single_site_expectation_value(
    ttns: TreeTensorNetworkState,
    operators: TensorProduct | list[TensorProduct] | dict[str, TensorProduct | list[TensorProduct]],
    move_orth: bool = False

    ) -> dict[str, list[complex]]:
    """
    Computes the expectation value of multiple single-site operators.

    Args:
        ttns (TTNS): The state as a TTNS.
        operators (TensorProduct | list[TensorProduct] | dict[str, TensorProduct | list[TensorProduct]]):
            The single-site operators to be prepared. This can be a single
            operator, a list of operators, or a dictionary mapping node IDs to
            operators. If a single operator is provided, it will be converted to
            multiple single-site operators.
        move_orth (bool): If True, the orthogonality center will be moved to the
            root at the end of this computation. Default is False.

    Returns:
        dict[str, list[complex]]: The resulting expectation values < TTNS | operator | TTNS>
    """
    prepared_operators = prepare_single_site_operators(operators)
    if move_orth:
        old_orth = None
    else:
        old_orth = ttns.orthogonality_center_id
    # We move the orth centre to the root node, simplifying most contractions.
    root_id = ttns.root_id
    # We want to keep the bond dimensions for this, if this is undesired, the
    # TTNS should be truncated first.
    ttns.canonical_form(root_id, mode=SplitMode.KEEP)
    # Now we can precompute all the environments
    results = {node_id: [] for node_id in prepared_operators.keys()}
    env_dict = _trivial_env_dict(ttns)
    for node_id, local_operators in prepared_operators.items():
        # Copy envs to avoid overwriting them
        temp_env_dict = env_dict.copy()
        # We contract from the root to the node, so we can reuse the result
        # for all operators acting on the same node.
        path = ttns.path_from_to(root_id, node_id)
        for i, node_id_path in enumerate(path[:-1]):
            node, tensor = ttns[node_id_path]
            next_id = path[i + 1]
            new_env = contract_any_nodes(next_id,
                                         node,
                                         node,
                                         tensor,
                                         tensor.conj(),
                                         temp_env_dict)
            temp_env_dict.add_entry(node_id_path, next_id, new_env)
        # Now we have the environment for the node with the operator
        node, tensor = ttns[node_id]
        tensor_c = tensor.conj()
        for operator in local_operators:
            exp_val = contract_single_site_operator_env(node, tensor,
                                                        node, tensor_c,
                                                        operator[node_id],
                                                        temp_env_dict)
            results[node_id].append(exp_val)
        if old_orth is not None:
            # We move the orthogonality center back to the old position
            ttns.move_orthogonalization_center(old_orth)
    return results
