"""
This module provides functions to multiply a TTNS with a TTNO via the zip-up method.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable

import numpy as np

from ...core.node import Node
from ...contractions.tree_cach_dict import PartialTreeCachDict
from ...util.tensor_splitting import (SVDParameters,
                                      contr_truncated_svd_splitting)
from ..ttns import TreeTensorNetworkState
from ...contractions.local_contr import LocalContraction
from ...util.std_utils import identity_mapping

if TYPE_CHECKING:
    from ...ttno.ttno_class import TreeTensorNetworkOperator

__all__ = ['zipup']

def zipup(operator: TreeTensorNetworkOperator,
          state: TreeTensorNetworkState,
          svd_params: SVDParameters | None = None,
          id_trafo: Callable = identity_mapping,
          canonicalize: bool = False
          ) -> TreeTensorNetworkState:
    """
    Apply a TTNO to a TTNS.

    Args:
        operator (TreeTensorNetworkOperator): The TTNO to apply.
        state (TreeTensorNetworkState): The TTNS that TTNO acts on.
        svd_params (SVDParameters | None): The SVD parameters.
        id_trafo (Callable): A function that transforms the node
            identifiers of the state to the identifiers of the operator.
            By default the identity mapping is used.
        canonicalize (bool): Whether to canonicalize before runnin ZipUp.
            This improves accuracy but can be costly due to running the
            canonicalization procedure. By default False.

    Returns:    
        TreeTensorNetworkState: The result of the application of the TTNO to the TTNS.
    """
    if canonicalize:
        return canonical_zipup(operator,
                               state,
                               svd_params,
                               id_trafo)
    return non_canonical_zipup(operator,
                               state,
                               svd_params,
                               id_trafo)

def non_canonical_zipup(operator: TreeTensorNetworkOperator,
                        state: TreeTensorNetworkState,
                        svd_params: SVDParameters | None = None,
                        id_trafo: Callable = identity_mapping
                        ) -> TreeTensorNetworkState:
    """
    Apply a TTNO to a TTNS.

    Args:
        operator (TreeTensorNetworkOperator): The TTNO to apply.
        state (TreeTensorNetworkState): The TTNS that TTNO acts on.
        svd_params (SVDParameters | None): The SVD parameters.
        id_trafo (Callable): A function that transforms the node
            identifiers of the state to the identifiers of the operator.
            By default the identity mapping is used.

    Returns:    
        TreeTensorNetworkState: The result of the application of the TTNO to the TTNS.
    """
    if svd_params is None:
        svd_params = SVDParameters()
    dictionary = PartialTreeCachDict()
    # Getting a linear list of all identifiers
    computation_order = state.linearise()
    new_tensors = {}
    for node_id in computation_order[:-1]: # The last one is the root node
        node = state.nodes[node_id]
        parent_id = node.parent
        assert parent_id is not None
        # Due to the linearisation the children should already be contracted.
        q,r = contract_any(node_id, parent_id,
                             state, operator,
                             dictionary, svd_params,
                             id_trafo)
        new_tensors[node_id] = q
        dictionary.add_entry(node_id,parent_id,r)
        # The children contraction results are not needed anymore.
        children = node.children
        for child_id in children:
            dictionary.delete_entry(child_id,node_id)
    # Now everything remaining is contracted into the root tensor.
    root_id = state.root_id
    assert root_id is not None
    tensor = contract_root(state[root_id],
                            operator[id_trafo(root_id)],
                            dictionary,
                            id_trafo)
    new_tensors[root_id] = tensor
    resl_ttns = TreeTensorNetworkState.from_tensors(state, new_tensors)
    return resl_ttns

def contract_root(ket_node_tensor: tuple[Node, np.ndarray],
                  op_node_tensor: tuple[Node, np.ndarray],
                  dictionary: PartialTreeCachDict,
                  id_trafo: Callable = identity_mapping
                  ) -> np.ndarray:
    """
    Contracts the root node with all its subtrees.

    All subtrees are already contracted and stored in the dictionary.

    Args:
        ket_node_tensor (tuple[Node, np.ndarray]): The node and tensor
            corresponding to the state at the root.
        op_node_tensor (tuple[Node, np.ndarray]): The node and tensor
            corresponding to the operator at the root.
        dictionary (PartialTreeCacheDict): The dictionary containing the
            already contracted subtrees.
        id_trafo (Callable): A function that transforms the node
            identifiers of the state to the identifiers of the operator.
    
    Returns:
        np.ndarray: The contracted tensor::

                 _____                   _____
                |     |        |        |     |
                |     |     ___|__      |     |
                |     |    |      |     |     |
            ____|     |____|      |_____|     |____
                |  C1 |    |   H  |     |  C2 |
                |     |    |______|     |     |
                |     |        |        |     |
                |     |     ___|__      |     |
                |     |    |      |     |     |
                |     |____|  A   |_____|     |
                |_____|    |______|     |_____|
    
    """
    nodes_tensors = [ket_node_tensor, op_node_tensor]
    id_trafos = [identity_mapping, id_trafo]
    loc_contr = LocalContraction(nodes_tensors,
                                 dictionary,
                                 id_trafos=id_trafos)
    new_tensor = loc_contr()
    return new_tensor

def contract_any(node_id: str,
                 next_node_id: str,
                 state: TreeTensorNetworkState,
                 operator: TreeTensorNetworkOperator,
                 dictionary: PartialTreeCachDict,
                 svd_params: SVDParameters,
                 id_trafo: Callable
                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Contracts any node. 
    
    Rather the entire subtree starting from the node is contracted. The
    subtrees below the node already have to be contracted, except for the
    specified neighbour.
    This function combines the two options of contracting a leaf node or
    a general node using the dictionary in one function.
    
    Args:
        node_id (str): Identifier of the node.
        next_node_id (str): Identifier of the neighbour node to which
            the remaining legs should point.
        state (TreeTensorNetworkState): The TTNS representing the state.
        operator (TTNO): The TTNO representing the Hamiltonian.
        dictionary (PartialTreeCachDict): The dictionary containing the
            already contracted subtrees.
        svd_params (SVDParameters): The SVD parameters for the splitting.
        id_trafo (Callable): A function that transforms the node
            identifiers of the state to the identifiers of the operator.
        
    Returns:
        tuple[np.ndarray, np.ndarray]: The resulting tensors (Q,R) after
            splitting the contracted tensor with a QR decomposition.
    """
    return contract_any_nodes(next_node_id,
                              state[node_id],
                              operator[id_trafo(node_id)],
                              dictionary,
                              svd_params,
                              id_trafo)

def contract_any_nodes(ignored_node_id: str,
                       ket_node_tensor: tuple[Node, np.ndarray],
                       op_node_tensor: tuple[Node, np.ndarray],
                       dictionary: PartialTreeCachDict,
                       svd_params: SVDParameters,
                       id_trafo: Callable
                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    Contracts a node with all its subtrees except for one.

    All subtrees except for one are already contracted and stored in the
    dictionary. The one that is not contracted is the one that the remaining
    legs point towards. The dictionary has to be according to the node
    identifiers of the ket node neighbours.

    Args:
        ignored_node_id (str): Identifier of the ket node to which the
            remaining legs should point.
        ket_node_tensor (tuple[Node, np.ndarray]): The node and tensor
            corresponding to the state at this position.
        op_node_tensor (tuple[Node, np.ndarray]): The node and tensor
            corresponding to the operator at this position.
        dictionary (PartialTreeCacheDict): The dictionary containing the
            already contracted subtrees.
        svd_params (SVDParameters): The SVD parameters for the splitting.
        id_trafo (Callable): A function that transforms the node
            identifiers of the state to the identifiers of the operator.

    Returns:
        tuple[np.ndarray, np.ndarray]: The contracted tensor::

                                ______
                     __|__     |      |
              1 ____|     |____|      |
                    |  H  |    |  C   |
                    |_____|    |      |
                       |       |      |_____
                       |       |      |
                     __|__     |      |
              0 ____|     |____|      |
                    |  A  |    |      |
                    |_____|    |______|

            the resulting tensor is split with the physical legs and the legs
            remaining from the subtree tensor (C) on the Q and the legs
            pointing towards the ignored node on the R.
    """
    nodes_tensors = [ket_node_tensor, op_node_tensor]
    id_trafos = [identity_mapping, id_trafo]
    loc_contr = LocalContraction(nodes_tensors,
                                 dictionary,
                                 ignored_leg=ignored_node_id,
                                 id_trafos=id_trafos)
    new_tensor = loc_contr()
    neigh_leg = ket_node_tensor[0].neighbour_index(ignored_node_id)
    r_legs = (neigh_leg, neigh_leg + 1)
    q_legs = tuple(filter(lambda i: i not in r_legs, range(new_tensor.ndim)))
    q, r = contr_truncated_svd_splitting(new_tensor,
                                         tuple(q_legs), r_legs,
                                         svd_params=svd_params)
    new_state_tensor = np.moveaxis(q, -1, neigh_leg)
    cache_tensor = r.transpose([1,2,0])
    return new_state_tensor, cache_tensor

def canonical_zipup(operator: TreeTensorNetworkOperator,
                    state: TreeTensorNetworkState,
                    svd_params: SVDParameters | None = None,
                    id_trafo: Callable = identity_mapping
                    ) -> TreeTensorNetworkState:
    """
    Apply a TTNO to a TTNS.

    Args:
        operator (TreeTensorNetworkOperator): The TTNO to apply.
        state (TreeTensorNetworkState): The TTNS that TTNO acts on.
        svd_params (SVDParameters | None): The SVD parameters.
        id_trafo (Callable): A function that transforms the node
            identifiers of the state to the identifiers of the operator.
            By default the identity mapping is used.

    Returns:    
        TreeTensorNetworkState: The result of the application of the TTNO to the TTNS.
    """
    if svd_params is None:
        svd_params = SVDParameters()
    leafs = state.get_leafs()
    if len(leafs) == 0:
        # There are no tensors in both TTN.
        return state
    r_tensors = PartialTreeCachDict()
    if len(leafs) <= 2:
        # This is basically the MPS case, in which we can optimise the zip-up.
        new_tensors: dict[str, np.ndarray] = {}
        current_node_id = leafs[0]
        state.canonical_form(current_node_id)
        operator.canonical_form(id_trafo(current_node_id))
        # We need to pull the node after the canonicalisation,
        # as the nodes in the state will change due to it.
        current_node = state.nodes[current_node_id]
        while not current_node.is_root():
            # We are on the way up to the root.
            ignored_node_id = current_node.parent
            assert ignored_node_id is not None
            q, r = contract_any(current_node_id,
                                 ignored_node_id,
                                 state, operator,
                                 r_tensors, svd_params,
                                 id_trafo)
            new_tensors[current_node_id] = q
            r_tensors.add_entry(current_node_id,
                                ignored_node_id,
                                r)
            # Move to the parent node.
            last_node_id = current_node_id # Needed to know for the root.
            current_node_id = ignored_node_id
            current_node = state.nodes[current_node_id]
        # Now we are at the root node.
        assert current_node_id == state.root_id
        children = current_node.children
        if len(children) == 1:
            # The only child is already contracted,
            # So we only need to contract the r-tensors.
            new_root_tensor = contract_root(state[current_node_id],
                                            operator[id_trafo(current_node_id)],
                                            r_tensors,
                                            id_trafo)
            new_tensors[current_node_id] = new_root_tensor
        else:
            # We need to walk down the tree to contract everything.
            # This also means, the root is treated as any other node.
            child_id = next(filter(lambda cid: cid != last_node_id, children))
            while not current_node.is_leaf():
                ignored_node_id = child_id
                q, r = contract_any(current_node_id,
                                     ignored_node_id,
                                     state, operator,
                                     r_tensors, svd_params,
                                     id_trafo)
                new_tensors[current_node_id] = q
                r_tensors.add_entry(current_node_id,
                                    ignored_node_id,
                                    r)
                # Move to the child node.
                current_node_id = child_id
                current_node = state.nodes[current_node_id]
                if not current_node.is_leaf():
                    # The only child is the next one to contract.
                    child_id = current_node.children[0]
            # Now we are at the other leaf node, which we only need to contract
            new_leaf_tensor = contract_root(state[current_node_id],
                                            operator[id_trafo(current_node_id)],
                                            r_tensors,
                                            id_trafo)
            new_tensors[current_node_id] = new_leaf_tensor
        resl_ttns = TreeTensorNetworkState.from_tensors(state, new_tensors)
        return resl_ttns
    errstr = "Canonical zip-up is currently only implemented for TTNS with at most two leafs."
    raise NotImplementedError(errstr)
