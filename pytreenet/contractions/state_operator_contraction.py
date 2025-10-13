"""
This module provides functions to contract a TTNS with a TTNO
"""

from __future__ import annotations
from typing import Union, Callable, TYPE_CHECKING
from enum import Enum

import numpy as np

from .tree_cach_dict import PartialTreeCachDict
from .contraction_util import contract_all_neighbour_blocks_to_ket
from .state_state_contraction import contract_node_with_environment_nodes as env_contract_node_state
from .local_contr import LocalContraction

if TYPE_CHECKING:
    from ..core.node import Node
    from ..ttns.ttns import TreeTensorNetworkState
    from ..ttno.ttno_class import TreeTensorNetworkOperator

__all__ = ['expectation_value', 'single_node_expectation_value']

def expectation_value(state: TreeTensorNetworkState,
                      operator: TreeTensorNetworkOperator,
                      operator_id_trafo: Callable | None = None,
                      ) -> complex:
    """
    Computes the Expecation value of a state with respect to an operator.

    The operator is given as a TTNO and the state as a TTNS. The expectation
    is obtained by "sandwiching" the operator between the state and its complex
    conjugate: <psi|H|psi>.

    Args:
        state (TreeTensorNetworkState): The TTNS representing the state.
        operator (TreeTensorNetworkOperator): The TTNO representing the Operator.
        operator_id_trafo (Callable | None, optional): A function that
            transforms the node identifier of the ket neighours to the
            identifiers of the operator node's neighbours. If None, the
            identity is assumed.

    Returns:
        complex: The expectation value.
    """
    dictionary = PartialTreeCachDict()
    # Getting a linear list of all identifiers
    computation_order = state.linearise()
    errstr = "The last element of the linearisation should be the root node."
    assert computation_order[-1] == state.root_id, errstr
    assert computation_order[-1] == operator.root_id, errstr
    for node_id in computation_order[:-1]: # The last one is the root node
        node = state.nodes[node_id]
        parent_id = node.parent
        # Due to the linearisation the children should already be contracted.
        block = contract_any(node_id, parent_id,
                             state, operator,
                             dictionary,
                             id_trafo_op=operator_id_trafo
                             )
        dictionary.add_entry(node_id,parent_id,block)
        # The children contraction results are not needed anymore.
        children = node.children
        for child_id in children:
            dictionary.delete_entry(child_id,node_id)
    # Now everything remaining is contracted into the root tensor.
    result = contract_node_with_environment(state.root_id,
                                            state, operator,
                                            dictionary,
                                            operator_id_trafo=operator_id_trafo)
    return result

def get_matrix_element(bra_state: TreeTensorNetworkState,
                       operator: TreeTensorNetworkOperator,
                       ket_state: TreeTensorNetworkState,
                       operator_id_trafo: Callable | None = None,
                       bra_id_trafo: Callable | None = None
                       ) -> complex:
    """
    Computes the matrix element of an operator between a bra and a ket state.

    The operator is given as a TTNO and the state as a TTNS. The matrix
    element is obtained by "sandwiching" the operator between the bra and
    ket state: <bra|H|state>.

    Args:
        bra_state (TreeTensorNetworkState): The TTNS representing the bra state.
        operator (TreeTensorNetworkOperator): The TTNO representing the Operator.
        ket_state (TreeTensorNetworkState): The TTNS representing the ket state.
        operator_id_trafo (Callable | None, optional): A function that
            transforms the node identifier of the ket neighours to the
            identifiers of the operator node's neighbours. If None, the
            identity is assumed.
        bra_id_trafo (Callable | None, optional): A function that
            transforms the node identifier of the ket neighours to the
            identifiers of the bra node's neighbours. If None, the identity
            is assumed.

    Returns:
        complex: The expectation value.
    """
    dictionary = PartialTreeCachDict()
    # Getting a linear list of all identifiers
    computation_order = ket_state.linearise()
    errstr = "The last element of the linearisation should be the root node."
    assert computation_order[-1] == ket_state.root_id, errstr
    assert computation_order[-1] == operator.root_id, errstr
    for node_id in computation_order[:-1]: # The last one is the root node
        node = ket_state.nodes[node_id]
        parent_id = node.parent
        # Due to the linearisation the children should already be contracted.
        block = contract_any(node_id, parent_id,
                             ket_state, operator,
                             dictionary,
                             bra_state=bra_state,
                             id_trafo_op=operator_id_trafo,
                             id_trafo_bra=bra_id_trafo)
        dictionary.add_entry(node_id,parent_id,block)
        # The children contraction results are not needed anymore.
        children = node.children
        for child_id in children:
            dictionary.delete_entry(child_id,node_id)
    # Now everything remaining is contracted into the root tensor.
    # Now everything remaining is contracted into the root tensor.
    result = contract_node_with_environment(ket_state.root_id,
                                            ket_state,
                                            operator,
                                            dictionary,
                                            bra_state=bra_state,
                                            operator_id_trafo=operator_id_trafo,
                                            bra_id_trafo=bra_id_trafo)
    return result

class FirstContraction(Enum):
    """
    An enum to decide which node is contracted to the environments first.

    Attributes
        KET (str): Contract the ket node first.
        OPERATOR (str): Contract the operator node first.
    """
    KET = "ket"
    OPERATOR = "operator"

    def contraction_order(self) -> list[int]:
        """
        Returns the contraction order as a list of integers.

        Returns:
            list[int]: The contraction order.
        """
        if self is FirstContraction.KET:
            return [0,1,2]
        return [1,0,2]

def compare_contr_orders(ket_node: Node,
                         op_node: Node
                         ) -> FirstContraction:
    """
    Compares the contraction cost of different orders of contracting.

    Args:
        ket_node (Node): The ket node.
        op_node (Node): The operator node.
    
    Returns:
        FirstContraction: The preferred contraction order.
    """
    phys_dim = ket_node.open_dimension()
    op_neigh_dim = op_node.virtual_dimension()
    ket_neigh_dim = ket_node.virtual_dimension()
    cost_lhs = (phys_dim - 1) * op_neigh_dim * ket_neigh_dim
    cost_rhs = phys_dim * (op_neigh_dim -  ket_neigh_dim)
    if cost_lhs >= cost_rhs:
        return FirstContraction.KET
    return FirstContraction.OPERATOR

def contract_node_with_environment(node_id: str,
                                   state: TreeTensorNetworkState,
                                   operator: TreeTensorNetworkOperator,
                                   dictionary: PartialTreeCachDict,
                                   bra_state: Union[TreeTensorNetworkState,None] = None,
                                   operator_id_trafo: Union[Callable,None] = None,
                                   bra_id_trafo: Union[Callable,None] = None
                                   ) -> complex:
    """
    Contracts a node with its environment.

    Assumes that all subtrees starting from this node are already contracted
    and the results stored in the dictionary.

    Args:
        node_id (str): The identifier of the node.
        state (TreeTensorNetworkState): The TTNS representing the state.
        operator (TreeTensorNetworkOperator): The TTNO representing the Hamiltonian.
        dictionary (PartialTreeCacheDict): The dictionary containing the
         already contracted subtrees.
        bra_state (Union[TreeTensorNetworkState,None]): The TTNS representing the bra state.
            If not given, the bra state is assumed to be the conjugate of the
            ket state.
        operator_id_trafo (Union[Callable,None], optional): A function that
            transforms the node identifier of the ket neighours to the
            identifiers of the operator node's neighbours. If None, the
            identity is assumed.
        bra_id_trafo (Union[Callable,None], optional): A function that
            transforms the node identifier of the ket neighours to the
            identifiers of the bra node's neighbours. If None, the identity
            is assumed.

    Returns:
        complex: The resulting tensor. A and B are the tensors in state1 and
            state2, respectively, corresponding to the node with the identifier
            node_id. C aer the tensors in the dictionary corresponding to the
            subtrees going away from the node::

                            ______
                 _____     |      |      _____
                |     |____|  A*  |_____|     |
                |     |    |______|     |     |
                |     |        |        |     |
                |     |     ___|__      |     |
                |     |    |      |     |     |
                |     |____|      |_____|     |
                |  C1 |    |   H  |     |  C2 |
                |     |    |______|     |     |
                |     |        |        |     |
                |     |     ___|__      |     |
                |     |    |      |     |     |
                |     |____|  A   |_____|     |
                |_____|    |______|     |_____|
    
    """
    ket_node, ket_tensor = state[node_id]
    op_node, op_tensor = operator[operator_id_trafo(node_id)
                                  if operator_id_trafo is not None else node_id]
    if bra_state is None:
        bra_node, bra_tensor = ket_node, ket_tensor.conj()
    else:
        bra_node, bra_tensor = bra_state[bra_id_trafo(node_id)
                                         if bra_id_trafo is not None else node_id]
    return contract_node_with_environment_nodes(ket_node, ket_tensor,
                                                op_node, op_tensor,
                                                dictionary,
                                                bra_node=bra_node,
                                                bra_tensor=bra_tensor,
                                                operator_id_trafo=operator_id_trafo,
                                                bra_id_trafo=bra_id_trafo)

def contract_node_with_environment_nodes(ket_node: Node, ket_tensor: np.ndarray,
                                         op_node: Node, op_tensor: np.ndarray,
                                         dictionary: PartialTreeCachDict,
                                         bra_node: Union[Node,None] = None,
                                         bra_tensor: Union[np.ndarray,None] = None,
                                         operator_id_trafo: Union[Callable,None] = None,
                                         bra_id_trafo: Union[Callable,None] = None
                                        ) -> complex:
    """
    Contracts a node with its environment.

    Assumes that all subtrees starting from this node are already contracted
    and the results stored in the dictionary.

    Args:
        ket_node (Node): The ket node.
        ket_tensor (np.ndarray): The ket tensor.
        op_node (Node): The operator node.
        op_tensor (np.ndarray): The operator tensor.
        dictionary (PartialTreeCacheDict): The dictionary containing the
         already contracted subtrees.
        bra_node (Union[Node,None], optional): The bra node. If None, the bra
            tensor is assumed to be the adjoint of the given ket tensor.
        bra_tensor (Union[np.ndarray,None], optional): The bra tensor. If None,
            the bra tensor is assumed to be the adjoint of the given ket tensor.
        operator_id_trafo (Union[Callable,None], optional): A function that
            transforms the node identifier of the ket neighours to the
            identifiers of the operator node's neighbours. If None, the
            identity is assumed.
        bra_id_trafo (Union[Callable,None], optional): A function that
            transforms the node identifier of the ket neighours to the
            identifiers of the bra node's neighbours. If None, the identity
            is assumed.

    Returns:
        complex: The resulting tensor. A and B are the tensors in state1 and
            state2, respectively, corresponding to the node with the identifier
            node_id. C aer the tensors in the dictionary corresponding to the
            subtrees going away from the node::

                            ______
                 _____     |      |      _____
                |     |____|  A*  |_____|     |
                |     |    |______|     |     |
                |     |        |        |     |
                |     |     ___|__      |     |
                |     |    |      |     |     |
                |     |____|      |_____|     |
                |  C1 |    |   H  |     |  C2 |
                |     |    |______|     |     |
                |     |        |        |     |
                |     |     ___|__      |     |
                |     |    |      |     |     |
                |     |____|  A   |_____|     |
                |_____|    |______|     |_____|
    
    """
    contr_order = compare_contr_orders(ket_node, op_node).contraction_order()
    nodes_tensors = [(ket_node, ket_tensor),
                     (op_node, op_tensor),
                     init_bra_node_and_tensor(ket_node,ket_tensor,
                                              bra_node,bra_tensor)]
    loc_contr = LocalContraction(nodes_tensors,
                                 dictionary,
                                 id_trafos=[None,operator_id_trafo,bra_id_trafo],
                                 contraction_order=contr_order)
    return loc_contr.contract_to_scalar()

def contract_ket_ham_with_envs(ket_node: Node,
                              ket_tensor: np.ndarray,
                              ham_node: Node,
                              ham_tensor: np.ndarray,
                              dictionary: PartialTreeCachDict,
                              operator_id_trafo: Union[Callable,None] = None
                              ) -> np.ndarray:
    """
    Contract a state node and a Hamiltonian node with their environments.

    Args:
        ket_node (Node): The ket node.
        ket_tensor (np.ndarray): The ket tensor.
        ham_node (Node): The Hamiltonian node.
        ham_tensor (np.ndarray): The Hamiltonian tensor.
        dictionary (PartialTreeCachDict): The dictionary containing the
            already contracted subtrees.
        operator_id_trafo (Union[Callable,None], optional): A function that
            transforms the node identifier of the ket neighours to the
            identifiers of the operator node's neighbours. If None, the
            identity is assumed.

    Returns:
        np.ndarray: The contracted tensor.

                 _____                   _____
                |     |____        _____|     |
                |     |                 |     |
                |     |        |        |     |
                |     |     ___|__      |     |
                |     |    |      |     |     |
                |     |____|      |_____|     |
                |  C1 |    |   H  |     |  C2 |
                |     |    |______|     |     |
                |     |        |        |     |
                |     |     ___|__      |     |
                |     |    |      |     |     |
                |     |____|  A   |_____|     |
                |_____|    |______|     |_____|
    
    """
    nodes_tensors = [(ket_node, ket_tensor),
                     (ham_node, ham_tensor)]
    loc_contr = LocalContraction(nodes_tensors,
                                 dictionary,
                                 id_trafos=[None,operator_id_trafo])
    return loc_contr()

def contract_single_site_operator_env(ket_node: Node,
                                      ket_tensor: np.ndarray,
                                      bra_node: Node,
                                      bra_tensor: np.ndarray,
                                      operator: np.ndarray,
                                      dictionary: PartialTreeCachDict,
                                      id_trafo: Union[Callable,None] = None
                                      ) -> complex:
    """
    Contract a node to which a single-site operator is applied with given
    environment tensors.

    Args:
        ket_node (Node): The ket node.
        ket_tensor (np.ndarray): The ket tensor.
        bra_node (Node): The bra node.
        bra_tensor (np.ndarray): The bra tensor.
        operator (np.ndarray): The single-site operator.
        dictionary (PartialTreeCacheDict): The dictionary containing the
            already contracted subtrees.
        id_trafo (Union[None,Callable]): A function to transform the node
            identifiers of the ket node into the node identifiers of the bra
            node. If None, it is assumed that the identifiers are the same.
            Defaults to None.

    Returns:
        complex: The contraction result.
                            ______
                 _____     |      |      _____
                |     |____|  B   |_____|     |
                |     |    |______|     |     |
                |     |        |        |     |
                |     |     ___|__      |     |
                |     |    |      |     |     |
                |     |    |      |     |     |
                |  C1 |    |  O   |     |  C2 |
                |     |    |______|     |     |
                |     |        |        |     |
                |     |     ___|__      |     |
                |     |    |      |     |     |
                |     |____|  K   |_____|     |
                |_____|    |______|     |_____|
    
    """
    # Contract the operator with the ket tensor
    # The ket's open leg is the last leg and the operators
    # input leg as well.
    nopen_ket = ket_node.open_legs()
    noperator_legs = operator.ndim
    if noperator_legs % 2 != 0:
        errstr = "The operator must have an even number of legs!"
        raise ValueError(errstr)
    half = noperator_legs // 2
    if nopen_ket != half:
        errstr = f"The operator has {half} input legs, but the ket node has {nopen_ket} open legs!"
        raise ValueError(errstr)
    # To be consistend with the convential matrix-matrix product the input
    # legs are the second half of legs.
    operator_legs = list(range(half,noperator_legs))
    new_ket_tensor = np.tensordot(ket_tensor, operator,
                                  axes=(nopen_ket,operator_legs))
    return env_contract_node_state(ket_node, new_ket_tensor,
                                   bra_node, bra_tensor,
                                   dictionary,
                                   id_trafo=id_trafo)

def contract_any(node_id: str, next_node_id: str,
                 state: TreeTensorNetworkState,
                 operator: TreeTensorNetworkOperator,
                 dictionary: PartialTreeCachDict,
                 bra_state: Union[TreeTensorNetworkState,None] = None,
                 bra_state_conjugated: bool = True,
                 id_trafo_op: Union[Callable,None] = None,
                 id_trafo_bra: Union[Callable,None] = None
                    ) -> np.ndarray:
    """
    Contracts any node. 
    
    Rather the entire subtree starting from the node is contracted. The
    subtrees below the node already have to be contracted, except for the
    specified neighbour.
    This function combines the two options of contracting a leaf node or
    a general node using the dictionary in one function.
    
    Args:
        node_id (str): Identifier of the node.
        next_node_id (str): Identifier of the node towards which the open
            legs will point.
        state (TreeTensorNetworkState): The TTNS representing the state.
        operator (TreeTensorNetworkOperator): The TTNO representing the Hamiltonian.
        dictionary (PartialTreeCachDict): The dictionary containing the
            already contracted subtrees.
        bra_state (Union[TreeTensorNetworkState,None]): The TTNS representing
            the bra state.
        bra_state_conjugated (bool): Whether the bra state is conjugated.
        id_trafo_op (Union[Callable,None], optional): A function that transforms
            the node identifier of the ket neighours to the identifiers of the
            operator node's neighbours. If None, the identity is assumed.
        id_trafo_bra (Union[Callable,None], optional): A function that transforms
            the node identifier of the ket neighours to the identifiers of the
            bra node's neighbours. If None, the identity is assumed.

    Returns:
        np.ndarray: The contracted tensor.
    """
    state_node, state_tensor = state[node_id]
    operator_node, operator_tensor = operator[id_trafo_op(node_id)
                                              if id_trafo_op is not None else node_id]
    if bra_state is None:
        bra_node, bra_tensor = state_node, state_tensor.conj()
    else:
        bra_node, bra_tensor = bra_state[id_trafo_bra(node_id)
                                         if id_trafo_bra is not None else node_id]
        if not bra_state_conjugated:
            bra_tensor = bra_tensor.conj()
    return contract_any_node(next_node_id,
                                state_node, state_tensor,
                                operator_node, operator_tensor,
                                dictionary,
                                bra_node=bra_node,
                                bra_tensor=bra_tensor,
                                id_trafo_op=id_trafo_op,
                                id_trafo_bra=id_trafo_bra)

def contract_any_node(ignored_node_id: str,
                        ket_node: Node, ket_tensor: np.ndarray,
                        op_node: Node, op_tensor: np.ndarray,
                        dictionary: PartialTreeCachDict,
                        bra_node: Union[Node,None] = None,
                        bra_tensor: Union[np.ndarray,None] = None,
                        id_trafo_op: Union[Callable,None] = None,
                        id_trafo_bra: Union[Callable,None] = None
                        ) -> np.ndarray:
    """
    Contracts a node with all its subtrees except for one.

    All subtrees except for one are already contracted and stored in the
    dictionary. The one that is not contracted is the one that the remaining
    legs point towards. The dictionary has to be according to the node
    identifiers of the ket node neighbours.

    Args:
        ignored_node_id (str): Identifier of the ket node to which the
            remaining legs should point.
        ket_node (Node): The ket node.
        ket_tensor (np.ndarray): The ket tensor.
        op_node (Node): The operator node.
        op_tensor (np.ndarray): The operator tensor.
        dictionary (PartialTreeCachDict): The dictionary containing the
            already contracted subtrees.
        bra_node (Union[Node,None], optional): The bra node. If None, the bra
            tensor is assumed to be the adjoint of the given ket tensor.
        bra_tensor (Union[Node,None], optional): The bra tensor. If None, the
            bra tensor is assumed to be the adjoint of the given ket tensor.
        id_trafo_op (Union[Callable,None], optional): A function that transforms
            the node identifier of the ket neighours to the identifiers of the
            operator node's neighbours. If None, the identity is assumed.
        id_trafo_bra (Union[Callable,None], optional): A function that transforms
            the node identifier of the ket neighours to the identifiers of the
            bra node's neighbours. If None, the identity is assumed.

    Returns:
        np.ndarray: The contracted tensor::

                     _____      ______
              2 ____|     |____|      |
                    |  A* |    |      |
                    |_____|    |      |
                       |       |      |
                       |       |      |
                     __|__     |      |
              1 ____|     |____|      |
                    |  H  |    |  C   |
                    |_____|    |      |
                       |       |      |
                       |       |      |
                     __|__     |      |
              0 ____|     |____|      |
                    |  A  |    |      |
                    |_____|    |______|

    """
    bra_node, bra_tensor = init_bra_node_and_tensor(ket_node, ket_tensor,
                                                    bra_node, bra_tensor)
    node_tensors = [(ket_node, ket_tensor),
                    (op_node, op_tensor),
                    (bra_node, bra_tensor)]
    loc_contr = LocalContraction(node_tensors,
                                 dictionary,
                                 ignored_leg=ignored_node_id,
                                 id_trafos=[None,id_trafo_op,id_trafo_bra])
    return loc_contr()

def contract_leaf_node(ket_node: Node, ket_tensor: np.ndarray,
                       op_node: Node, op_tensor: np.ndarray,
                       bra_node: Node | None = None,
                       bra_tensor: np.ndarray | None = None,
                       id_trafo_op: Union[Callable,None] = None,
                       id_trafo_bra: Union[Callable,None] = None
                       ) -> np.ndarray:
    """
    Contracts a leaf node with an operator and a bra node.

    Args:
        ignored_node_id (str): Identifier of the ket node to which the
            remaining legs should point.
        ket_node (Node): The ket node.
        ket_tensor (np.ndarray): The ket tensor.
        op_node (Node): The operator node.
        op_tensor (np.ndarray): The operator tensor.
        dictionary (PartialTreeCachDict): The dictionary containing the
            already contracted subtrees.
        bra_node (Union[Node,None], optional): The bra node. If None, the bra
            tensor is assumed to be the adjoint of the given ket tensor.
        bra_tensor (Union[Node,None], optional): The bra tensor. If None, the
            bra tensor is assumed to be the adjoint of the given ket tensor.
        id_trafo_op (Union[Callable,None], optional): A function that transforms
            the node identifier of the ket neighours to the identifiers of the
            operator node's neighbours. If None, the identity is assumed.
        id_trafo_bra (Union[Callable,None], optional): A function that transforms
            the node identifier of the ket neighours to the identifiers of the
            bra node's neighbours. If None, the identity is assumed.

    Returns:
        np.ndarray: The contracted tensor::

                     _____ 
              2 ____|     |
                    |  A* |
                    |_____|
                       |   
                       |   
                     __|__ 
              1 ____|     |
                    |  H  |
                    |_____|
                       |   
                       |   
                     __|__ 
              0 ____|     |
                    |  A  |
                    |_____|

    """
    bra_node, bra_tensor = init_bra_node_and_tensor(ket_node, ket_tensor,
                                                    bra_node, bra_tensor)
    node_tensors = [(ket_node, ket_tensor),
                    (op_node, op_tensor),
                    (bra_node, bra_tensor)]
    loc_contr = LocalContraction(node_tensors,
                                 PartialTreeCachDict(),
                                 contraction_order=[0,1,2],
                                 id_trafos=[None,id_trafo_op,id_trafo_bra])
    return loc_contr()

def single_node_expectation_value(node: Node,
                                  ket_tensor: np.ndarray,
                                  operator_tensor: np.ndarray,
                                  bra_tensor: Union[np.ndarray,None] = None
                                  ) -> complex:
    """
    This is a function for the special case, in which a TTN is only one node.

    The function computes the expectation value of the given ket tensor with
    respect to the given operator tensor.

    Args:
        node (Node): The node of the TTN.
        ket_tensor (np.ndarray): The ket tensor.
        operator_tensor (np.ndarray): The operator tensor.
        bra_tensor (Union[np.ndarray,None], optional): The bra tensor. If None,
            the bra tensor is assumed to be the adjoint of the given ket tensor.

    Returns:
        complex: The expectation value.

    """
    assert node.is_leaf() and node.is_root(), "Node is not a leaf and a root!"
    assert ket_tensor.ndim == 1, "Ket tensor has too many legs"
    assert operator_tensor.ndim == 2, "Operator tensor has too many legs"
    if bra_tensor is None:
        bra_tensor = ket_tensor.conj().T
    else:
        assert bra_tensor.ndim == 1, "Bra tensor has too many legs"
    return bra_tensor @ operator_tensor @ ket_tensor

def contract_bond_tensor(
                        ket_tensor: np.ndarray,
                        ket_node: Node,
                        tensor_cache: PartialTreeCachDict
                        ) -> np.ndarray:
    """
    Contract an effective bond tensor with its neighbouring blocks.

    Args:
        state_tensor (NDArray[np.complex128]): The tensor of the state to be
            updated, representing a bond in the TTNS without an equivalent
            tensor in the TTNO.
        state_node (Node): The node of the state tensor.
        tensor_cache (SandwichCache): The cache for the neighbour blocks.

    Returns:
        NDArray[np.complex128]: The contracted tensor. The ket tensor and the
            neighbouring blocks are contracted with each other.

                 _____                   _____
                |     |____        _____|     |
                |     |                 |     |
                |     |                 |     |
                |     |                 |     |
                |     |                 |     |
                |     |_________________|     |
                |  C1 |                 |  C2 |
                |     |                 |     |
                |     |                 |     |
                |     |     ______      |     |
                |     |    |      |     |     |
                |     |____|  A   |_____|     |
                |_____|    |______|     |_____|
    """
    if ket_node.nneighbours()!= 2:
        errstr = "The node must have exactly two neighbours to perform " \
                "effective bond time evolution!"
        errstr += f" Node {ket_node.identifier} has {ket_node.nneighbours()} neighbours."
        raise ValueError(errstr)
    # We want to ensure the order of the legs
    order = [ket_node.parent,ket_node.children[0]]
    # The environments in the cache point to the other neighbour
    # Thus, we have to add keys to the cache
    ket_id = ket_node.identifier
    tensor_cache.add_entry(order[0], ket_id,
                           tensor_cache.get_entry(order[0], order[1]))
    tensor_cache.add_entry(order[1], ket_id,
                            tensor_cache.get_entry(order[1], order[0]))
    ketandblocks = contract_all_neighbour_blocks_to_ket(ket_tensor,
                                                        ket_node,
                                                        tensor_cache,
                                                        order=order)
    # Delete the new keys from the cache again
    for node_id in order:
        tensor_cache.delete_entry(node_id, ket_id)
    # Now we connect the Hamiltonian legs of the two neighbours.
    return np.einsum("ijil -> jl", ketandblocks)

def init_bra_node_and_tensor(ket_node: Node,
                              ket_tensor: np.ndarray,
                              bra_node: Node | None,
                              bra_tensor: np.ndarray | None
                              ) -> tuple[Node, np.ndarray]:
    """
    Initializes the bra node and tensor.

    If no bra node and tensor are given, the bra node and tensor are assumed
    to be the adjoint of the ket node and tensor.

    Args:
        ket_node (Node): The ket node.
        ket_tensor (np.ndarray): The ket tensor.
        bra_node (Node | None): The bra node. If None, the bra
            tensor is assumed to be the adjoint of the given ket tensor.
        bra_tensor (np.ndarray | None): The bra tensor. If None, the
            bra tensor is assumed to be the adjoint of the given ket tensor.

    Returns:
        tuple[Node, np.ndarray]: The bra node and tensor.
    
    """
    if bra_node is None:
        bra_node = ket_node
    if bra_tensor is None:
        bra_tensor = ket_tensor.conj()
    return bra_node, bra_tensor
