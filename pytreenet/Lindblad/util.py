from ..core.node import Node
from ..contractions.state_state_contraction import contract_two_ttns
from ..ttno import TTNO
from copy import deepcopy
from ..util.tensor_util import tensor_matricization
from ..util.tensor_splitting import _determine_tensor_shape , SplitMode
from ..ttns import TreeTensorNetworkState
import numpy as np
from ..contractions.contraction_util import (get_equivalent_legs, 
                                             contract_all_but_one_neighbour_block_to_ket,
                                             contract_all_neighbour_blocks_to_ket)
from ..contractions.tree_cach_dict import PartialTreeCachDict
from ..contractions.state_operator_contraction import (_node_operator_input_leg,
                                                       _node_operator_output_leg,
                                                       _node_state_phys_leg,
                                                       contract_operator_tensor_ignoring_one_leg,
                                                       contract_bra_tensor_ignore_one_leg)
import pytreenet as ptn

def transpose_node_with_neighbouring_nodes(state, ket_id, neighbours):
    perm = []
    for neighbour in neighbours: 
        n = state.nodes[ket_id].neighbour_index(neighbour)
        perm.append(n)    
    if state.nodes[ket_id].nopen_legs() == 1:    
       perm = tuple(perm) + (len(perm),)
    elif state.nodes[ket_id].nopen_legs() == 2:
        perm = tuple(perm) + (len(perm), len(perm) + 1)   
    T = np.transpose(state.tensors[ket_id], perm)
    state.tensors[ket_id] = T
    state.nodes[ket_id].link_tensor(T)
    if state.nodes[ket_id].is_root():
        state.nodes[ket_id].children = neighbours
    else:
        state.nodes[ket_id].children = neighbours[1:]  

def convert_sites_and_nodes(input_list):
    converted_list = []
    
    for item in input_list:
        if item.startswith("Site"):
            converted_item = item.replace("Site", "Node")
        elif item.startswith("Node"):
            converted_item = item.replace("Node", "Site")
        else:
            converted_item = item
        
        converted_list.append(converted_item)      
    return converted_list

def adjust_bra_to_ket(vectorized_pho):
    vectorized_pho_copy = deepcopy(vectorized_pho)
    for ket_id in [node.identifier for node in vectorized_pho_copy.nodes.values() if str(node.identifier).startswith("S")]:
        if vectorized_pho_copy.nodes[ket_id].is_root():
            bra_id = ket_id.replace("Site", "Node")

            perm = list(range(vectorized_pho_copy.tensors[ket_id].ndim - 1))
            n = vectorized_pho_copy.nodes[ket_id].neighbour_index(bra_id)
            perm.pop(n)
            perm.insert(0,n)
            neighbours = np.array(vectorized_pho_copy.nodes[ket_id].neighbouring_nodes())
            neighbours = neighbours[perm].tolist()
            transpose_node_with_neighbouring_nodes(vectorized_pho_copy, ket_id, neighbours)
            neighbours = convert_sites_and_nodes(neighbours)
            transpose_node_with_neighbouring_nodes(vectorized_pho_copy, bra_id, neighbours)
        else:
            bra_id = ket_id.replace("Site", "Node")               
            neighbours = vectorized_pho_copy.nodes[ket_id].neighbouring_nodes()
            neighbours = convert_sites_and_nodes(neighbours)
            transpose_node_with_neighbouring_nodes(vectorized_pho_copy, bra_id, neighbours) 
    return vectorized_pho_copy        

def split_root_qr(psi):
    psi_copy = deepcopy(psi)    
    root_id = psi_copy.root_id
    root_node , root_tensor = psi_copy[root_id]

    open_leg_idx = root_node.open_legs[0]
    perm = list(range(root_tensor.ndim))
    perm.pop(open_leg_idx)
    q_legs = tuple(perm)
    r_legs = (open_leg_idx,)
    correctly_order = q_legs + r_legs == list(range(len(q_legs) + len(r_legs)))
    matrix = tensor_matricization(root_tensor, q_legs, r_legs,
                                  correctly_ordered=correctly_order)
    q, r = np.linalg.qr(matrix)
    shape = root_tensor.shape
    q_shape = _determine_tensor_shape(shape, q, q_legs, output=True)
    r_shape = _determine_tensor_shape(shape, r, r_legs, output=False)
    q = np.reshape(q, q_shape)
    r = np.reshape(r, r_shape)
    psi_copy.tensors[root_id] = q
    psi_copy.nodes[root_id].link_tensor(q)
    new_child = Node(tensor = r , identifier = root_id + "_R")
    psi_copy.add_child_to_parent(new_child , r , 0 , root_id , q.ndim -1)

    return psi_copy
                       
def devectorize_pho(vectorized_pho , connections): 
    vectorized_pho_copy = deepcopy(vectorized_pho)
    vectorized_pho_copy = adjust_bra_to_ket(vectorized_pho_copy)   
    n = int(np.sqrt(len(vectorized_pho_copy.nodes) // 2)) 
    
    sites = {
        (i, j): (Node(   tensor = vectorized_pho_copy.tensors[f"Site({i},{j})"], 
                            identifier= f"Vertex({i},{j})"    )    , 
                            vectorized_pho_copy.tensors[f"Site({i},{j})"])
                for i in range(n) for j in range(n)
    }

    ket = TreeTensorNetworkState()
    ket.add_root(sites[(0, 0)][0], sites[(0, 0)][1])
    for (parent, child, parent_leg, child_leg) in connections:
        parent_id = f"Vertex({parent[0]},{parent[1]})"
        ket.add_child_to_parent(sites[child][0], sites[child][1], child_leg, parent_id, parent_leg)

    sites = {
        (i, j): (Node(   tensor = vectorized_pho_copy.tensors[f"Node({i},{j})"], 
                            identifier= f"Vertex({i},{j})"    )    , 
                            vectorized_pho_copy.tensors[f"Node({i},{j})"])
                for i in range(n) for j in range(n)
    }

    bra = TreeTensorNetworkState()
    bra.add_root(sites[(0, 0)][0], sites[(0, 0)][1])

    for (parent, child, parent_leg, child_leg) in connections:
        parent_id = f"Vertex({parent[0]},{parent[1]})"
        bra.add_child_to_parent(sites[child][0], sites[child][1], child_leg, parent_id, parent_leg)  

    ket_QR = split_root_qr(ket)
    bra_QR = split_root_qr(bra)

    return ket_QR , bra_QR 

def devectorize_pho_1d(vectorized_pho , connections , n): 
    vectorized_pho_copy = deepcopy(vectorized_pho)
    vectorized_pho_copy = adjust_bra_to_ket(vectorized_pho_copy)   
        
    sites = {
        (0, j): (Node(   tensor = vectorized_pho_copy.tensors[f"Site({0},{j})"], 
                            identifier= f"Vertex({0},{j})"    )    , 
                            vectorized_pho_copy.tensors[f"Site({0},{j})"])
                            for j in range(n)
    }

    ket = TreeTensorNetworkState()
    ket.add_root(sites[(0, 0)][0], sites[(0, 0)][1])
    
    for (parent, child, parent_leg, child_leg) in connections:
        parent_id = f"Vertex({parent[0]},{parent[1]})"
        ket.add_child_to_parent(sites[child][0], sites[child][1], child_leg, parent_id, parent_leg)

    sites = {
        (0, j): (Node(   tensor = vectorized_pho_copy.tensors[f"Node({0},{j})"], 
                            identifier= f"Vertex({0},{j})"    )    , 
                            vectorized_pho_copy.tensors[f"Node({0},{j})"])
                         for j in range(n)
    }

    bra = TreeTensorNetworkState()
    bra.add_root(sites[(0, 0)][0], sites[(0, 0)][1])

    for (parent, child, parent_leg, child_leg) in connections:
        parent_id = f"Vertex({parent[0]},{parent[1]})"
        bra.add_child_to_parent(sites[child][0], sites[child][1], child_leg, parent_id, parent_leg)  

    ket_QR = split_root_qr(ket)
    bra_QR = split_root_qr(bra)

    return ket_QR , bra_QR 

def contract_ttno_with_ttn(ttno: TTNO, ttn: TreeTensorNetworkState) -> TreeTensorNetworkState:
    if not isinstance(ttno, TTNO):
        raise TypeError("The first argument must be a TTNO.")
    contracted_ttn = deepcopy(ttn)
    for node_id in list(ttn.nodes.keys()) :
        ttno_neighbours = ttno.nodes[node_id].neighbouring_nodes()
        element_map = {elem: i for i, elem in enumerate(ttno_neighbours)}
        ttn_neighbours = ttn.nodes[node_id].neighbouring_nodes()
        permutation = tuple(element_map[elem] for elem in ttn_neighbours)
        nneighbours = ttno.nodes[node_id].nneighbours()
        ttno_tensor = ttno.tensors[node_id].transpose(permutation +(nneighbours,nneighbours+1)) 
        T = contract_tensors_ttno_with_ttn(ttno_tensor, ttn.tensors[node_id])   
        contracted_ttn.tensors[node_id] = T
        contracted_ttn.nodes[node_id].link_tensor(T)        
    return contracted_ttn     

def contract_tensors_ttno_with_ttn(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    C = np.tensordot( B, A, axes=((-1,), (A.ndim-1,)))
    perm = []
    for i in range(((C.ndim-1)//2)): 
        perm.append(i + ((C.ndim-1)//2) )
        perm.append(i)   
    perm.append((C.ndim-1))
    C = np.transpose(C, tuple(perm))
    original_shape = C.shape
    new_shape = []
    for i in range(0, len(original_shape)-2, 2):
        new_shape += (original_shape[i] * original_shape[i + 1],)
    new_shape += (original_shape[-1],)
    C = C.reshape(new_shape)   
    return C

def normalize_ttn_Lindblad_1(vectorized_pho , orth_center_id_1, orth_center_id_2, connections): 
    pho_normalized_str = deepcopy(vectorized_pho)
    pho_normalized = deepcopy(vectorized_pho)
    pho_normalized = adjust_bra_to_ket(pho_normalized)
    pho_normalized.canonical_form_twosite( orth_center_id_1, orth_center_id_2 , mode = SplitMode.REDUCED)
    pho_normalized = adjust_ttn1_structure_to_ttn2(pho_normalized , pho_normalized_str)
    
    ket , bra = devectorize_pho(pho_normalized ,connections)
    #ket , bra = devectorize_pho_1d(pho_normalized ,connections , 5)
    bra = bra.conjugate()
    norm = contract_two_ttns(bra , ket)
    norm = np.sqrt(norm)

    T = pho_normalized.tensors[orth_center_id_1].astype(complex)
    T /= norm
    pho_normalized.tensors[orth_center_id_1] = T
    pho_normalized.nodes[orth_center_id_1].link_tensor(T)

    T = pho_normalized.tensors[orth_center_id_2].astype(complex)
    T /= norm
    pho_normalized.tensors[orth_center_id_2] = T
    pho_normalized.nodes[orth_center_id_2].link_tensor(T)
    return pho_normalized

def normalize_ttn_Lindblad_11(vectorized_pho , connections): 
    pho_normalized_str = deepcopy(vectorized_pho)
    pho_normalized = deepcopy(vectorized_pho)
    pho_normalized = adjust_bra_to_ket(pho_normalized)

    ket , bra = devectorize_pho(pho_normalized ,connections)
    #ket , bra = devectorize_pho_1d(pho_normalized ,connections , 5)

    bra = bra.conjugate()
    norm = contract_two_ttns(bra , ket)
    n = len(pho_normalized.nodes) // 2
    norm = np.sqrt(norm ** (1/n))
    for ket_id in [node.identifier for node in pho_normalized.nodes.values() if str(node.identifier).startswith("S")]:
        bra_id = ket_id.replace('Site', 'Node')
        T = pho_normalized.tensors[ket_id].astype(complex)
        T /= norm
        pho_normalized.tensors[ket_id] = T
        pho_normalized.nodes[ket_id].link_tensor(T)

        T = pho_normalized.tensors[bra_id].astype(complex)
        T /= norm  
        pho_normalized.tensors[bra_id] = T
        pho_normalized.nodes[bra_id].link_tensor(T)

    pho_normalized = adjust_ttn1_structure_to_ttn2(pho_normalized , pho_normalized_str)
    return pho_normalized

def expectation_value_Lindblad_1(vectorized_pho: TreeTensorNetworkState,
                               connections: list,
                               operator: TTNO) -> complex:
    ket , bra = devectorize_pho(vectorized_pho ,connections)
    #ket , bra = devectorize_pho_1d(vectorized_pho ,connections , 5)

    bra = adjust_ttn1_structure_to_ttn2(bra, ket)
    #bra = bra.conjugate()

    ## update "Vertex(0,0)_R" according to the new bond dimension ##
    new_dim = ket.tensors[ket.root_id + "_R"].shape[-1]
    op_tensor = np.eye(new_dim).reshape((1, new_dim, new_dim))
    operator.tensors[operator.root_id + "_R"] = op_tensor
    operator.nodes[operator.root_id + "_R"].link_tensor(op_tensor)

    operator = adjust_ttno_structure_to_ttn(operator, ket)
    op_ket = contract_ttno_with_ttn(operator, ket)
    return complex(contract_two_ttns(op_ket , bra))

def bra_ket(vectorized_pho: TreeTensorNetworkState,
            connections: list) -> complex:
    ket , bra = devectorize_pho(vectorized_pho ,connections)
    #ket , bra = devectorize_pho_1d(vectorized_pho ,connections , 5)
    
    bra = adjust_ttn1_structure_to_ttn2(bra, ket)
    #bra = bra.conjugate()
    return complex(contract_two_ttns(bra , ket))

import cmath
def expectation_value_Lindblad_2(vectorized_pho: TreeTensorNetworkState,
                                connections: list,
                                operator: TTNO) -> complex:
    """
    Computes the Expecation value of a state with respect to an operator.

    The operator is given as a TTNO and the state as a TTNS. The expectation
    is obtained by "sandwiching" the operator between the state and its complex
    conjugate: <psi|H|psi>.

    Args:
        state (TreeTensorNetworkState): The TTNS representing the state.
        operator (TTNO): The TTNO representing the Operator.

    Returns:
        complex: The expectation value.
    """
    # compute normalization factor
    bra_ket = ptn.bra_ket(vectorized_pho, connections)
    n = len(vectorized_pho.nodes) // 2
    norm_factor = 1 / np.sqrt(bra_ket ** (1/n))
    #norm_factor = np.array(1)
    #vectorized_pho.canonical_form("Node(1,1)", mode = SplitMode.REDUCED)


    ket , bra = ptn.devectorize_pho(vectorized_pho , connections)
    #ket , bra = devectorize_pho_1d(vectorized_pho ,connections , 5)
    #bra = bra.conjugate()

    ## update "Vertex(0,0)_R" according to the new bond dimension ##
    new_dim = ket.tensors[ket.root_id + "_R"].shape[-1]
    op_tensor = np.eye(new_dim).reshape((1, new_dim, new_dim))
    operator.tensors[operator.root_id + "_R"] = op_tensor
    operator.nodes[operator.root_id + "_R"].link_tensor(op_tensor)


    dictionary = PartialTreeCachDict()
    # Getting a linear list of all identifiers
    computation_order = ket.linearise()
    errstr = "The last element of the linearisation should be the root node."
    assert computation_order[-1] == ket.root_id, errstr
    assert computation_order[-1] == operator.root_id, errstr
    for node_id in computation_order[:-1]: # The last one is the root node
        node = ket.nodes[node_id]
        parent_id = node.parent
        # Due to the linearisation the children should already be contracted.
        block = contract_any_Lindblad(node_id,
                                      parent_id,
                                      bra, 
                                      ket,
                                      operator,
                                      dictionary,
                                      norm_factor)
        dictionary.add_entry(node_id,parent_id,block)
        # The children contraction results are not needed anymore.
        children = node.children
        for child_id in children:
            dictionary.delete_entry(child_id,node_id)
    # Now everything remaining is contracted into the root tensor.
    bra = adjust_ttn1_structure_to_ttn2(bra, ket)
    return complex(contract_node_with_environment_Lindblad(ket.root_id,
                                                           bra,
                                                           ket,
                                                           operator,
                                                           dictionary,
                                                           norm_factor))

def contract_any_Lindblad(node_id: str, 
                          next_node_id: str,
                          bra: TreeTensorNetworkState,
                          ket: TreeTensorNetworkState,
                          operator: TTNO,
                          dictionary: PartialTreeCachDict,
                          norm_factor) -> np.ndarray:
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
        operator (TTNO): The TTNO representing the Hamiltonian.
        dictionary (PartialTreeCachDict): The dictionary containing the
            already contracted subtrees.
        
    Returns:
        np.ndarray: The contracted tensor.
    """
    node = ket.nodes[node_id]
    #if not node_id == "Vertex(1,1)":
    #   norm_factor = np.array(1)
    if node.is_leaf():
        return contract_leaf_Lindblad(node_id, 
                                      bra,
                                      ket, 
                                      operator,
                                      norm_factor)
    #norm_factor = np.array(1)
    return contract_subtrees_using_dictionary_Lindblad(node_id,
                                                       next_node_id,
                                                       bra,
                                                        ket,
                                                       operator,
                                                       dictionary,
                                                       norm_factor)

def contract_leaf_Lindblad(node_id: str,
                  bra: TreeTensorNetworkState,
                  ket: TreeTensorNetworkState,
                  operator: TTNO,
                  norm_factor) -> np.ndarray:
    """
    Contracts for a leaf node the state, operator and conjugate state tensors.

    If the current subtree starts at a leaf, only the three tensors
    corresponding to that site must be contracted. Furthermore, the retained
    legs must point towards the leaf's parent.

    Args:
        node_id (str): Identifier of the leaf node
        state (TreeTensorNetworkState): The TTNS representing the state.
        operator (TTNO): The TTNO representing the Hamiltonian.

    Returns:
        np.ndarray: The contracted partial tree::
    
                     _____
           2    ____|     |
                    |  A* |
                    |_____|
                       |
                       |1
                     __|__
           1    ____|     |
                  0 |  H  |
                    |_____|
                       |2
                       |
                     __|__
           0    ____|     |
                    |  A  |
                    |_____|
        
    """
    bra = adjust_ttn1_structure_to_ttn2(bra, ket)
    ket_node, ket_tensor = ket[node_id]
    _ , bra_tensor = bra[node_id]
    ket_tensor = ket_tensor * norm_factor
    bra_tensor = bra_tensor * norm_factor
    ham_node, ham_tensor = operator[node_id]
    bra_ham = np.tensordot(ham_tensor, bra_tensor,
                           axes=(_node_operator_output_leg(ham_node),
                                 _node_state_phys_leg(ket_node)))
    bra_ham_ket = np.tensordot(ket_tensor, bra_ham,
                               axes=(_node_state_phys_leg(ket_node),
                                     _node_operator_input_leg(ham_node)-1))
    return bra_ham_ket
    
def contract_subtrees_using_dictionary_Lindblad(node_id: str, 
                                                ignored_node_id: str,
                                                bra: TreeTensorNetworkState,
                                                ket: TreeTensorNetworkState,
                                                operator: TTNO,
                                                dictionary: PartialTreeCachDict,
                                                norm_factor) -> np.ndarray:
    """
    Contracts a node with all its subtrees except for one.

    All subtrees except for one are already contracted and stored in the
    dictionary. The one that is not contracted is the one that the remaining
    legs point towards.

    Args:
        node_id (str): Identifier of the node.
        ignored_node_id (str): Identifier of the node to which the remaining
            legs should point.
        state (TreeTensorNetworkState): The TTNS representing the state.
        operator (TTNO): The TTNO representing the operator.
        dictionary (PartialTreeCachDict): The dictionary containing the
            already contracted subtrees.

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
    ket_node, ket_tensor = ket[node_id]
    ket_tensor = ket_tensor * norm_factor
    tensor = contract_all_but_one_neighbour_block_to_ket(ket_tensor,
                                                         ket_node,
                                                         ignored_node_id,
                                                         dictionary)
    op_node, op_tensor = operator[node_id]
    tensor = contract_operator_tensor_ignoring_one_leg(tensor,
                                                       ket_node,
                                                       op_tensor,
                                                       op_node,
                                                       ignored_node_id)
    _ , bra_tensor = bra[node_id]
    bra_tensor = bra_tensor * norm_factor
    return contract_bra_tensor_ignore_one_leg(bra_tensor,
                                              tensor,
                                              ket_node,
                                              ignored_node_id)    
 
def contract_node_with_environment_Lindblad(node_id: str,
                                   bra: TreeTensorNetworkState,
                                   ket: TreeTensorNetworkState,
                                   operator: TTNO,
                                   dictionary: PartialTreeCachDict,
                                   norm_factor) -> np.ndarray:
    """
    Contracts a node with its environment.

    Assumes that all subtrees starting from this node are already contracted
    and the results stored in the dictionary.

    Args:
        node_id (str): The identifier of the node.
        state (TreeTensorNetworkState): The TTNS representing the state.
        operator (TTNO): The TTNO representing the Hamiltonian.
        dictionary (PartialTreeCacheDict): The dictionary containing the
         already contracted subtrees.
    
    Returns:
        np.ndarray: The resulting tensor. A and B are the tensors in state1 and
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
    bra = adjust_ttn1_structure_to_ttn2(bra, ket)
    ket_node, ket_tensor = ket[node_id]
    ket_tensor = ket_tensor * norm_factor
    ket_neigh_block = contract_all_neighbour_blocks_to_ket(ket_tensor,
                                                           ket_node,
                                                           dictionary)
    op_node, op_tensor = operator[node_id]
    state_legs, ham_legs = get_equivalent_legs(ket_node, op_node)
    ham_legs.append(_node_operator_input_leg(op_node))
    block_legs = list(range(1,2*ket_node.nneighbours(),2))
    block_legs.append(0)
    kethamblock = np.tensordot(ket_neigh_block, op_tensor,
                               axes=(block_legs, ham_legs))
    _ , bra_tensor = bra[node_id]
    bra_tensor = bra_tensor * norm_factor
    state_legs.append(len(state_legs))
    return np.tensordot(bra_tensor, kethamblock,
                        axes=(state_legs,state_legs))

def cehck_two_ttn_compatibility(ttn1, ttn2):
    for nodes in ttn1.nodes:
        legs = get_equivalent_legs(ttn1.nodes[nodes], ttn2.nodes[nodes])
        assert legs[0] == legs[1]

def adjust_ttn1_structure_to_ttn2(ttn1, ttn2):
    try:
        cehck_two_ttn_compatibility(ttn1, ttn2)
        return ttn1
    except AssertionError:    
        ttn3 = deepcopy(ttn2)
        orth_center = ttn1.orthogonality_center_id
        for node_id in ttn3.nodes:
            ttn1_neighbours = ttn1.nodes[node_id].neighbouring_nodes()
            element_map = {elem: i for i, elem in enumerate(ttn1_neighbours)}
            ttn1_neighbours = ttn2.nodes[node_id].neighbouring_nodes()
            permutation = tuple(element_map[elem] for elem in ttn1_neighbours)
            nneighbours = ttn2.nodes[node_id].nneighbours()
            ttn1_tensor = ttn1.tensors[node_id].transpose(permutation + (nneighbours,))
            
            ttn3.tensors[node_id] = ttn1_tensor
            ttn3.nodes[node_id].link_tensor(ttn1_tensor)
            ttn3.orthogonality_center_id = orth_center    
        return ttn3       

def adjust_ttno_structure_to_ttn(ttno, ttn):
    try: 
        cehck_two_ttn_compatibility(ttno, ttn)
        return ttno
    except AssertionError:    
        ttno_copy = deepcopy(ttno)
        for node_id in ttno_copy.nodes:
            ttno_neighbours = ttno.nodes[node_id].neighbouring_nodes()
            element_map = {elem: i for i, elem in enumerate(ttno_neighbours)}
            ttn1_neighbours = ttn.nodes[node_id].neighbouring_nodes()
            permutation = tuple(element_map[elem] for elem in ttn1_neighbours)
            nneighbours = ttn.nodes[node_id].nneighbours()
            ttno_tensor = ttno.tensors[node_id].transpose(permutation + (nneighbours,) + (nneighbours + 1,))
            ttno_copy.tensors[node_id] = ttno_tensor
            ttno_copy.nodes[node_id].link_tensor(ttno_tensor)
            ttn_neighbours = ttn.nodes[node_id].neighbouring_nodes()
            if ttno_copy.nodes[node_id].is_root():
                ttno_copy.nodes[node_id].children = ttn_neighbours
            else:
                ttno_copy.nodes[node_id].children = ttn_neighbours[1:] 
        return ttno_copy 

def normalize_ttn_Lindblad_A(vectorized_pho , connections) : 
    pho_normalized_str = deepcopy(vectorized_pho)
    pho_normalized = deepcopy(vectorized_pho)
    norm = bra_ket(pho_normalized , connections)
    n = len(pho_normalized.nodes) // 2
    norm = np.sqrt(norm ** (1/n))
    pho_normalized = adjust_ttn1_structure_to_ttn2(pho_normalized , pho_normalized_str)

    for ket_id in [node.identifier for node in pho_normalized.nodes.values() if str(node.identifier).startswith("S")]:
        bra_id = ket_id.replace('Site', 'Node')
        T = pho_normalized.tensors[ket_id].astype(complex)
        T /= norm
        pho_normalized.tensors[ket_id] = T
        pho_normalized.nodes[ket_id].link_tensor(T)

        T = pho_normalized.tensors[bra_id].astype(complex)
        T /= norm
        pho_normalized.tensors[bra_id] = T
        pho_normalized.nodes[bra_id].link_tensor(T)

    return pho_normalized

def normalize_ttn_Lindblad_X(vectorized_pho , orth_center_id_1 ,connections): 
    pho_normalized_str = deepcopy(vectorized_pho)
    pho_normalized = deepcopy(vectorized_pho)
    pho_normalized.canonical_form( orth_center_id_1 , mode = SplitMode.REDUCED)
    pho_normalized = adjust_ttn1_structure_to_ttn2(pho_normalized , pho_normalized_str)
    norm = bra_ket(pho_normalized , connections)
    
    T = pho_normalized.tensors[orth_center_id_1].astype(complex)
    T /= norm
    pho_normalized.tensors[orth_center_id_1] = T
    pho_normalized.nodes[orth_center_id_1].link_tensor(T)
    return pho_normalized

def normalize_ttn_Lindblad_XX(vectorized_pho , orth_center_id_1 , orth_center_id_2 , connections): 
    pho_normalized_str = deepcopy(vectorized_pho)
    pho_normalized = deepcopy(vectorized_pho)
    pho_normalized.canonical_form_twosite( orth_center_id_1, orth_center_id_2,mode = SplitMode.REDUCED)
    pho_normalized = adjust_ttn1_structure_to_ttn2(pho_normalized , pho_normalized_str)
    norm = ptn.bra_ket(pho_normalized, connections)
    T = pho_normalized.tensors[orth_center_id_1].astype(complex)
    T /= np.sqrt(norm)
    pho_normalized.tensors[orth_center_id_1] = T
    pho_normalized.nodes[orth_center_id_1].link_tensor(T)

    T = pho_normalized.tensors[orth_center_id_2].astype(complex)
    T /= np.sqrt(norm)
    pho_normalized.tensors[orth_center_id_2] = T
    pho_normalized.nodes[orth_center_id_2].link_tensor(T)
    return pho_normalized