from ..core.node import Node
from ..contractions.state_state_contraction import contract_two_ttns
from ..ttno import TTNO
from copy import deepcopy
from ..util.tensor_util import tensor_matricization
from ..util.tensor_splitting import _determine_tensor_shape , SplitMode
from ..ttns import TreeTensorNetworkState
import numpy as np


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

def adjust_ttn1_structure_to_ttn2(ttn1, ttn2):
    """
    Adjusts the structure of ttn1 to match the structure of ttn2.

    Args:
        ttn1 (TTN): The original Tensor Train Network.
        ttn2 (TTN): The target Tensor Train Network.

    Returns:
        ttn3 (TTN): The adjusted ttn1 with the structure of ttn2.
    """
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
        
    sites = {
        (i, j): (Node(   tensor = vectorized_pho_copy.tensors[f"Site({i},{j})"], 
                            identifier= f"Vertex({i},{j})"    )    , 
                            vectorized_pho_copy.tensors[f"Site({i},{j})"])
                for i in range(3) for j in range(3)
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
                for i in range(3) for j in range(3)
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


def normalize_ttn_Lindblad(vectorized_pho , orth_center_id, connections): 
    pho_normalized_str = deepcopy(vectorized_pho)
    pho_normalized = deepcopy(vectorized_pho)
    pho_normalized.canonical_form(orth_center_id, mode = SplitMode.REDUCED) 
    pho_normalized = adjust_ttn1_structure_to_ttn2(pho_normalized , pho_normalized_str)
    
    ket , bra = devectorize_pho(vectorized_pho ,connections)
    bra = bra.conjugate()
    norm = contract_two_ttns(bra , ket)
    print(norm , np.abs(norm))
    T = pho_normalized.tensors[orth_center_id].astype(complex)
    T /= norm
    pho_normalized.tensors[orth_center_id] = T
    pho_normalized.nodes[orth_center_id].link_tensor(T)
    return pho_normalized

def expectation_value_Lindblad(vectorized_pho: TreeTensorNetworkState,
                               connections: list,
                               operator: TTNO) -> complex:
    ket , bra = devectorize_pho(vectorized_pho ,connections)
    bra = bra.conjugate()
    # update "Vertex(0,0)_R" according to the new bond dimension
    new_dim = ket.tensors[ket.root_id + "_R"].shape[-1]
    op_tensor = np.eye(new_dim).reshape((1, new_dim, new_dim))
    operator.tensors[operator.root_id + "_R"] = op_tensor
    operator.nodes[operator.root_id + "_R"].link_tensor(op_tensor)
    # compute normalization factor 

    op_ket = contract_ttno_with_ttn(operator, ket)
    return contract_two_ttns(bra , op_ket)