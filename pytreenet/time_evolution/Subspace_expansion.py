import numpy as np
from copy import deepcopy
from typing import Tuple, Any, Dict, List
from scipy.sparse.linalg import eigsh


from ..util.tensor_splitting import (SplitMode , truncated_tensor_svd)
from ..ttns import TreeTensorNetworkState
from ..util.tensor_splitting import SVDParameters
from ..util.tensor_util import compute_transfer_tensor
from ..ttno.ttno_class import TTNO
from ..util.tensor_splitting import SVDParameters , ContractionMode
from ..time_evolution.time_evo_util.update_path import TDVPUpdatePathFinder
from ..core.leg_specification import LegSpecification
from copy import copy
from enum import Enum


def Krylov_basis(ttn: TreeTensorNetworkState, 
                 ttno: TTNO, 
                 num_vecs: int, 
                 tau: float, 
                 SVDParameters : SVDParameters):
   ttn_copy = deepcopy(ttn)
   ttn_structure = deepcopy(ttn)
   ttno_copy = deepcopy(ttno)

   I = TTNO.Identity(ttno_copy)
   ttno_copy.multiply_const(-1j*tau) 
   ttno_copy = sum_two_ttns(I, ttno_copy)
   
   if ttn_copy.orthogonality_center_id is not TDVPUpdatePathFinder(ttn_copy).find_path()[0]:
        ttn_copy.canonical_form(TDVPUpdatePathFinder(ttn_copy).find_path()[0] , SplitMode.REDUCED) 
   basis_list = [ttn_copy]
   for _ in range(num_vecs):
      #ttno_copy = adjust_ttno_structure_to_ttn(ttno_copy,ttn_copy)
      ttn_copy = contract_ttno_with_ttn(ttno_copy,ttn_copy)
      ttn_copy.canonical_form(TDVPUpdatePathFinder(ttn_copy).find_path()[0] , SVDParameters)  
      # ttn_copy.normalize_ttn()
      basis_list.append(ttn_copy)
   return basis_list


class KrylovBasisMode(Enum):
    apply_ham = "apply_ham"
    apply_1st_order_expansion = "apply_1st_order_expansion"

def Krylov_basis2(ttn: TreeTensorNetworkState, 
                 ttno: TTNO, 
                 num_vecs: int, 
                 SVDParameters : SVDParameters):
    ttn_copy = deepcopy(ttn)
    ttno_copy = deepcopy(ttno)
    ttn_structure = deepcopy(ttn_copy)
    ttn_copy.canonical_form(TDVPUpdatePathFinder(ttn_copy).find_path()[0], SplitMode.KEEP)

    basis_list = [ttn_copy]
    for _ in range(num_vecs):
        ttn_copy = contract_ttno_with_ttn(ttno_copy,ttn_copy)
        ttn_structure = deepcopy(ttn_copy)
        ttn_copy.canonical_form(TDVPUpdatePathFinder(ttn_copy).find_path()[0],SVDParameters)
        ttn_copy.normalize_ttn() 
        basis_list.append(ttn_copy)
    return basis_list

def expand_subspace(ttn: TreeTensorNetworkState, 
                    ttno: TTNO,
                    num_vecs: int,
                    tau: float,
                    SVDParameters : SVDParameters,
                    tol,
                    mode: KrylovBasisMode ):
    if mode == KrylovBasisMode.apply_ham:
        basis = Krylov_basis2(ttn,ttno,num_vecs,SVDParameters)    
    elif mode == KrylovBasisMode.apply_1st_order_expansion:
        basis = Krylov_basis(ttn, ttno, num_vecs, tau, SVDParameters)

    ttn_copy = deepcopy(basis[0])
    for i in range(len(basis)-1):
        ttn_copy = enlarge_ttn1_bond_with_ttn2(ttn_copy, basis[i+1], tol)
        #ttn_copy.normalize_ttn()
    #ttn_copy = original_form(ttn_copy , dict1)    
    return ttn_copy

def enlarge_ttn1_bond_with_ttn2(ttn1, ttn2, tol):
   ttn1_copy = deepcopy(ttn1)
   ttn3 = deepcopy(ttn1)
   path_main = TDVPUpdatePathFinder(ttn1_copy).find_path()
   path_next = find_tdvp_orthogonalization_path(ttn1_copy,path_main) 

   for i,node_id in enumerate(path_main[:-1]): 
        next_node_id = path_next[i][0]
        index = ttn1_copy.nodes[node_id].neighbour_index(next_node_id)
        index_prime = ttn1_copy.nodes[next_node_id].neighbour_index(node_id) 
        pho_A = compute_transfer_tensor(ttn1_copy.tensors[node_id],(index,))
        pho_B = compute_transfer_tensor(ttn2.tensors[node_id],(index,))
        pho = pho_A + pho_B

        v = compute_v(pho, index, tol)

        ttn3.tensors[node_id] = v
        ttn3.nodes[node_id].link_tensor(v)

        v_legs = list(range(0,v.ndim))
        v_legs.remove(index)
        # print( ttn1_copy.tensors[node_id].shape , np.conjugate(v).shape , v_legs)
        CVd = np.tensordot(ttn1_copy.tensors[node_id] , np.conjugate(v) , (v_legs,v_legs))
        ttn1_copy.tensors[next_node_id] = absorb_matrix_into_tensor(CVd, ttn1_copy.tensors[next_node_id], (0,index_prime))
        CVd = np.tensordot(ttn2.tensors[node_id] , np.conjugate(v) , (v_legs,v_legs))
        ttn2.tensors[next_node_id] = absorb_matrix_into_tensor(CVd, ttn2.tensors[next_node_id], (0,index_prime))
        
        if ttn1_copy.orthogonality_center_id != None:
            if len(path_next[i]) > 1:
                ttn1_copy.orthogonality_center_id = path_next[i][0]

                ttn1_structure = deepcopy(ttn1_copy)
                ttn1_copy.move_orthogonalization_center(path_next[i][-1])
        if ttn2.orthogonality_center_id != None:
            if len(path_next[i]) > 1:
                ttn2.orthogonality_center_id = path_next[i][0]

                ttn2_structure = deepcopy(ttn2)
                ttn2.move_orthogonalization_center(path_next[i][-1])

   last_node_id = path_main[-1]
   ttn3.tensors[last_node_id] = ttn1_copy.tensors[last_node_id]
   ttn3.nodes[last_node_id].link_tensor(ttn1_copy.tensors[last_node_id]) 
   
   ttn1_structure = deepcopy(ttn1_copy)
   ttn3.canonical_form(path_main[0], SplitMode.REDUCED) 

   return ttn3

def compute_v(pho, index, tol):
    shape = pho.shape
    shape_prime = (np.prod(shape[:len(shape)//2]),) + (np.prod(shape[len(shape)//2:]),)
    pho = pho.reshape(shape_prime)
    #v = eig_Lanczos(pho, tol, Lanczos_threshold , k_fraction, validity_fraction, increase_fraction, max_iter)
    v = eig(pho, tol)
    v = v.T
    v = v.reshape( (v.shape[0],) + shape[len(shape)//2:])

    perm = list(range(0,v.ndim))
    perm.remove(index)
    perm = [index] + perm
    v = v.transpose(perm)
    return v



def eig_Lanczos(pho, tol, Lanczos_threshold , k_fraction, validity_fraction, increase_fraction, max_iter):
    """
    Compute eigenvectors of a matrix 'pho' corresponding to eigenvalues whose magnitudes
    exceed a given tolerance 'tol', using the Lanczos method with dynamic adjustments to 
    the number of eigenvectors computed in each iteration.

    Parameters:
    - pho (ndarray): The input matrix (should be symmetric). This represents a square matrix 
                     for which we want to compute the eigenvectors.
    - tol (float): The tolerance threshold for filtering eigenvalues by magnitude. Eigenvalues 
                   whose magnitudes are below this threshold will be ignored.
    - Lanczos_threshold (int): The minimum size of the matrix for the Lanczos method to be applied. If 
                            the matrix size is smaller than this threshold, the function will return 
                            an empty result.               
    - k_fraction (float): The fraction of the matrix size to determine the initial number 
                                of eigenvectors to compute. For example, a value of 0.6 means 
                                that the initial computation will use 60% of the total matrix size.
    - validity_fraction (float): The fraction of the initially computed eigenvectors that need to 
                                 exceed the tolerance in order to terminate the computation early.
                                 For example, 0.8 means number of significant eigenvectors must be
                                 less than 80% of the k
    - increase_fraction (float): The fraction of the matrix size by which the number of eigenvectors 
                                 will be increased in each iteration if the desired fraction of 
                                 significant eigenvalues is not reached.
    - max_iter (int): The maximum number of iterations to attempt increasing the number of eigenvectors.

    Returns:
    - v (ndarray): The matrix of eigenvectors corresponding to eigenvalues whose magnitudes exceed the 
                   tolerance. The size of this matrix will vary based on the number of significant 
                   eigenvalues.
    """
    
    # Check if the matrix 'pho' has a sufficient number of rows/columns
    if pho.shape[0] > Lanczos_threshold:
        print("Lanczos method" , pho.shape[0])
        # Determine the initial number of eigenvectors to compute
        k = int(pho.shape[0] * k_fraction) + 1

        # Iterate to increase k and recompute eigenvectors
        for iter_count in range(max_iter):
            k = min(k, pho.shape[0])  # Ensure k does not exceed the matrix size
            #print(f"Iteration {iter_count}: k = {k}")
            
            # Compute eigenvalues and eigenvectors using the Lanczos method
            w, v = eigsh(pho, k=k, which='LM', tol=tol)

            # Compute the magnitudes of the eigenvalues and sort them
            magnitudes = np.abs(w)
            sorted_indices = np.argsort(magnitudes)[::-1]

            # Sort the eigenvalues and eigenvectors in descending order of magnitude
            w = w[sorted_indices]
            v = v[:, sorted_indices]
            
            print("v.shape" , v.shape)

            # Find indices of eigenvalues that are significant (above tolerance)
            valid_indices = np.where(magnitudes > tol)[0]
            #print(f"Number of valid eigenvectors: {len(valid_indices)}")

            # If no significant eigenvalues are found, return the first eigenvector
            if len(valid_indices) == 0:
                v = np.reshape(v[:, 1], (v.shape[0], 1))
                print(f"len(valid_indices) = 0")
                return v
            
            # If a sufficient fraction of significant eigenvalues is found, return them
            if len(valid_indices) < int(k * validity_fraction) or k == pho.shape[0]:
                v = v[:, valid_indices]
                #print(f"2 Eigenvectors corresponding to significant eigenvalues: {v.shape}")
                return v
            
            
            # Increase k for the next iteration
            k += int(pho.shape[0] * increase_fraction)
            

        # If the loop completes, return the most recent set of eigenvectors
        # print(f"3 Eigenvectors corresponding to significant eigenvalues: {v.shape}")    
        return v
    else:
        return eig(pho, tol)


def eig(pho, tol):
    # Check if pho is an empty matrix
    #print(f"pho matrix shape: {pho.shape}")
        
    w, v = np.linalg.eig(pho)
        
    magnitudes = np.abs(w)
    sorted_indices = np.argsort(magnitudes)[::-1]
        
    w = w[sorted_indices]
    v = v[:, sorted_indices]
    
    k = np.sum(magnitudes > tol)
    #print(f"Significant eigenvalues (magnitude > tol): {k}")
    
    v = v[:, :k]
    #print(f"Eigenvectors : {v.shape}")
    return v




def absorb_matrix_into_tensor(A, B, axes):
    C = np.tensordot(A, B, axes)
    perm = tuple(list(range(1,axes[1]+1))) + (0,) + tuple(list(range(axes[1]+1, B.ndim)))
    return C.transpose(perm)

def find_tdvp_orthogonalization_path(ttn,update_path):
    orthogonalization_path = []
    for i in range(len(update_path)-1):
        sub_path = ttn.path_from_to(update_path[i], update_path[i+1])
        orthogonalization_path.append(sub_path[1::])
    return orthogonalization_path


def sum_two_ttns(ttn1: TreeTensorNetworkState, ttn2: TreeTensorNetworkState):
    """
    Adds two tree tensor networks into one. The two tree tensor networks
    should have the same structure, i.e. the same nodes and edges.
    
    Args:
        self (TreeTensorNetworkState): The first tree tensor network.
        ttn2 (TreeTensorNetworkState): The second tree tensor network.
    
    Returns:
        TreeTensorNetworkState: The resulting tree tensor network.
    """
    path = TDVPUpdatePathFinder(ttn1).find_path()

    ttn3 = deepcopy(ttn1)
    for node_id in path :
        if isinstance(ttn1, TreeTensorNetworkState) and isinstance(ttn2, TreeTensorNetworkState):
            ttn2_neighbours = ttn2.nodes[node_id].neighbouring_nodes()
            element_map = {elem: i for i, elem in enumerate(ttn2_neighbours)}
            ttn1_neighbours = ttn1.nodes[node_id].neighbouring_nodes()
            permutation = tuple(element_map[elem] for elem in ttn1_neighbours)
            nneighbours = ttn1.nodes[node_id].nneighbours()
            ttn2_tensor = ttn2.tensors[node_id].transpose(permutation +(nneighbours,))           
            T = sum_tensors_ttn(ttn1.tensors[node_id], ttn2_tensor)
        if isinstance(ttn1, TTNO) and isinstance(ttn2, TTNO):
           T = sum_tensors_ttno(ttn1.tensors[node_id], ttn2.tensors[node_id])   
        ttn3.tensors[node_id] = T
        ttn3.nodes[node_id].link_tensor(T)        
    return ttn3 

def sum_tensors_ttn(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    assert A.shape[-1] == B.shape[-1]
    delta = delta_tensor(A.shape[-1])
    C = np.tensordot(A, delta, axes=((-1,), (0,)))
    T = np.tensordot(B, C, axes=((-1,), (-1,)))
    perm = []
    for i in range(((T.ndim-1)//2)):  
        perm.append(i + ((T.ndim-1)//2) )       
        perm.append(i)
    perm.append(T.ndim-1)    
    T = np.transpose(T, tuple(perm))
    original_shape = T.shape
    new_shape = []
    for i in range(0, len(original_shape)-1, 2):
        new_shape += (original_shape[i] * original_shape[i + 1],)
    new_shape += (original_shape[-1],)
    T = np.reshape(T, new_shape)
    return T

def sum_tensors_ttno(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    delta = delta_tensor(A.shape[-1])
    C = np.tensordot(A, delta, axes=((-1,), (0,)))
    T = np.tensordot(C, B, axes=((-1,), (-1,)))
    K = np.tensordot(T, delta, axes=((A.ndim-2,-1), (0,1))) 
    perm = []
    for i in range(((K.ndim-1)//2)): 
        perm.append(i)   
        perm.append(i + ((T.ndim-1)//2) )
    perm.append(T.ndim-2)
    perm.append(A.ndim-2) 
    K = np.transpose(K, tuple(perm))
    original_shape = K.shape
    new_shape = []
    for i in range(0, len(original_shape)-2, 2):
        new_shape += (original_shape[i] * original_shape[i + 1],)
    new_shape += (original_shape[-1],original_shape[-1])
    K = K.reshape(new_shape)
    return K

def contract_ttno_with_ttno(ttno1: TTNO, ttno2: TTNO) -> TTNO:
    if not isinstance(ttno1, TTNO) or not isinstance(ttno2, TTNO):
        raise TypeError("The arguments must be TTNOs.")
    expanded_ttno = deepcopy(ttno1)
    path = TDVPUpdatePathFinder(ttno1).find_path()
    for node_id in path :
        T = contract_tensors_ttno_with_ttno(ttno1.tensors[node_id], ttno2.tensors[node_id])
        expanded_ttno.tensors[node_id] = T
        expanded_ttno.nodes[node_id].link_tensor(T)        
    return expanded_ttno

def contract_ttno_with_ttn(ttno: TTNO, ttn: TreeTensorNetworkState) -> TreeTensorNetworkState:
    if not isinstance(ttno, TTNO):
        raise TypeError("The first argument must be a TTNO.")
    expanded_ttn = deepcopy(ttn)
    for node_id in list(ttn.nodes.keys()) :
        ttno_neighbours = ttno.nodes[node_id].neighbouring_nodes()
        element_map = {elem: i for i, elem in enumerate(ttno_neighbours)}
        ttn_neighbours = ttn.nodes[node_id].neighbouring_nodes()
        permutation = tuple(element_map[elem] for elem in ttn_neighbours)
        nneighbours = ttno.nodes[node_id].nneighbours()
        ttno_tensor = ttno.tensors[node_id].transpose(permutation +(nneighbours,nneighbours+1)) 
        T = contract_tensors_ttno_with_ttn(ttno_tensor, ttn.tensors[node_id])   
        expanded_ttn.tensors[node_id] = T
        expanded_ttn.nodes[node_id].link_tensor(T)        
    return expanded_ttn     

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

def kronecker_product_ttno_with_ttno(ttno1: TTNO, ttno2: TTNO) -> TTNO:
    if not isinstance(ttno1, TTNO) or not isinstance(ttno2, TTNO):
        raise TypeError("The arguments must be TTNOs.")
    ttno3 = deepcopy(ttno1)
    path = TDVPUpdatePathFinder(ttno1).find_path()
    for node_id in path:
        T = np.kron(ttno1.tensors[node_id], ttno2.tensors[node_id])
        ttno3.tensors[node_id] = T
        ttno3.nodes[node_id].link_tensor(T)        
    return ttno3

def contract_tensors_ttno_with_ttno(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    C = np.tensordot(A, B, axes=((-1,),(-2,)))
    perm = []
    for i in range(((C.ndim-1)//2)): 
        perm.append(i) 
        perm.append(i + ((A.ndim-1)) )
    perm.append((A.ndim-2))    
    perm.append((C.ndim-1))
    C = np.transpose(C, tuple(perm))
    original_shape = C.shape 
    new_shape = []
    for i in range(0, len(original_shape)-2, 2):
        new_shape += (original_shape[i] * original_shape[i + 1],)
    new_shape += (original_shape[-1],original_shape[-1])
    C = C.reshape(new_shape)    
    return C 

def delta_tensor(dim):
    delta = np.zeros((dim, dim, dim))
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                delta[i, j, k] = 1 if i == j == k else 0
    return delta


def pad_tensor(tensor1, tensor2):
    shape1 = np.array(tensor1.shape)
    shape2 = np.array(tensor2.shape)
    
    pad_width = np.maximum(np.subtract(shape2, shape1), 0)
        
    padding = [(p,0) for p in pad_width]
    
    padded_tensor = np.pad(tensor1, padding, mode='constant', constant_values=0)
    
    return padded_tensor

def product_state(ttn, bond_dim=2):
    zero_state = deepcopy(ttn)
    zero = np.asarray([1/np.sqrt(2), 1/np.sqrt(2)])
    for node_id in zero_state.nodes.keys():
        n = zero_state.tensors[node_id].ndim - 1
        tensor = zero.reshape((1,) * n + (2,))
        T = np.pad(tensor, n*((0, bond_dim-1),) + ((0, 0),))
        zero_state.tensors[node_id] = T
        zero_state.nodes[node_id].link_tensor(T)  
    return zero_state

def pad_zero_ttn1_with_ttn2(ttn1, ttn2):
    ttn3 = deepcopy(ttn1)

    for node_id in ttn1.nodes.keys():
        padded_tensor = pad_tensor(ttn1.tensors[node_id], ttn2.tensors[node_id])

        ttn3.tensors[node_id] = padded_tensor
        ttn3.nodes[node_id].link_tensor(padded_tensor)
    
    return ttn3    

def pad_ttn_with_zeros(ttn, bond_dim):
    padded_ttn = deepcopy(ttn)
    for node_id in padded_ttn.nodes.keys():
        n = padded_ttn.tensors[node_id].ndim - 1
        tensor = padded_ttn.tensors[node_id]
        T = np.pad(tensor, n*((0, bond_dim-2),) + ((0, 0),))
        padded_ttn.tensors[node_id] = T
        padded_ttn.nodes[node_id].link_tensor(T)
    return padded_ttn
###############################################################################################


def direct_sum_ttn1_with_ttn2(ttn1 , ttn2):
    ttn3 = deepcopy(ttn1)
    for node_id in ttn2.nodes:
        A = ttn1.tensors[node_id]
        # permute ttn2 to match ttn1 structure
        ttn2_neighbours = ttn2.nodes[node_id].neighbouring_nodes()
        element_map = {elem: i for i, elem in enumerate(ttn2_neighbours)}
        ttn1_neighbours = ttn1.nodes[node_id].neighbouring_nodes()
        permutation = tuple(element_map[elem] for elem in ttn1_neighbours)
        nneighbours = ttn1.nodes[node_id].nneighbours()
        B = ttn2.tensors[node_id].transpose(permutation +(nneighbours,)) 
        # Direct sum site tensors
        shape = tuple(A.shape[i]+B.shape[i] for i in range(A.ndim-1)) + (2,)
        M = np.zeros(shape,dtype=np.complex128)
        M[tuple(  [slice(0, A.shape[i]) for i in range(A.ndim-1)]   +  [slice(None)]  )] = A
        M[tuple(  [slice(A.shape[i],  A.shape[i] + B.shape[i]) for i in range(A.ndim-1)]   +  [slice(None)]  )] = B
        ttn3.tensors[node_id] = M
        ttn3.nodes[node_id].link_tensor(M)
    return ttn3 

def max_two_neighbour_form(ttn , node_order = None): 
    state = deepcopy(ttn)
    nodes = deepcopy(state.nodes)
    dict = {}
    for node_id in nodes:
        node = state.nodes[node_id]
        
        if node_order is None:
            node_order = random_order_generator(state)

        if node.nneighbours() > 2:
            neighbour_id = node_order[node_id]
            u_legs, v_legs = build_qr_leg_specs2(node, neighbour_id)
            state.split_node_svd(node_id,svd_params = SVDParameters(max_bond_dim= np.inf, rel_tol= -np.inf, total_tol= -np.inf),
                                 u_legs = u_legs, v_legs = v_legs,
                                    u_identifier=  node_id + "_u",
                                    v_identifier=node_id,
                                    contr_mode = ContractionMode.VCONTR)
            shape = state.tensors[ node_id + "_u"].shape
            if isinstance(state , TreeTensorNetworkState):
                T = state.tensors[ node_id + "_u"].reshape(shape + (1,))
                state.tensors[ node_id + "_u"] = T 
                state.nodes[ node_id + "_u"].link_tensor(T)
            elif isinstance(state , TTNO):    
                T = state.tensors[ node_id + "_u"].reshape(shape + (1,1))
                state.tensors[ node_id + "_u"] = T 
                state.nodes[ node_id + "_u"].link_tensor(T)
            dict[node_id] = neighbour_id
    state.orthogonality_center_id = None        
    return state , dict

def random_order_generator(ttn):
    exclude_element = "Node(0,0)"
    result_dict = {}
    for ket_node in [node for node in ttn.nodes.values() if str(node.identifier).startswith("S")]:
        if ket_node.nneighbours() > 2:
            # Filter out the specific element from ket_node.children
            filtered_children = [child for child in ket_node.children if child != exclude_element and child.startswith('S')]
            if filtered_children:  # Ensure there are still children left to choose from
                result_dict[ket_node.identifier] = np.random.choice(filtered_children)
                result_dict[ket_node.identifier.replace("Site", "Node")] = result_dict[ket_node.identifier].replace("Site", "Node")
    return result_dict

def random_order_generator2(ttn):
    exclude_element = "Node(0,0)"
    result_dict = {}
    for node in ttn.nodes.values():
        if node.nneighbours() > 2:
            # Filter out the specific element from node.children
            filtered_children = [child for child in node.children if child != exclude_element]
            if filtered_children:  # Ensure there are still children left to choose from
                result_dict[node.identifier] = np.random.choice(filtered_children)
    return result_dict

def original_form(state, dict):
    ttn = deepcopy(state)
    for node_id in dict:
        if isinstance(ttn , TreeTensorNetworkState):
           T = ttn.tensors[node_id + "_u"].reshape(ttn.tensors[node_id + "_u"].shape[:-1]) 
        elif isinstance(ttn , TTNO):
           T = ttn.tensors[node_id + "_u"].reshape(ttn.tensors[node_id + "_u"].shape[:-2])   
        ttn.tensors[node_id + "_u"] = T
        ttn.nodes[node_id + "_u"].link_tensor(T)
        ttn.contract_nodes(node_id + "_u", node_id, node_id)
    return ttn

def build_qr_leg_specs2(node ,
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
    q_legs = LegSpecification(None, copy(node.children), [])
    if node.is_child_of(min_neighbour_id):
        r_legs = LegSpecification(min_neighbour_id, [], node.open_legs)
    else:
        q_legs.parent_leg = node.parent
        q_legs.child_legs.remove(min_neighbour_id)
        r_legs = LegSpecification(None, [min_neighbour_id], node.open_legs)
    if node.is_root():
        q_legs.is_root = True
    return q_legs, r_legs

def reduced_density_matrix(ttn, node_id: str, next_id : str ) -> np.ndarray: 
    working_ttn = deepcopy(ttn)
    working_ttn.canonical_form(node_id , SplitMode.FULL)
    contracted_legs = working_ttn.nodes[node_id].neighbour_index(next_id)
    reduced_density = compute_transfer_tensor(working_ttn.tensors[node_id], contracted_legs)
    return reduced_density 

def move_orth_for_path(ttn , path: List[str]):
        if len(path) == 0:
            return
        assert ttn.orthogonality_center_id == path[0]
        for i, node_id in enumerate(path[1:]):
            ttn.move_orthogonalization_center(node_id) 

def reduced_density_matrix_dict(ttn): 

    update_path = TDVPUpdatePathFinder(ttn).find_path()
    ttn = ttn.normalize_ttn(to_copy = True)
    ttn.canonical_form(update_path[0],SplitMode.FULL)

    orthogonalization_path = []
    for i in range(len(update_path)-1):
        sub_path = ttn.path_from_to(update_path[i], update_path[i+1])
        orthogonalization_path.append(sub_path[1::])

    dict = {}
    for i, node_id in enumerate(update_path):
        contracted_legs = tuple(range(ttn.tensors[node_id].ndim - 1 )) 
        if i == len(update_path)-1:
            reduced_density = compute_transfer_tensor(ttn.tensors[node_id], contracted_legs)        
            dict[node_id] = reduced_density
        elif i == 0:
            reduced_density = compute_transfer_tensor(ttn.tensors[node_id], contracted_legs)        
            dict[node_id] = reduced_density
            next_node_id = orthogonalization_path[0][0]
            move_orth_for_path(ttn,[node_id, next_node_id])
        else:
            current_orth_path = orthogonalization_path[i-1]
            move_orth_for_path(ttn,current_orth_path)
            reduced_density = compute_transfer_tensor(ttn.tensors[node_id], contracted_legs)        
            dict[node_id] = reduced_density
            next_node_id = orthogonalization_path[i][0]
            move_orth_for_path(ttn,[node_id, next_node_id])
    return dict
