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

class KrylovBasisMode(Enum):
    apply_ham = "apply_ham"
    apply_1st_order_expansion = "apply_1st_order_expansion"

def Krylov_basis(ttn: TreeTensorNetworkState, 
                 ttno: TTNO, 
                 num_vecs: int, 
                 tau: float, 
                 SVDParameters : SVDParameters,
                 mode: KrylovBasisMode):
   ttn_copy = deepcopy(ttn)
   ttno_copy = deepcopy(ttno)

   if mode == KrylovBasisMode.apply_1st_order_expansion:
        I = TTNO.Identity(ttno_copy)
        ttno_copy.multiply_const(-1j*tau)
        ttno_copy = sum_two_ttns(I, ttno_copy)
   if ttn_copy.orthogonality_center_id is not TDVPUpdatePathFinder(ttn_copy).find_path()[0]:
        ttn_copy.canonical_form(TDVPUpdatePathFinder(ttn_copy).find_path()[0] , SplitMode.REDUCED) 
   basis_list = [ttn_copy]
   for _ in range(num_vecs):
      ttn_copy = contract_ttno_with_ttn(ttno_copy,ttn_copy)
      ttn_copy.canonical_form(TDVPUpdatePathFinder(ttn_copy).find_path()[0] , SVDParameters)  
      # ttn_copy.normalize_ttn()
      basis_list.append(ttn_copy)
   return basis_list

def expand_subspace(ttn: TreeTensorNetworkState, 
                    ttno: TTNO,
                    expansion_params: Dict[str, Any]):
    basis = Krylov_basis(ttn,
                         ttno,
                         expansion_params["num_vecs"],
                         expansion_params["tau"],
                         expansion_params["SVDParameters"],
                         expansion_params["KrylovBasisMode"] )
    ttn_copy = deepcopy(basis[0])
    for i in range(len(basis)-1):
        ttn_copy = enlarge_ttn1_bond_with_ttn2(ttn_copy, basis[i+1], expansion_params["tol"])
        #ttn_copy.normalize_ttn()
    return ttn_copy

#from ..time_evolution.time_evolution impor
from dataclasses import replace

def perform_expansion(state, hamiltonian, tol, config):
    config.Expansion_params["tol"] = tol
    state_ex = expand_subspace(state, 
                               hamiltonian,
                               config.Expansion_params)
    after_ex_total_bond = state_ex.total_bond_dim()
    expanded_dim_tot = after_ex_total_bond - state.total_bond_dim()
    return state_ex, after_ex_total_bond, expanded_dim_tot

def phase1_increase_tol(state, hamiltonian, tol, expanded_dim_tot , config):
    max_trials = config.Expansion_params["num_second_trial"]
    num_trials = 0
    min_rel_tot_bond, max_rel_tot_bond = config.Expansion_params["rel_tot_bond"]
    while num_trials < max_trials:
        #print(f"Phase 1 - Trial {num_trials+1}:")
        if expanded_dim_tot > max_rel_tot_bond:
            #print(f"Expanded dim ({expanded_dim_tot}) > rel_tot_bond ({config.Expansion_params['rel_tot_bond']})")
            # Increase tol to reduce expanded_dim_tot
            tol *= config.Expansion_params["tol_step_increase"]
            #print("Increasing tol:", tol)
            state_ex, _, expanded_dim_tot = perform_expansion(state, hamiltonian, tol, config)
            num_trials += 1
            if min_rel_tot_bond < expanded_dim_tot <= max_rel_tot_bond:
                # Acceptable expansion found
                #print("Acceptable expansion found in Phase 1:", expanded_dim_tot)
                return state_ex, tol, expanded_dim_tot, False
            elif expanded_dim_tot <= min_rel_tot_bond:
                # Need to switch to Phase 2
                #print("Expanded dim became negative:", expanded_dim_tot)
                state_ex = state
                #print("Switching to Phase 2")
                return state_ex, tol, expanded_dim_tot, True  # Proceed to Phase 2                
    # Exceeded max trials
    #print("Exceeded maximum trials in Phase 1 without acceptable expansion")
    state_ex = state
    return state_ex, tol, expanded_dim_tot, False  # Proceed to Phase 2

def phase2_decrease_tol(state, hamiltonian, tol, expanded_dim_tot, config):
    max_trials = config.Expansion_params["num_second_trial"]
    num_trials = 0
    min_rel_tot_bond, max_rel_tot_bond = config.Expansion_params["rel_tot_bond"]
    while num_trials < max_trials:
        num_trials += 1
        #print(f"Phase 2 - Trial {num_trials}:")
        # Decrease tol to increase expanded_dim_tot
        tol *= config.Expansion_params["tol_step_decrease"]
        #print("Decreasing tol:", tol)
        state_ex, _, expanded_dim_tot = perform_expansion(state, hamiltonian, tol, config)
        #print("Expanded_dim_tot:", expanded_dim_tot)
        if min_rel_tot_bond < expanded_dim_tot <= max_rel_tot_bond:
            # Acceptable expansion found
            #print("Acceptable expansion found in Phase 2:", expanded_dim_tot)
            return state_ex, tol, expanded_dim_tot
        elif expanded_dim_tot > max_rel_tot_bond:
            # Expanded dimension exceeded rel_tot_bond again
            #print("Expanded dim exceeded rel_tot_bond again:", expanded_dim_tot)
            # Reset state_ex to initial state
            state_ex = state
            return state_ex, tol, expanded_dim_tot  # Reset and exit
    # Exceeded max trials
    #print("Exceeded maximum trials in Phase 2 without acceptable expansion")
    # Reset state_ex to initial state
    state_ex = state
    return state_ex, tol, expanded_dim_tot  # Reset and exit

def adjust_tol_and_expand(state, hamiltonian, tol , config):

    before_ex_total_bond = state.total_bond_dim()

    config.Expansion_params["SVDParameters"] = replace(config.Expansion_params["SVDParameters"],max_bond_dim=state.max_bond_dim())
    #print("SVD MAX:", state.max_bond_dim())
    #print("Initial tol:", tol)

    # Initial Expansion Attempt
    state_ex, after_ex_total_bond, expanded_dim_tot = perform_expansion(state, hamiltonian, tol, config)

    # Unpack the acceptable range
    min_rel_tot_bond, max_rel_tot_bond = config.Expansion_params["rel_tot_bond"]

    # Check initial expansion
    if min_rel_tot_bond < expanded_dim_tot <= max_rel_tot_bond:
        # Acceptable expansion found in initial attempt
        print("Acceptable expansion found in initial attempt:", expanded_dim_tot)
    elif expanded_dim_tot > max_rel_tot_bond:
        # Need to adjust tol in Phase 1
        state_ex, tol, expanded_dim_tot, switch_to_phase_2 = phase1_increase_tol(state, hamiltonian, tol, expanded_dim_tot, config)
        if switch_to_phase_2:
            # Switch to Phase 2
            state_ex, tol, expanded_dim_tot = phase2_decrease_tol(state, hamiltonian, tol, expanded_dim_tot, config)
    elif expanded_dim_tot <= min_rel_tot_bond:
        # Need to adjust tol in Phase 2
        state_ex, tol, expanded_dim_tot = phase2_decrease_tol(state, hamiltonian, tol, expanded_dim_tot, config)

    after_ex_total_bond = state_ex.total_bond_dim()
    expanded_dim_total_bond = after_ex_total_bond - before_ex_total_bond

    print("Final expanded_dim:", expanded_dim_total_bond, ":", before_ex_total_bond, "--->", after_ex_total_bond)

    return state_ex, tol




from pytreenet.contractions.contraction_util import get_equivalent_legs
import random

def random_leaf(state) :
    leaf_nodes = []
    for node in state.nodes.values():
        if node.is_leaf():
            leaf_nodes.append(node.identifier)
    return random.choice(leaf_nodes)

def random_update_path(state):
    TDVPUpdatePath = TDVPUpdatePathFinder(state)
    TDVPUpdatePath.start = random_leaf(state)
    TDVPUpdatePath.main_path = TDVPUpdatePath.state.find_path_to_root(TDVPUpdatePath.start)
    return TDVPUpdatePath.find_path()


def enlarge_ttn1_bond_with_ttn222(ttn1, ttn2, tol):
   ttn1_copy = deepcopy(ttn1)
   ttn3 = deepcopy(ttn1)
   update_path = TDVPUpdatePathFinder(ttn1_copy).find_path()
   path_next = find_tdvp_orthogonalization_path(ttn1_copy,update_path) 


def enlarge_ttn1_bond_with_ttn2(ttn1, ttn2, tol):
   ttn1_copy = deepcopy(ttn1)
   ttn3 = deepcopy(ttn1)
   update_path = random_update_path(ttn1_copy)
   path_next = find_tdvp_orthogonalization_path(ttn1_copy,update_path) 
   ttn1_copy.canonical_form(update_path[0])
   ttn2.canonical_form(update_path[0])   
   
   for i,node_id in enumerate(update_path[:-1]): 
        next_node_id = path_next[i][0]
        index = ttn1_copy.nodes[node_id].neighbour_index(next_node_id)
        index_prime = ttn1_copy.nodes[next_node_id].neighbour_index(node_id) 
        pho_A = compute_transfer_tensor(ttn1_copy.tensors[node_id],(index,))
        pho_B = compute_transfer_tensor(ttn2.tensors[node_id],(index,))
        pho = pho_A + pho_B
        v = compute_v(pho, index, tol)

        ttn3.tensors[node_id] = v
        ttn3.nodes[node_id].link_tensor(v)
        
        ttn1_copy.move_orthogonalization_center(next_node_id)
        ttn2.move_orthogonalization_center(next_node_id)

        legs = get_equivalent_legs(ttn1_copy.nodes[node_id], ttn2.nodes[node_id])
        if legs[0] != legs[1]:
            print(node_id , legs)
        
        v_legs = list(range(0,v.ndim))
        v_legs.remove(index)
        #print(node_id , next_node_id)
        #print( ttn1_copy.tensors[node_id].shape , np.conjugate(v).shape , v_legs)
        CVd = np.tensordot(ttn1_copy.tensors[node_id] , np.conjugate(v) , (v_legs,v_legs))
        ttn1_copy.tensors[next_node_id] = absorb_matrix_into_tensor(CVd, ttn1_copy.tensors[next_node_id], (0,index_prime))
        CVd = np.tensordot(ttn2.tensors[node_id] , np.conjugate(v) , (v_legs,v_legs))
        ttn2.tensors[next_node_id] = absorb_matrix_into_tensor(CVd, ttn2.tensors[next_node_id], (0,index_prime))
        
        if ttn1_copy.orthogonality_center_id != None:
            if len(path_next[i]) > 1:
                ttn1_copy.orthogonality_center_id = path_next[i][0]
                ttn1_copy.move_orthogonalization_center(path_next[i][-1])
        if ttn2.orthogonality_center_id != None:
            if len(path_next[i]) > 1:
                ttn2.orthogonality_center_id = path_next[i][0]
                ttn2.move_orthogonalization_center(path_next[i][-1])

        legs = get_equivalent_legs(ttn1_copy.nodes[node_id], ttn2.nodes[node_id])
        if legs[0] != legs[1]:
            print(node_id , legs)                

   last_node_id = update_path[-1]
   ttn3.tensors[last_node_id] = ttn1_copy.tensors[last_node_id]
   ttn3.nodes[last_node_id].link_tensor(ttn1_copy.tensors[last_node_id]) 
   

   #ttn3.canonical_form(update_path[0], SplitMode.REDUCED) 

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
    perm = perm[1:index+1] + [perm[0]] + perm[index+1:]
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

import scipy
def eig_absolute_tol(pho, tol):
    # Check if pho is an empty matrix
    #num_nevs = pho.shape[0] - 1 
    #w, v = scipy.sparse.linalg.eigsh(pho , k = num_nevs, which='LM')
    w , v = np.linalg.eigh(pho)

    magnitudes = np.abs(w)
    sorted_indices = np.argsort(magnitudes)[::-1]
        
    w = w[sorted_indices]
    v = v[:, sorted_indices]
    
    k = np.sum(magnitudes > tol)
    #print(f"Significant eigenvalues (magnitude > tol): {k}")
    
    v = v[:, :k]
    #print(f"Eigenvectors : {v.shape}")
    return v


def eig(pho, tol):
    """
    Truncates the eigenvectors based on relative tolerance.

    Args:
        pho (np.ndarray): Input matrix.
        tol (float): Relative tolerance as a fraction of the largest eigenvalue.

    Returns:
        np.ndarray: Truncated eigenvectors.
    """
    w, v = np.linalg.eigh(pho)  # Compute eigenvalues and eigenvectors

    magnitudes = np.abs(w)  # Get the magnitudes of the eigenvalues
    sorted_indices = np.argsort(magnitudes)[::-1]  # Sort indices in descending magnitude order

    w = w[sorted_indices]  # Sort eigenvalues in descending order
    v = v[:, sorted_indices]  # Sort eigenvectors correspondingly

    max_eigenvalue = magnitudes[0]  # Largest eigenvalue
    rel_cutoff = tol * max_eigenvalue  # Relative tolerance cutoff

    # Keep eigenvalues above the relative cutoff
    k = np.sum(magnitudes > rel_cutoff)
    v = v[:, :k]  # Retain eigenvectors corresponding to significant eigenvalues

    return v



import numpy as np
import warnings
import logging
import tenpy
from tenpy.linalg.np_conserved import Array, LegCharge, ChargeInfo
from tenpy.linalg.sparse import NpcLinearOperator
from tenpy.tools.params import asConfig
from tenpy.linalg.krylov_based import Arnoldi

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def generate_trivial_legcharges(tensor_shape):
    """
    Generates a list of trivial LegCharge objects for each leg in the tensor.
    
    Parameters
    ----------
    tensor_shape : tuple
        Shape of the tensor (e.g., eff_H or psi).
        
    Returns
    -------
    list of LegCharge
        A list of LegCharge objects with trivial charges for each leg of the tensor.
    """
    charge_info = ChargeInfo(mod=[])  # Empty mod implies no charge conservation
    return [LegCharge.from_trivial(ind_len=dim, chargeinfo=charge_info) for dim in tensor_shape]

def eig_tenpy(pho, tol):
    """
    Computes significant eigenvectors of a Hermitian matrix using the Arnoldi algorithm via TenPy.
    
    Parameters
    ----------
    pho : np.ndarray
        A square Hermitian matrix.
    tol : float
        Tolerance for determining significant eigenvalues. The algorithm stops if the energy difference per step
        is below this tolerance.
    num_eigvals : int, optional
        Number of eigenvalues and corresponding eigenvectors to compute. Defaults to 1 (ground state).
    max_iterations : int, optional
        Maximum number of Arnoldi iterations. Defaults to 100.
    reortho : bool, optional
        Whether to re-orthogonalize the Krylov basis to maintain numerical stability. Defaults to False.
    
    Returns
    -------
    eigenvalues : np.ndarray
        Array of computed eigenvalues.
    eigenvectors : np.ndarray
        Matrix whose columns are the corresponding eigenvectors.
    """
    
    num_eigvals = pho.shape[0] - 1  
    
    # Define the linear operator H using TenPy's NpcLinearOperator
    class DenseLinearOperator(NpcLinearOperator):
        def __init__( matrix, legcharges):
            super().__init__()
            matrix = matrix
            legcharges = legcharges

        def matvec( x):
            # x is an Array object; extract the ndarray data
            x_ndarray = x.to_ndarray()
            result = np.dot(matrix, x_ndarray)
            # Convert result back to TenPy's Array with the same leg charges
            return Array.from_ndarray(result, legcharges, dtype=matrix.dtype)
    
    # Initialize the starting vector psi0 as a normalized random vector
    np.random.seed(42)  # For reproducibility
    random_vector = np.random.randn(pho.shape[0]) + 1j * np.random.randn(pho.shape[0])
    psi0_numpy = random_vector.astype(pho.dtype)
    psi0_norm = np.linalg.norm(psi0_numpy)
    if psi0_norm == 0:
        raise ValueError("Initial vector 'psi0' has zero norm.")
    psi0_numpy /= psi0_norm
    
    # Generate trivial legcharges for psi0
    legcharges_psi0 = generate_trivial_legcharges(psi0_numpy.shape)
    
    # Convert NumPy array to TenPy's Array with proper legs using Array.from_ndarray
    psi0 = Array.from_ndarray(psi0_numpy, legcharges_psi0, dtype=pho.dtype)
    
    # Initialize the linear operator with the initial matrix and legcharges
    H_operator = DenseLinearOperator(pho, legcharges_psi0)
    
    # Define options for the Arnoldi method
    options = {
        'E_tol': np.inf,
        'which': 'LM',             # 'LM' for Largest Magnitude
        'num_ev': num_eigvals,     # Number of desired eigenvalues
        'N_min': 2,                # Minimum number of iterations before checking convergence
        'N_max': 100,   # Maximum number of iterations
        'N_max': num_eigvals +1 ,   # Maximum number of iterations
        'P_tol': 1e-14,            # Projection tolerance
        'reortho': True            # Reorthogonalize the Krylov basis
    }
    
    
    # Convert options to TenPy's Config
    options = asConfig(options, 'Arnoldi')
    
    # Initialize Arnoldi
    arnoldi = Arnoldi(H_operator, psi0, options)
    
    # Run the Arnoldi algorithm
    w, v, N = arnoldi.run()

    magnitudes = np.abs(w)
    sorted_indices = np.argsort(magnitudes)[::-1]
    w = w[sorted_indices]
    v = [v.to_ndarray() for v in v] 
    v = np.zeros((pho.shape[0], len(sorted_indices)+ 1), dtype=complex)
    for i, idx in enumerate(sorted_indices):
        v[:, i] = v[idx]
    k = np.sum(magnitudes > tol)
    v = v[:, :k]  
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
        (TreeTensorNetworkState): The first tree tensor network.
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
        elif isinstance(ttn1, TTNO) and isinstance(ttn2, TTNO):
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
    #if not isinstance(ttno1, TTNO) or not isinstance(ttno2, TTNO):
    #    raise TypeError("The arguments must be TTNOs.")
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
