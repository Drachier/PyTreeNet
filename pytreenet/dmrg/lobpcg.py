import numpy as np
import scipy
from copy import deepcopy
from typing import Callable, List, Tuple
from .als import AlternatingLeastSquares
from .variational_fitting import VariationalFitting

from ..util.misc_functions import linear_combination, add, orthogonalise_to, scale
from ..util.tensor_splitting import SVDParameters

from ..ttns.ttns_ttno.zipup import zipup 
from ..contractions.state_operator_contraction import get_matrix_element
from ..ttno.ttno_class import TTNO
from ..ttns import TreeTensorNetworkState

def precond_lobpcg(shifted_ttno: TTNO, state: TreeTensorNetworkState, svd_params: SVDParameters, num_sweeps: int=2, max_iter: int=100) -> TreeTensorNetworkState:
    als = AlternatingLeastSquares(shifted_ttno, state_x = deepcopy(state), state_b = deepcopy(state), num_sweeps=num_sweeps, max_iter=max_iter, svd_params=svd_params, site='one-site', residual_rank=0)
    als.run()    
    return als.state_x

def lobpcg_single(ttno:TTNO, state_x: TreeTensorNetworkState,precond_func: Callable, svd_params: SVDParameters, max_iter: int, file_path: List[str]) -> Tuple[TreeTensorNetworkState, list[float]]:
    """
    Perform LOBPCG on a single TTNS.
    """
    num_sweeps = 2
    rayleigh = state_x.operator_expectation_value(ttno).real
    energy = [rayleigh]
    state_p = None 
    
    for i in range(max_iter):
        state_new = zipup(ttno, state_x, svd_params)
        varfit = VariationalFitting([ttno], [deepcopy(state_x)], state_new.conjugate(), num_sweeps, 100, svd_params, "one-site", [1.])
        varfit.run()
        state_r = varfit.y.conjugate()
        state_r = add(state_x, state_r, -rayleigh, 1.)
        state_r.canonical_form(state_r.root_id)
        state_r = precond_func(state_r, svd_params)
        if len(file_path) > 0:
            state_r = orthogonalise_to(state_r, file_path, svd_params.max_bond_dim, num_sweeps)
            state_x = orthogonalise_to(state_x, file_path, svd_params.max_bond_dim, num_sweeps)
        xrp_list = [state_x, state_r]
        if state_p is not None:
            xrp_list.append(state_p)

        H = np.zeros((len(xrp_list),len(xrp_list)), dtype=np.complex128)
        M = np.zeros((len(xrp_list),len(xrp_list)), dtype=np.complex128)
        
        for j in range(len(xrp_list)):
            H[j,j] = xrp_list[j].operator_expectation_value(ttno).real
            M[j,j] = xrp_list[j].scalar_product(xrp_list[j]).real
            for k in np.arange(j+1,len(xrp_list)):
                H[j,k] = get_matrix_element(xrp_list[j].conjugate(), ttno, xrp_list[k])
                H[k,j] = H[j,k].conj()
                M[j,k] = xrp_list[k].scalar_product(xrp_list[j])
                M[k,j] = M[j,k].conj()
        ew, ev = scipy.linalg.eigh(H, M)
        ev = ev.real
        state_x = linear_combination(xrp_list, ev[:,0], int(svd_params.max_bond_dim), num_sweeps = num_sweeps)

        if state_p is None:
            state_p = scale(state_r, ev[1,0])
        else:
            state_p = linear_combination([state_r, state_p], [ev[1,0], ev[2,0]], svd_params.max_bond_dim, num_sweeps = num_sweeps)
        norm = state_x.normalise()
        rayleigh = ew[0].real 
        energy.append(rayleigh)
   
    return state_x, energy

def lobpcg_block(ttno:TTNO, state_x_list: List[TreeTensorNetworkState],precond_func: Callable, svd_params: SVDParameters, max_iter: int, file_path: List[str], varfit_num_sweeps: int=1) -> Tuple[TreeTensorNetworkState, list[float]]:
    n_states = len(state_x_list)
    energies = []
    rayleigh = [state.operator_expectation_value(ttno).real for state in state_x_list]
    state_p_list = None 
    
    for i in range(max_iter):
        state_r_list = []
        for ix, state_x in enumerate(state_x_list):
            state_new = zipup(ttno, state_x, svd_params)
            varfit = VariationalFitting([ttno], [deepcopy(state_x)], state_new.conjugate(), varfit_num_sweeps, 500, svd_params, "one-site", [1.])
            varfit.run()
            state_r = varfit.y.conjugate()
            state_r = add(state_x, state_r, -rayleigh[ix], 1.)
            state_r.canonical_form(state_r.root_id)
            state_r = precond_func(state_r, svd_params)
            if len(file_path) > 0:
                state_r = orthogonalise_to(state_r, file_path, svd_params.max_bond_dim, varfit_num_sweeps)
                state_x = orthogonalise_to(state_x, file_path, svd_params.max_bond_dim, varfit_num_sweeps)
            state_x_list[ix] = state_x
            state_r_list.append(state_r)
            
        xrp_list = state_x_list + state_r_list
        if state_p_list is not None:
            xrp_list += state_p_list
        H = np.zeros((len(xrp_list),len(xrp_list)), dtype=np.float64)
        M = np.zeros((len(xrp_list),len(xrp_list)), dtype=np.float64)

        for j in range(len(xrp_list)):
            H[j,j] = xrp_list[j].operator_expectation_value(ttno).real
            M[j,j] = xrp_list[j].scalar_product(xrp_list[j]).real
            for k in np.arange(j+1,len(xrp_list)):
                H[j,k] = get_matrix_element(xrp_list[j].conjugate(), ttno, xrp_list[k]).real
                H[k,j] = H[j,k]
                M[j,k] = xrp_list[k].scalar_product(xrp_list[j]).real
                M[k,j] = M[j,k]
        ew, ev = scipy.linalg.eigh(H, M)
        ev = ev.real
        state_x_list_new = []
        state_p_list_new = []
        for n in range(n_states):
            state_x = linear_combination(xrp_list, ev[:3*n_states,n], int(svd_params.max_bond_dim), num_sweeps = varfit_num_sweeps)
            state_x_list_new.append(state_x)

            if state_p_list is None:
                state_p = linear_combination(state_r_list, ev[n_states:,n], int(svd_params.max_bond_dim), num_sweeps = varfit_num_sweeps)
            else:
                state_p = linear_combination(xrp_list[n_states:], ev[n_states:3*n_states,n], svd_params.max_bond_dim, num_sweeps = varfit_num_sweeps)
            if np.linalg.norm(ev[n_states:,n]) < 1e-4:
                print("Warning: the residual is too small, numerical instability is likely to occur.")
        state_x_list = state_x_list_new
        state_p_list = state_p_list_new
        rayleigh = ew[:n_states].real 
        energies.append(rayleigh)
    return state_x_list, np.array(energies).T.tolist()
   