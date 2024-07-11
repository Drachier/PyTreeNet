from copy import deepcopy
from typing import Dict, List
from argparse import ArgumentParser
import re

import numpy as np
from scipy.linalg import expm
import h5py

from pytreenet.time_evolution import TrotterStep, TrotterSplitting
from pytreenet.operators import TensorProduct
from pytreenet.random import random_hermitian_matrix

def first_order_splitting(op1: TensorProduct,
                          op2: TensorProduct) -> TrotterSplitting:
    """Construct a first order Trotter splitting from two matrices."""
    return TrotterSplitting([TrotterStep(op1,1),
                             TrotterStep(op2,1)])

def strang_splitting(op1: TensorProduct,
                     op2: TensorProduct) -> TrotterSplitting:
    """Construct a second order Trotter splitting from two matrices."""
    return TrotterSplitting([TrotterStep(op1,1/2),
                             TrotterStep(op2,1),
                             TrotterStep(op1,1/2)])

def create_split_dict(A: TensorProduct,
                      B: TensorProduct) -> Dict[str, TrotterSplitting]:
    """Create a dictionary Trotter splittings."""
    first_order = first_order_splitting(A,B)
    strang_split = strang_splitting(A,B)
    return {"first_order": first_order,
            "strang_split": strang_split}

def compute_comutator(A: TensorProduct,
                      B: TensorProduct) -> np.ndarray:
    """Compute the commutator of two operators."""
    first_term = np.asarray([1], dtype=complex)
    for key, op in A.items():
        factor = op @ B[key]
        first_term = np.kron(first_term,factor)
    second_term = np.asarray([1], dtype=complex)
    for key, op in A.items():
        factor = B[key] @ op
        second_term = np.kron(second_term,factor)
    return first_term - second_term

def commutator_norm(A: TensorProduct,
                    B: TensorProduct) -> float:
    """Compute the norm of the commutator."""
    commutator = compute_comutator(A,B)
    return np.linalg.norm(commutator)

def find_ref_exp(A: TensorProduct,
                 B: TensorProduct,
                 final_time: float) -> np.ndarray:
    """Compute the reference exponential."""
    Amat = A.into_operator().to_matrix().operator
    Bmat = B.into_operator().to_matrix().operator
    return expm(-1j*(Amat+Bmat)*final_time)

def final_ref_state(A: TensorProduct,
                    B: TensorProduct,
                    final_time: float,
                    state: np.ndarray) -> np.ndarray:
    """Compute the reference state at the final time."""
    ref_exp = find_ref_exp(A,B,final_time)
    return ref_exp @ state

def standard_basis(dim: int) -> Dict[str, np.ndarray]:
    """Return the standard basis."""
    full_zeros = np.zeros(dim**2)
    basis = []
    for i in range(len(full_zeros)):
        basis_vec = deepcopy(full_zeros)
        basis_vec[i] = 1
        basis.append(basis_vec)
    return {f"{i:02b}": state for i, state in enumerate(basis)}

def bell_basis() -> Dict[str, np.ndarray]:
    """Return the Bell basis."""
    factor = 1/np.sqrt(2)
    basis = [factor*np.array([1,0,0,1]),
             factor*np.array([1,0,0,-1]),
             factor*np.array([0,1,1,0]),
             factor*np.array([0,1,-1,0])]
    return {f"phi{i}": basis[i] for i in range(4)}

def create_bases_dict(dim: int) -> Dict[str, Dict[str, np.ndarray]]:
    """Create a dictionary of bases."""
    return {"standard": standard_basis(dim),
            "bell": bell_basis()}

def save_metadata(file: h5py.File,
                  final_time: float,
                  dim: int,
                  A: TensorProduct,
                  B: TensorProduct):
    """Save the metadata."""
    file.attrs["final_time"] = final_time
    file.attrs["dim"] = dim
    grpA = file.create_group("tensor_product_A")
    for key, op in A.items():
        grpA.create_dataset(key, data=op)
    grpB = file.create_group("tensor_product_B")
    for key, op in B.items():
        grpB.create_dataset(key, data=op)

def save_results(file: h5py.File,
                 time_step_sizes: List[float],
                 results: Dict[str, Dict[str, Dict[str, List[float]]]]):
    """
    Saves the time step sizes and results to the file.
    """
    _ = file.create_dataset("time_step_sizes",
                            data=time_step_sizes)
    grp_results = file.create_group("results")
    for split_name, split_results in results.items():
        grp_split = grp_results.create_group(split_name)
        for basis_name, basis_results in split_results.items():
            grp_basis = grp_split.create_group(basis_name)
            for state_id, state_results in basis_results.items():
                _ = grp_basis.create_dataset(state_id,
                                             data=state_results)
  
def input_handling() -> str:
    """
    Parse the input arguments.
    """
    parser = ArgumentParser(description="Trotterisation scaling experiment.")
    parser.add_argument("--filename", type=str, default="trotter_user_guide.hdf5",
                        help="Name of the file to save the data.")
    args = vars(parser.parse_args())
    filepath = args["filepath"][0]
    if not re.match(r".*\.hdf5", filepath):
        filepath += ".hdf5"
    return filepath

if __name__ == "__main__":
    filepath = input_handling()
    print(f"Saving data to {filepath}")
    with h5py.File(filepath, "w") as file:
        final_time = 1
        dim = 2
        A = TensorProduct({f"q{i}": random_hermitian_matrix(dim) for i in range(2)})
        B = TensorProduct({f"q{i}": random_hermitian_matrix(dim) for i in range(2)})
        save_metadata(file, final_time, dim, A, B)
        print(f"Commutator norm: {commutator_norm(A,B)}")
        split_dict = create_split_dict(A,B)
        ref_exp = find_ref_exp(A,B,final_time)
        bases = create_bases_dict(dim)
        num_step_list = np.arange(10,1000,10)
        times_steps_sizes = []
        results = {}
        for num_steps in num_step_list:
            delta_t = final_time / num_steps
            times_steps_sizes.append(delta_t)
            for split_name, split in split_dict.items():
                if split_name not in results:
                    results[split_name] = {}
                split_exp = split.exponentiate_splitting(delta_t,dim=dim)
                split_exp_mat = [operator.to_matrix() for operator in split_exp]
                for basis_name, basis in bases.items():
                    if basis_name not in results[split_name]:
                        results[split_name][basis_name] = {}
                    for state_id, state in basis.items():
                        if state_id not in results[split_name][basis_name]:
                            results[split_name][basis_name][state_id] = []
                        ref_state = final_ref_state(A,B,final_time,state)
                        for _ in range(num_steps):
                            for operator in split_exp_mat:
                                state = operator.operator @ state
                        error = np.linalg.norm(state-ref_state)
                        results[split_name][basis_name][state_id].append(error)
        save_results(file, times_steps_sizes, results)
