from argparse import ArgumentParser

import h5py
import numpy as np

from pytreenet.operators.common_operators import pauli_matrices
from pytreenet.time_evolution.exact_time_evolution import ExactTimeEvolution
from pytreenet.time_evolution.tdvp import TDVPAlgorithm
from pytreenet.util import SVDParameters

from user_guide_tdvp_util import (generate_fo_1tdvp,
                                  generate_so_1tdvp,
                                  generate_so_2tdvp,
                                  choose_state,
                                  create_file_path)

def num_sites(length: int) -> int:
    """
    The number of sites for a given length.
    """
    return 3*length +1

def exact_magn_operator(length: int = 2) -> np.ndarray:
    """
    Generate the exact magnetisation operator.
    """
    nsites = num_sites(length)
    magn_op = 1
    for _ in range(nsites):
        magn_op = np.kron(magn_op, pauli_matrices()[2])
    return magn_op

def exact_hamiltonian(g :float, length: int = 2) -> np.ndarray:
    """
    Generate the exact Hamiltonian.
    """
    X, _, Z = pauli_matrices()
    if length != 2:
        raise ValueError("Only length 2 is currently supported!")
    gX = -1*g*X
    ham = np.kron(gX,np.eye(2**6))
    ham += np.kron(np.eye(2), np.kron(gX, np.eye(2**5)))
    ham += np.kron(np.eye(2**2), np.kron(gX, np.eye(2**4)))
    ham += np.kron(np.eye(2**3), np.kron(gX, np.eye(2**3)))
    ham += np.kron(np.eye(2**4), np.kron(gX, np.eye(2**2)))
    ham += np.kron(np.eye(2**5), np.kron(gX, np.eye(2)))
    ham += np.kron(np.eye(2**6), gX)
    ham += np.kron(np.eye(2), np.kron(-1*Z, np.kron(Z, np.eye(2**4))))
    ham += np.kron(np.eye(2**3), np.kron(-1*Z, np.kron(Z, np.eye(2**2))))
    ham += np.kron(np.eye(2**5), np.kron(-1*Z, Z))
    ham += np.kron(-1*Z, np.kron(Z, np.eye(2**5)))
    ham += np.kron(-1*Z, np.kron(np.eye(2**2), np.kron(Z, np.eye(2**3))))
    ham += np.kron(-1*Z, np.kron(np.eye(2**4), np.kron(Z, np.eye(2))))
    ham += np.kron(-1*Z, np.kron(Z, np.kron(np.eye(2), np.kron(Z, np.kron(np.eye(2), np.kron(Z, np.eye(2)))))))
    return ham

def exact_init_state(length: int = 2) -> np.ndarray:
    """
    Generate the exact initial state.
    """
    state = np.asarray([1,0],dtype=complex)
    for _ in range(3):
        for position, _ in enumerate(range(length)):
            local_state = choose_state(position+1)
            state = np.kron(state, local_state)
    return state

def create_exact_time_evolution(time_step_size: float,
                                final_time: float,
                                g: float,
                                length: int = 2) -> ExactTimeEvolution:
    """
    Generate the exact time-evolution used as reference.
    """
    initial_state = exact_init_state(length)
    magn_op = exact_magn_operator(length)
    ops = {"magn": magn_op}
    hamiltonian = exact_hamiltonian(g, length)
    exact_evo = ExactTimeEvolution(initial_state, hamiltonian,
                                   time_step_size, final_time,
                                   ops)
    return exact_evo

def save_hyperparameters(file: h5py.File, delta_t: float, final_time: float,
                         g: float, length: int):
    """
    Save the hyperparameters.
    """
    file.attrs["delta_t"] = delta_t
    file.attrs["final_time"] = final_time
    file.attrs["g"] = g
    file.attrs["length"] = length

def save_exact_results(file: h5py.File, exact_evo: ExactTimeEvolution):
    """
    Save the exact results.
    """
    group = file.create_group("exact")
    group.create_dataset("magn", data=exact_evo.operator_result("magn", realise=True))

def save_times(file: h5py.File, exact_evo: ExactTimeEvolution):
    """
    Save the times.
    """
    file.create_dataset("times", data=exact_evo.times())

def save_so2tdvp_results(file: h5py.File, max_bd: int,
                         tdvp: TDVPAlgorithm, svd_params: SVDParameters):
    """
    Saves the results of the second order two-site TDVP.
    """
    if "so2tdvp" not in file:
        file.create_group("so2tdvp")
        group2tdvp = file["so2tdvp"]
        group2tdvp.attrs["rel_tol"] = svd_params.rel_tol
        group2tdvp.attrs["total_tol"] = svd_params.total_tol
    else:
        group2tdvp = file["so2tdvp"]
    group = group2tdvp.create_group(f"max_bd_{max_bd}")
    group.attrs["max_bd"] = max_bd
    bd_group = group.create_group("bond_dim")
    for key, value in tdvp.operator_result("bond_dim").items():
        bd_group.create_dataset(str(key), data=value)
    group.create_dataset("magn", data=tdvp.operator_result("magn",
                                                           realise=True))

def save_1tdvp_results(file: h5py.File, max_bd: int, 
                       tdvp: TDVPAlgorithm, order: int):
    """
    Saves the results of the first order TDVP.
    """
    if order == 1:
        orderstr = "fo1tdvp"
    elif order == 2:
        orderstr = "so1tdvp"
    else:
        raise ValueError("Only first and second order 1TDVP are supported!")
    if orderstr not in file:
        file.create_group(orderstr)
    group1tdvp = file[orderstr]
    group = group1tdvp.create_group(f"max_bd_{max_bd}")
    group.attrs["max_bd"] = max_bd
    bd_group = group.create_group("bond_dim")
    for key, value in tdvp.operator_result("bond_dim").items():
        bd_group.create_dataset(str(key), data=value)
    group.create_dataset("magn", data=tdvp.operator_result("magn",
                                                           realise=True))

def input_handling():
    """
    Handle command line arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("filepath", nargs=1, type=str)
    args = vars(parser.parse_args())
    filepath = args["filepath"][0]
    return create_file_path(filepath)

def main():
    filepath = input_handling()
    with h5py.File(filepath, "w") as file:
        delta_t = 0.01
        final_time = 1
        g = 0.1
        length = 2
        save_hyperparameters(file, delta_t, final_time, g, length)
        exact_evo = create_exact_time_evolution(delta_t, final_time,
                                                g, length)
        exact_evo.run()
        save_exact_results(file, exact_evo)
        save_times(file, exact_evo)
        max_bds = [1,2,3,4]
        rel_tol = 1e-10
        total_tol = 1e-10
        for max_bd in max_bds:
            fo1tdvp = generate_fo_1tdvp(length, g, max_bd, delta_t, final_time)
            fo1tdvp.run()
            save_1tdvp_results(file, max_bd, fo1tdvp, 1)
            so1tdvp = generate_so_1tdvp(length, g, max_bd, delta_t, final_time)
            so1tdvp.run()
            save_1tdvp_results(file, max_bd, so1tdvp, 2)
            so2tdvp = generate_so_2tdvp(length, g, delta_t, final_time,
                                        max_bd, rel_tol, total_tol)
            so2tdvp.run()
            save_so2tdvp_results(file, max_bd, so2tdvp, so2tdvp.svd_parameters)

if __name__ == "__main__":
    main()
        