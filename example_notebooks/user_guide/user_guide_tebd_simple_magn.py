"""
For the exact computation we take the convention that the root node 0 is the
first site in a tensor product, the next ones are the chains in anti-clockwise
order.
"""
from copy import copy
from argparse import ArgumentParser

import numpy as np
import h5py

from pytreenet.operators.common_operators import pauli_matrices
from pytreenet.time_evolution.exact_time_evolution import ExactTimeEvolution
from pytreenet.time_evolution.tebd import TEBD

from tebd_user_guide_util import (create_tebd,
                                  choose_state,
                                  create_file_path,
                                  input_handling_filepath)

def num_sites(length: int) -> int:
    """
    The number of sites for a given length.
    """
    return 3*length + 1

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
    # nsites = num_sites(length)
    # terms = []
    # for i in range(nsites):
    #     term = 1
    #     for j in range(nsites):
    #         if i == j:
    #             term = np.kron(term, -1*g*X)
    #         else:
    #             term = np.kron(term, np.eye(2))
    #     terms.append(term)
    # single_site = sum(terms)

    # terms_one_chain = []
    # for i in range(length):
    #     term = 1
    #     for j in range(length):
    #         if i == j:
    #             term = np.kron(term, -1*Z)
    #         elif i == j + 1:
    #             term = np.kron(term, Z)
    #         else:
    #             term = np.kron(term, np.eye(2))
    #     terms_one_chain.append(term)
    # for s, term in enumerate(terms_one_chain):
    #     terms_one_chain[s] = np.kron(np.eye(2), term)
    # terms_chain0 = []
    # ident = np.eye(2**(2*length))
    # for term in terms_one_chain:
    #     terms_chain0.append(np.kron(term, ident))
    # terms_chain1 = []
    # ident = np.eye(2**length)
    # for term in terms_one_chain:
    #     op = np.kron(ident, term)
    #     terms_chain1.append(np.kron(op, ident))
    # terms_chain2 = []
    # ident = np.eye(2**(2*length))
    # for term in terms_one_chain:
    #     terms_chain2.append(np.kron(ident, term))
    # non_root_terms = terms_chain0 + terms_chain1 + terms_chain2
    # non_root_two_site_terms = sum(non_root_terms)

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
    return ham

    # root_terms = []
    # op = np.kron(-1*Z, Z)
    # op = np.kron(op, np.eye(2**(3*length-1)))
    # root_terms.append(op)
    # op = np.kron(-1*Z, np.eye(2**length))
    # op = np.kron(op, Z)
    # op = np.kron(op, np.eye(2**(2*length-1)))
    # root_terms.append(op)
    # op = np.kron(-1*Z, np.eye(2**(2*length)))
    # op = np.kron(op, Z)
    # op = np.kron(op, np.eye(2**(length-1)))
    # root_terms.append(op)
    # root_terms = sum(root_terms)
    # return single_site + non_root_two_site_terms + root_terms

def exact_init_state(length: int = 2) -> np.ndarray:
    """
    Generate the exact initial state.
    """
    zero_state = np.array([1, 0])
    state = copy(zero_state)
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

def save_tebd_results(file: h5py.File, max_bd: int, tebd: TEBD):
    """
    Saves the results of this time-evolution.
    """
    group = file.create_group(f"max_bd_{max_bd}")
    group.attrs["max_bd"] = max_bd
    bd_group = group.create_group("bond_dim")
    for key, value in tebd.operator_result("bond_dim").items():
        bd_group.create_dataset(str(key), data=value)
    group.create_dataset("magn", data=tebd.operator_result("magn",
                                                           realise=True))

def save_metadata(file: h5py.File, g: float, length: int,
                  time_step_size: float, final_time: float,
                  rel_tol: float, total_tol: float):
    """
    Save the metadata.
    """
    file.attrs["g"] = g
    file.attrs["length"] = length
    file.attrs["time_step_size"] = time_step_size
    file.attrs["final_time"] = final_time
    file.attrs["rel_tol"] = rel_tol
    file.attrs["total_tol"] = total_tol

def handle_input() -> str:
    """
    Obtains the filepath for saving via the command line.
    """
    parser = ArgumentParser()
    parser = input_handling_filepath(parser)
    args = vars(parser.parse_args())
    filepath = create_file_path(args["filepath"][0])
    return filepath

def main():
    filepath = handle_input()
    with h5py.File(filepath, "w") as file:
        g = 0.1
        time_step_size = 0.01
        final_time = 1
        length = 2
        rel_tol = 1e-10
        total_tol = 1e-10
        save_metadata(file, g, length, time_step_size, final_time,
                      rel_tol, total_tol)
        exact_evo = create_exact_time_evolution(time_step_size, final_time, g,
                                                length=length)
        exact_evo.run()
        save_exact_results(file, exact_evo)
        save_times(file, exact_evo)
        max_bond_dims = range(1,length**2+1)
        for max_bond_dim in max_bond_dims:
            tebd = create_tebd(length, g, time_step_size, final_time,
                            max_bond_dim, rel_tol, total_tol)
            tebd.run()
            save_tebd_results(file, max_bond_dim, tebd)


if __name__ == "__main__":
    main()
