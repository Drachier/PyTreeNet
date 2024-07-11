"""
The SVD seems to struggle to converge when the root is a leaf. Therefore,
 we run both cases (root at 5 and root at 6) at the same time and reuse the
 results of the root_at_5 SVD for the other case as well. This works, since
 the optimal bond dimension should always be the same for the same
 Hamiltonian.
"""

from __future__ import annotations
from typing import Dict
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
from h5py import File
from numpy.random import default_rng

from randomised_hamiltonians_to_TTNO import (save_metadata,
                                             generate_random_hamiltonian,
                                             create_bond_dim_data_sets,
                                             obtain_bond_dimensions)
import pytreenet as ptn
from pytreenet.random import random_tensor_node

def construct_tree_root_at_5():
    """
    Generates the desired tree tensor network with root at site 5 used as a
     reference to construct the Hamiltonian.
    """
    ttns = ptn.TreeTensorNetworkState()

    # Physical legs come last
    node1, tensor1 = random_tensor_node((1, 1, 2), identifier="site1")
    node2, tensor2 = random_tensor_node((1, 1, 1, 2), identifier="site2")
    node3, tensor3 = random_tensor_node((1, 2), identifier="site3")
    node4, tensor4 = random_tensor_node((1, 2), identifier="site4")
    node5, tensor5 = random_tensor_node((1, 1, 1, 2), identifier="site5")
    node6, tensor6 = random_tensor_node((1, 2), identifier="site6")
    node7, tensor7 = random_tensor_node((1, 1, 2), identifier="site7")
    node8, tensor8 = random_tensor_node((1, 2), identifier="site8")

    ttns.add_root(node5, tensor5)
    ttns.add_child_to_parent(node1, tensor1, 0, "site5", 0)
    ttns.add_child_to_parent(node2, tensor2, 0, "site1", 1)
    ttns.add_child_to_parent(node3, tensor3, 0, "site2", 1)
    ttns.add_child_to_parent(node4, tensor4, 0, "site2", 2)
    ttns.add_child_to_parent(node6, tensor6, 0, "site5", 1)
    ttns.add_child_to_parent(node7, tensor7, 0, "site5", 2)
    ttns.add_child_to_parent(node8, tensor8, 0, "site7", 1)
    return ttns

def construct_tree_root_at_6():
    """
    Generates the desired tree tensor network with root at site 6 used as a
     reference to construct the Hamiltonian.
    """
    ttns = ptn.TreeTensorNetworkState()

    # Physical legs come last
    node1, tensor1 = random_tensor_node((1, 1, 2), identifier="site1")
    node2, tensor2 = random_tensor_node((1, 1, 1, 2), identifier="site2")
    node3, tensor3 = random_tensor_node((1, 2), identifier="site3")
    node4, tensor4 = random_tensor_node((1, 2), identifier="site4")
    node5, tensor5 = random_tensor_node((1, 1, 1, 2), identifier="site5")
    node6, tensor6 = random_tensor_node((1, 2), identifier="site6")
    node7, tensor7 = random_tensor_node((1, 1, 2), identifier="site7")
    node8, tensor8 = random_tensor_node((1, 2), identifier="site8")

    ttns.add_root(node6, tensor6)
    ttns.add_child_to_parent(node5, tensor5, 0, "site6", 0)
    ttns.add_child_to_parent(node1, tensor1, 0, "site5", 1)
    ttns.add_child_to_parent(node2, tensor2, 0, "site1", 1)
    ttns.add_child_to_parent(node3, tensor3, 0, "site2", 1)
    ttns.add_child_to_parent(node4, tensor4, 0, "site2", 2)
    ttns.add_child_to_parent(node7, tensor7, 0, "site5", 2)
    ttns.add_child_to_parent(node8, tensor8, 0, "site7", 1)
    return ttns

def main(filename1: str, filename2: str,
         ref_tree1: ptn.TreeTensorNetworkState,
         ref_tree2: ptn.TreeTensorNetworkState,
         leg_dict1: Dict[str,int],
         leg_dict2: Dict[str,int],
         num_runs: int = 10000,
         min_num_terms: int=1,
         max_num_terms: int = 30):
    # Prepare variables
    X, Y, Z = ptn.pauli_matrices()
    conversion_dict = {"X": X, "Y": Y, "Z": Z, "I2": np.eye(2, dtype="complex")}
    num_bonds = 7
    seed = 49892894
    rng = default_rng(seed=seed)

    with File(filename1, "w") as file1:
        with File(filename2, "w") as file2:
            save_metadata(file1, seed, max_num_terms, num_runs,
                          conversion_dict,
                          leg_dict1)
            save_metadata(file2, seed, max_num_terms, num_runs,
                          conversion_dict,
                          leg_dict2)
            for num_terms in range(min_num_terms, max_num_terms + 1):
                dset_svd_1, dset_ham_1 = create_bond_dim_data_sets(file1,
                                                                   num_terms,
                                                                   num_bonds,
                                                                   num_runs)
                dset_svd_2, dset_ham_2 = create_bond_dim_data_sets(file2,
                                                                   num_terms,
                                                                   num_bonds,
                                                                   num_runs)
                run = 0
                while run < num_runs:
                    print(f"Run number {run}/{num_runs}", end="\r")
                    hamiltonian = generate_random_hamiltonian(conversion_dict,
                                                              ref_tree1,
                                                              rng,
                                                              num_terms)
                    if not hamiltonian.contains_duplicates():
                        ttno_ham1 = ptn.TTNO.from_hamiltonian(hamiltonian, ref_tree1)
                        ttno_ham2 = ptn.TTNO.from_hamiltonian(hamiltonian, ref_tree2)
                        total_tensor = hamiltonian.to_tensor(ref_tree1).operator
                        ttno_svd1 = ptn.TTNO.from_tensor(ref_tree1,
                                                        total_tensor,
                                                        leg_dict1,
                                                        mode=ptn.Decomposition.tSVD)
                        dset_ham_1[run, :] = obtain_bond_dimensions(ttno_ham1)
                        dset_ham_2[run, :] = obtain_bond_dimensions(ttno_ham2)
                        bond_dim_svd = obtain_bond_dimensions(ttno_svd1)
                        dset_svd_1[run, :] = deepcopy(bond_dim_svd)
                        # For root 6 the nodes are ordered differently in the TTNO
                        new_order = [4,0,1,2,3,5,6]
                        reordered_bond_dim = [bond_dim_svd[i] for i in new_order]
                        dset_svd_2[run, :] = np.asarray(reordered_bond_dim)
                        if np.any(dset_ham_1[run, :] < dset_svd_1[run, :]):
                            print("For root 5:", hamiltonian)
                        if np.any(dset_ham_2[run, :] < dset_svd_2[run, :]):
                            print("For root 6:", hamiltonian)
                        run += 1

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("filepath", type=str, nargs=1)
    filepath = vars(parser.parse_args())["filepath"][0]
    # For root at 5
    filepath5 = filepath + "_root_at_5.hdf5"
    print("Data will be saved in " + filepath5)
    leg_dict5 = {"site5": 0, "site1": 1, "site2": 2, "site3": 3, "site4": 4,
                 "site6": 5, "site7": 6, "site8": 7}
    # For root at 6
    filepath6 = filepath + "_root_at_6.hdf5"
    print("Data will be saved in " + filepath6)
    leg_dict6 = {"site6": 0, "site5": 1, "site1": 2, "site2": 3, "site3": 4,
                 "site4": 5, "site7": 6, "site8": 7}
    main(filepath5, filepath6,
         construct_tree_root_at_5(), construct_tree_root_at_6(),
         leg_dict5, leg_dict6,
         min_num_terms=30, num_runs=10000)
