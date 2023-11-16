from __future__ import annotations
from argparse import ArgumentParser

from randomised_hamiltonians_to_TTNO import main
import pytreenet as ptn

def construct_tree_root_at_5():
    """
    Generates the desired tree tensor network with root at site 5 used as a
     reference to construct the Hamiltonian.
    """
    ttns = ptn.TreeTensorNetworkState()

    # Physical legs come last
    node1, tensor1 = ptn.random_tensor_node((1, 1, 2), identifier="site1")
    node2, tensor2 = ptn.random_tensor_node((1, 1, 1, 2), identifier="site2")
    node3, tensor3 = ptn.random_tensor_node((1, 2), identifier="site3")
    node4, tensor4 = ptn.random_tensor_node((1, 2), identifier="site4")
    node5, tensor5 = ptn.random_tensor_node((1, 1, 1, 2), identifier="site5")
    node6, tensor6 = ptn.random_tensor_node((1, 2), identifier="site6")
    node7, tensor7 = ptn.random_tensor_node((1, 1, 2), identifier="site7")
    node8, tensor8 = ptn.random_tensor_node((1, 2), identifier="site8")

    ttns.add_root(node5, tensor5)
    ttns.add_child_to_parent(node1, tensor1, 0, "site5", 0)
    ttns.add_child_to_parent(node2, tensor2, 0, "site1", 1)
    ttns.add_child_to_parent(node3, tensor3, 0, "site2", 1)
    ttns.add_child_to_parent(node4, tensor4, 0, "site2", 2)
    ttns.add_child_to_parent(node6, tensor6, 0, "site5", 1)
    ttns.add_child_to_parent(node7, tensor7, 0, "site5", 2)
    ttns.add_child_to_parent(node8, tensor8, 0, "site7", 1)
    return ttns

def tree_root_at_6():
    """
    Generates the desired tree tensor network with root at site 6 used as a
     reference to construct the Hamiltonian.
    """
    ttns = ptn.TreeTensorNetworkState()

    # Physical legs come last
    node1, tensor1 = ptn.random_tensor_node((1, 1, 2), identifier="site1")
    node2, tensor2 = ptn.random_tensor_node((1, 1, 1, 2), identifier="site2")
    node3, tensor3 = ptn.random_tensor_node((1, 2), identifier="site3")
    node4, tensor4 = ptn.random_tensor_node((1, 2), identifier="site4")
    node5, tensor5 = ptn.random_tensor_node((1, 1, 1, 2), identifier="site5")
    node6, tensor6 = ptn.random_tensor_node((1, 2), identifier="site6")
    node7, tensor7 = ptn.random_tensor_node((1, 1, 2), identifier="site7")
    node8, tensor8 = ptn.random_tensor_node((1, 2), identifier="site8")

    ttns.add_root(node6, tensor6)
    ttns.add_child_to_parent(node5, tensor5, 0, "site6", 0)
    ttns.add_child_to_parent(node1, tensor1, 0, "site5", 1)
    ttns.add_child_to_parent(node2, tensor2, 0, "site1", 1)
    ttns.add_child_to_parent(node3, tensor3, 0, "site2", 1)
    ttns.add_child_to_parent(node4, tensor4, 0, "site2", 2)
    ttns.add_child_to_parent(node7, tensor7, 0, "site5", 2)
    ttns.add_child_to_parent(node8, tensor8, 0, "site7", 1)
    return ttns

# def main(filename1: str, filename2: str,
#          ref_tree1: ptn.TreeTensorNetworkState,
#          ref_tree2: ptn.TreeTensorNetworkState,
#          num_runs: int = 10000,
#          min_num_terms: int=1,
#          max_num_terms: int = 30):
#     # Prepare variables
#     X, Y, Z = ptn.pauli_matrices()
#     conversion_dict = {"X": X, "Y": Y, "Z": Z, "I2": np.eye(2, dtype="complex")}
#     leg_dict1 = {"site5": 0, "site1": 1, "site2": 2, "site3": 3, "site4": 4,
#                  "site6": 5, "site7": 6, "site8": 7}
#     leg_dict2 = {"site6": 0, "site5": 1, "site1": 2, "site2": 3, "site3": 4,
#                  "site4": 5, "site7": 6, "site8": 7}
#     num_bonds = 7
#     seed = 49892894
#     rng = default_rng(seed=seed)

#     with h5py.File(filename1, "w") as file1:
#         with h5py.File(filename2, "w") as file2:
#             save_metadata(file1, seed, max_num_terms, num_runs,
#                           conversion_dict,
#                           leg_dict1)
#             save_metadata(file2, seed, max_num_terms, num_runs,
#                           conversion_dict,
#                           leg_dict2)
#             for num_terms in tqdm(range(min_num_terms, max_num_terms + 1)):
#                 dset_svd_1, dset_ham_1 = create_bond_dim_data_sets(file1,
#                                                                    num_terms,
#                                                                    num_bonds,
#                                                                    num_runs)
#                 dset_svd_2, dset_ham_2 = create_bond_dim_data_sets(file2,
#                                                                    num_terms,
#                                                                    num_bonds,
#                                                                    num_runs)
#                 run = 0
#                 while run < num_runs:
#                     hamiltonian = generate_random_hamiltonian(conversion_dict,
#                                                             ref_tree,
#                                                             rng,
#                                                             num_terms)
#                     if not hamiltonian.contains_duplicates():
#                         ttno_ham = ptn.TTNO.from_hamiltonian(hamiltonian, ref_tree)
#                         total_tensor = hamiltonian.to_tensor(ref_tree).operator
#                         ttno_svd = ptn.TTNO.from_tensor(ref_tree,
#                                                         total_tensor,
#                                                         leg_dict,
#                                                         mode=ptn.Decomposition.tSVD)
#                         dset_ham[run, :] = obtain_bond_dimensions(ttno_ham)
#                         dset_svd[run, :] = obtain_bond_dimensions(ttno_svd)
#                         if np.all(dset_ham[run, :] > dset_svd[run, :]):
#                             print(hamiltonian)
#                         run += 1

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("filepath", type=str, nargs=1)
    filepath = vars(parser.parse_args())["filepath"][0]
    # For root at 5
    filepath5 = filepath + "_root_at_5.hdf5"
    print("Data will be saved in " + filepath5)
    leg_dict5 = {"site5": 0, "site1": 1, "site2": 2, "site3": 3, "site4": 4,
                 "site6": 5, "site7": 6, "site8": 7}
    main(filepath5, construct_tree_root_at_5(),
         leg_dict5, min_num_terms=30, num_runs=20)
    # For root at 6
    filepath6 = filepath + "_root_at_6.hdf5"
    print("Data will be saved in " + filepath6)
    leg_dict6 = {"site6": 0, "site5": 1, "site1": 2, "site2": 3, "site3": 4,
                 "site4": 5, "site7": 6, "site8": 7}
    main(filepath6, tree_root_at_6(),
         leg_dict6, min_num_terms=30, num_runs=20)
