import h5py
import numpy as np

from numpy.random import default_rng

import pytreenet as ptn


def build_reference_tree():
    ttns = ptn.TreeTensorNetwork()

    # Physical legs come last
    node1 = ptn.TensorNode(ptn.crandn((1, 1, 2)), identifier="site1")
    node2 = ptn.TensorNode(ptn.crandn((1, 1, 1, 2)), identifier="site2")
    node3 = ptn.TensorNode(ptn.crandn((1, 2)), identifier="site3")
    node4 = ptn.TensorNode(ptn.crandn((1, 2)), identifier="site4")
    node5 = ptn.TensorNode(ptn.crandn((1, 1, 1, 2)), identifier="site5")
    node6 = ptn.TensorNode(ptn.crandn((1, 2)), identifier="site6")
    node7 = ptn.TensorNode(ptn.crandn((1, 1, 2)), identifier="site7")
    node8 = ptn.TensorNode(ptn.crandn((1, 2)), identifier="site8")

    ttns.add_root(node1)
    ttns.add_child_to_parent(node2, 0, "site1", 0)
    ttns.add_child_to_parent(node3, 0, "site2", 1)
    ttns.add_child_to_parent(node4, 0, "site2", 2)
    ttns.add_child_to_parent(node5, 0, "site1", 1)
    ttns.add_child_to_parent(node6, 0, "site5", 1)
    ttns.add_child_to_parent(node7, 0, "site5", 2)
    ttns.add_child_to_parent(node8, 0, "site7", 1)

    return ttns

def save_metadata(file, seed, max_num_terms, num_runs, conversion_dict, leg_dict):
    file.attrs["seed"] = seed
    file.attrs["max_num_terms"] = max_num_terms
    file.attrs["num_runs"] = num_runs
    meta_dicts = file.create_group("meta_dicts")
    conv_dict_group = meta_dicts.create_group("conversion_dict")
    for operator_key in conversion_dict:
        conv_dict_group.create_dataset(operator_key,
                                       data=conversion_dict[operator_key])
    leg_dict_group = meta_dicts.create_group("leg_dict")
    for leg in leg_dict:
        dset = leg_dict_group.create_dataset(leg, (1,), dtype="i")
        dset[0] = leg_dict[leg]

def main():
    X, Y, Z = ptn.pauli_matrices()
    conversion_dict = {"X": X, "Y": Y, "Z": Z, "I": np.eye(2, dtype="complex")}
    leg_dict = {"site1": 0, "site2": 1, "site3": 2, "site4": 3, "site5": 4,
                "site6": 5, "site7": 6, "site8": 7}
    num_bonds = 7

    ref_tree = build_reference_tree()

    seed = 49892897
    rng = default_rng(seed=seed)

    max_num_terms = 30
    num_runs = 1000

    possible_operators = list(conversion_dict.keys())
    site_ids = list(ref_tree.nodes.keys())

    file_path = "/work_fast/ge24fum/experiment_data/ttno_constr/"
    filename = "ttno_bond_dim.hdf5"
    with h5py.File(file_path + filename, "w") as file:

        save_metadata(file, seed, max_num_terms, num_runs, conversion_dict,
                      leg_dict)

        for num_terms in range(1, max_num_terms + 1):
            grp = file.create_group(f"run_with_{num_terms}_terms")
            grp.attrs["num_terms"] = num_terms
            dset_svd = grp.create_dataset("svd_bond_dim",
                                          shape=(num_runs, num_bonds),
                                          dtype="i")
            dset_ham = grp.create_dataset("state_diag_bond_dim",
                                         shape=(num_runs, num_bonds),
                                         dtype = "i")

            run = 0
            while run < num_runs:
                random_terms = ptn.random_symbolic_terms(num_terms,
                                                         possible_operators,
                                                         site_ids,
                                                         min_num_sites=1,
                                                         max_num_sites=8,
                                                         seed=rng)

                hamiltonian = ptn.Hamiltonian(random_terms,
                                              conversion_dictionary=conversion_dict)
                hamiltonian.pad_with_identity(ref_tree, identity="I")

                if not hamiltonian.contains_duplicates():
                    total_tensor = hamiltonian.to_tensor(ref_tree)

                    ttno_ham = ptn.TTNO.from_hamiltonian(hamiltonian, ref_tree)
                    ttno_svd = ptn.TTNO.from_tensor(ref_tree,
                                                    total_tensor,
                                                    leg_dict, mode="tSVD")

                    dimensions1 = []
                    dimensions2 = []

                    for node_id in ttno_ham.nodes:
                        node1 = ttno_ham.nodes[node_id]
                        node2 = ttno_svd.nodes[node_id]

                        if not node1.is_root():
                            dimensions1.append(node1.get_parent_leg_dim())
                            dimensions2.append(node2.get_parent_leg_dim())

                    dset_ham[run, :] = np.asarray(dimensions1)
                    dset_svd[run, :] = np.asarray(dimensions2)

                    run += 1


if __name__ == "__main__":
    main()
