from __future__ import annotations
from typing import Any, Dict
from argparse import ArgumentParser

import h5py
import numpy as np
from tqdm import tqdm

from numpy.random import default_rng

import pytreenet as ptn

def build_reference_tree() -> ptn.TreeTensorNetworkState:
    """
    Generates the desired tree tensor network used as a reference to construct
     the Hamiltonian.
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

    ttns.add_root(node1, tensor1)
    ttns.add_child_to_parent(node2, tensor2, 0, "site1", 0)
    ttns.add_child_to_parent(node3, tensor3, 0, "site2", 1)
    ttns.add_child_to_parent(node4, tensor4, 0, "site2", 2)
    ttns.add_child_to_parent(node5, tensor5, 0, "site1", 1)
    ttns.add_child_to_parent(node6, tensor6, 0, "site5", 1)
    ttns.add_child_to_parent(node7, tensor7, 0, "site5", 2)
    ttns.add_child_to_parent(node8, tensor8, 0, "site7", 1)

    return ttns

def save_metadata(file: Any, seed: int, max_num_terms: int, num_runs: int,
                  conversion_dict: Dict[str, np.ndarray],
                  leg_dict: Dict[str, int]):
    """
    Saves all the metadata of a run in an h5py-file

    Args:
        file (Any): The file object to be saved in
        seed (int): The seed used for random number generation
        max_num_terms (int): The maximum number of terms that as used in this
         experiment.
        num_runs (int): The total number of runs
        conversion_dict (Dict[str, np.ndarray]): The dictionary used to convert
         between symbolic and numeric operators.
        leg_dict (Dict[str, int]): Here for every site identifier the
         corresponding leg in the total tensor is stored.
    """
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

def generate_random_hamiltonian(conversion_dict: Dict[str, np.ndarray],
                                ref_tree: ptn.TreeTensorNetworkState,
                                rng : np.random.Generator,
                                num_terms: int) -> ptn.Hamiltonian:
    """
    Generates the random unitary to be used in a run.

    Args:
        conversion_dict (Dict[str, np.ndarray]): _description_
        ref_tree (ptn.TreeTensorNetworkState): The tree used as a reference.
        rng (np.random.Generator): The random number generator used.
        num_terms (int): The number of terms to be contained in the
         Hamiltonian.

    Returns:
        ptn.Hamiltonian: A random Hamiltonian.
    """
    site_ids = list(ref_tree.nodes.keys())
    possible_operators = list(conversion_dict.keys())
    random_terms = ptn.random_symbolic_terms(num_terms,
                                             possible_operators,
                                             site_ids,
                                             min_num_sites=1,
                                             max_num_sites=8,
                                             seed=rng)
    hamiltonian = ptn.Hamiltonian(random_terms,
                                  conversion_dictionary=conversion_dict)
    return hamiltonian.pad_with_identities(ref_tree)

def obtain_bond_dimensions(ttno: ptn.TTNO) -> np.ndarray:
    """
    Obtains the bond dimensions of a TTN.

    Args:
        ttno (ptn.TTNO): The TTN for which to determine the bond dimensions.

    Returns:
        np.ndarray: A 1D-array containing all bond-dimensions
    """
    dimensions = []
    for node_id in ttno.nodes:
        node = ttno.nodes[node_id]
        if not node.is_root():
            dimensions.append(node.parent_leg_dim())
    return np.asarray(dimensions)

def main(filepath: str, ref_tree: ptn.TreeTensorNetworkState):
    # Prepare variables
    X, Y, Z = ptn.pauli_matrices()
    conversion_dict = {"X": X, "Y": Y, "Z": Z, "I2": np.eye(2, dtype="complex")}
    leg_dict = {"site1": 0, "site2": 1, "site3": 2, "site4": 3, "site5": 4,
                "site6": 5, "site7": 6, "site8": 7}
    num_bonds = 7
    seed = 49892894
    rng = default_rng(seed=seed)
    max_num_terms = 30
    num_runs = 10000

    with h5py.File(filepath, "w") as file:
        save_metadata(file, seed, max_num_terms, num_runs, conversion_dict,
                      leg_dict)
        for num_terms in tqdm(range(1, max_num_terms + 1)):
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
                hamiltonian = generate_random_hamiltonian(conversion_dict,
                                                          ref_tree,
                                                          rng,
                                                          num_terms)
                if not hamiltonian.contains_duplicates():
                    ttno_ham = ptn.TTNO.from_hamiltonian(hamiltonian, ref_tree)
                    total_tensor = hamiltonian.to_tensor(ref_tree).operator
                    ttno_svd = ptn.TTNO.from_tensor(ref_tree,
                                                    total_tensor,
                                                    leg_dict,
                                                    mode=ptn.Decomposition.tSVD)
                    dset_ham[run, :] = obtain_bond_dimensions(ttno_ham)
                    dset_svd[run, :] = obtain_bond_dimensions(ttno_svd)
                    run += 1

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("filepath", type=str, nargs=1)
    filepath = vars(parser.parse_args())["filepath"][0]
    print("Data will be saved in " + filepath)
    ref_tree = build_reference_tree()
    main(filepath, ref_tree)
