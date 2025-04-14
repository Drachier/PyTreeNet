import os

import h5py
from tqdm import tqdm

from pytreenet.ttno.state_diagram import StateDiagram, TTNOFinder

from mol_ham import generate_hamiltonian
from mol_tree import generate_molecule_tree

def run_creation(min_mol, max_mol,
                 num_baths_min, num_baths_max,
                 mode, homogenous_mm, homogenous_mc):
    mol_nums = range(min_mol, max_mol+1)
    bath_nums = range(num_baths_min, num_baths_max+1)
    bond_dim_data = {"avg_bond_dims": {}, "max_bond_dims": {}}

    for num_bath in tqdm(bath_nums):
        avg_bond_dims = []
        max_bond_dims = []
        for num_mol in mol_nums:
            ttns = generate_molecule_tree(20, 2, 20, 10,
                                        num_mol=num_mol,
                                        num_mol_bath=num_bath,
                                        num_cav_bath=num_bath)
            ham = generate_hamiltonian(num_mol,
                                       num_mol_bath=num_bath,
                                       num_cav_bath=num_bath,
                                       homogenous_mm=homogenous_mm,
                                       homogenous_mc=homogenous_mc)
            for term in ham.terms:
                for node_id in term[2].keys():
                    if node_id not in ttns.nodes:
                        print(f"Node {node_id} not in MPS")
            ham_pad = ham.pad_with_identities(ttns)
            sd = StateDiagram.from_hamiltonian(ham_pad, ttns,
                                            method=mode)
            bond_dims = []
            for _, vertex_coll in sd.vertex_colls.items():
                bond_dim = len(vertex_coll.contained_vertices)
                bond_dims.append(bond_dim)
            avg_bond_dim = sum(bond_dims)/len(bond_dims)
            max_bond_dim = max(bond_dims)
            avg_bond_dims.append(avg_bond_dim)
            max_bond_dims.append(max_bond_dim)
        bond_dim_data["avg_bond_dims"][num_bath] = avg_bond_dims
        bond_dim_data["max_bond_dims"][num_bath] = max_bond_dims
    return bond_dim_data

def mode_to_save_str(mode):
    if mode == TTNOFinder.BIPARTITE:
        return "bipartite"
    elif mode == TTNOFinder.SGE:
        return "sge"
    else:
        raise ValueError("Invalid mode")

def homogenous_to_save_str(homogenous):
    if homogenous:
        return "homogenous"
    else:
        return "heterogenous"

def save_data(bond_dim_data, metadata):
    save_path = os.getcwd() + "/data/data_tree_" + mode_to_save_str(metadata["mode"]) + "_" + homogenous_to_save_str(metadata["homogenous_mm"]) + ".h5"
    with h5py.File(save_path, "w") as file:
        for key, value in metadata.items():
            if isinstance(value, TTNOFinder):
                value = mode_to_save_str(value)
            file.attrs[key] = value
        for key, data in bond_dim_data.items():
            file.create_group(key)
            for num_bath, bond_dims in data.items():
                file[key].create_dataset(str(num_bath), data=bond_dims)

def main(metadata):
    modes = [TTNOFinder.BIPARTITE, TTNOFinder.SGE]
    homogenousities = [True, False]
    for mod in modes:
        for hom in homogenousities:
            metadata["mode"] = mod
            metadata["homogenous_mm"] = hom
            metadata["homogenous_mc"] = hom
            bond_dim_datas = run_creation(**metadata)
            save_data(bond_dim_datas, metadata)

if __name__ == "__main__":
    main({
        "min_mol": 2,
        "max_mol": 15,
        "num_baths_min": 1,
        "num_baths_max": 15}
    )