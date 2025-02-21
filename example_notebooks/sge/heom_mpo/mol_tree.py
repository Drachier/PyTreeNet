from copy import deepcopy
import re

from pytreenet.ttns import TreeTensorNetworkState
from pytreenet.core.node import Node
from pytreenet.random import crandn

def generate_molecule_tree(mol_dim: int, cav_dim: int,
                           mol_bath_dim: int, cav_bath_dim: int,
                           num_mol: int = 2, num_mol_bath: int = 4,
                           num_cav_bath: int = 4) -> TreeTensorNetworkState:
    """
    Generate the HEOM TTNS structure for moluces and the cavity.

    The convention is that the cavity site is named "cavity" and the molecule
    sites are named "mol0", "mol1", etc. The bath sites are named "cav_bath0",
    "cav_bath1", etc. for the cavity bath and "mol_bath0_0", "mol_bath0_1",
    etc., where the first integer is the molecule index and the second integer
    is the bath index. The twin sites of the cavity and the molecules are named
    "cavity_twin" and "mol0_twin", "mol1_twin", etc., respectively. The virtual
    sites are named "virt0", "virt1", etc.
    
    Args:
        mol_dim (int): Dimension of the molecular Hilbert space.
        cav_dim (int): Dimension of the cavity Hilbert space.
        mol_bath_dim (int): Dimension of the molecular bath Hilbert space.
        cav_bath_dim (int): Dimension of the cavity bath Hilbert space.
        num_mol (int, optional): Number of molecules. Defaults to 2.
        num_mol_bath (int, optional): Number of molecular bath modes. Defaults
            to 4.
        num_cav_bath (int, optional): Number of cavity bath modes. Defaults
            to 4.

    Returns:
        TreeTensorNetworkState: The TTNS structure for the molecule cavity
            system.

    """
    ttns = TreeTensorNetworkState()
    bond_dim = 2 # Not actually relevant, but 2 is nicer than 1 -\_(ツ)_/-
    ttns = _add_cavity_sites_and_cavity_bath_sites(cav_dim, cav_bath_dim,
                                                   num_cav_bath, bond_dim,
                                                   ttns)
    ttns = _add_virtual_sites(num_mol, bond_dim, ttns)
    ttns = _add_molecule_sites_and_baths(mol_dim, num_mol, mol_bath_dim,
                                         num_mol_bath, bond_dim, ttns)
    return ttns

def _add_cavity_sites_and_cavity_bath_sites(cav_dim: int,
                                            cav_bath_dim: int,
                                            num_cav_bath: int,
                                            bond_dim: int,
                                            ttns: TreeTensorNetworkState) -> TreeTensorNetworkState:
    """
    Add the cavity sites, its site in twin space, and the cavity bath sites to the TTNS.

    Args:
        cav_bath_dim (int): Dimension of the cavity bath Hilbert space.
        num_cav_bath (int): Number of cavity bath modes.
        bond_dim (int): The bond dimensions of the TTNS.
        ttns (TreeTensorNetworkState): The TTNS to add the sites to.

    Returns:
        TreeTensorNetworkState: The TTNS with the cavity sites and cavity bath
            sites added.

    """
    cav_tensor = crandn((bond_dim,bond_dim,cav_dim))
    cavity_node = Node(identifier="cavity")
    ttns.add_root(cavity_node, deepcopy(cav_tensor))
    cavity_twin_node = Node(identifier="cavity_twin")
    ttns.add_child_to_parent(cavity_twin_node, deepcopy(cav_tensor), 0,
                             "cavity", 0)
    # Add cavity bath sites
    cav_bath_tensor = crandn((bond_dim,bond_dim,cav_bath_dim))
    for i in range(num_cav_bath):
        cav_bath_node = Node(identifier=f"cav_bath{i}")
        if i == 0:
            ttns.add_child_to_parent(cav_bath_node, deepcopy(cav_bath_tensor), 0,
                                     "cavity_twin", 1)
        elif i == num_cav_bath - 1:
            ttns.add_child_to_parent(cav_bath_node, deepcopy(cav_bath_tensor[:,0,:]), 0,
                                     f"cav_bath{i-1}", 1)
        else:
            ttns.add_child_to_parent(cav_bath_node, deepcopy(cav_bath_tensor), 0,
                                     f"cav_bath{i-1}", 1)
    return ttns

def _add_virtual_sites(num_mol: int,
                       bond_dim: int,
                       ttns: TreeTensorNetworkState) -> TreeTensorNetworkState:
    """
    Add the virtual sites to the TTNS.

    Args:
        num_mol (int): Number of molecules.
        bond_dim (int): The bond dimensions of the TTNS.
        ttns (TreeTensorNetworkState): The TTNS to add the sites to.

    Returns:
        TreeTensorNetworkState: The TTNS with the virtual sites added.

    """
    if num_mol == 1:
        raise NotImplementedError("Virtual sites are not implemented for one molecule. Use an MPS instead.")

    virt_tensor = crandn((bond_dim,bond_dim,bond_dim,1))
    start_node = Node(identifier="virt0")
    ttns.add_child_to_parent(start_node, deepcopy(virt_tensor), 0,
                             "cavity", 1)
    num_open_legs = 2
    virt_nodes_current_level = [start_node]
    virt_nodes_next_level = []
    while num_open_legs < num_mol:
        for node in virt_nodes_current_level:
            if node.nvirt_legs() < 3:
                # Add new virtual node
                virt_node = Node(identifier=f"virt{num_open_legs-1}")
                ttns.add_child_to_parent(virt_node, deepcopy(virt_tensor), 0,
                                         node.identifier, node.nvirt_legs())
                virt_nodes_next_level.append(virt_node)
                num_open_legs += 1
                break
        else:
            # We need a new layer of virtual nodes
            virt_nodes_current_level = virt_nodes_next_level
            virt_nodes_next_level = []
    return ttns

def _add_molecule_sites_and_baths(mol_dim: int,
                                  num_mol: int,
                                  mol_bath_dim: int,
                                  num_mol_bath: int,
                                  bond_dim: int,
                                  ttns: TreeTensorNetworkState) -> TreeTensorNetworkState:
    """
    Attaches the molecule sites to the virtual sites and attaches the molecule
    bath sites as well.

    Args:
        mol_dim (int): Dimension of the molecular Hilbert space.
        num_mol (int): Number of molecules.
        mol_bath_dim (int): Dimension of the molecular bath Hilbert space.
        num_mol_bath (int): Number of molecular bath modes.
        bond_dim (int): The bond dimensions of the TTNS.
        ttns (TreeTensorNetworkState): The TTNS to add the sites to.
    
    Returns:
        TreeTensorNetworkState: The TTNS with the molecule sites and molecule
            bath sites added.
    """
    virtual_sites = [node_id for node_id in ttns.nodes.keys()
                     if re.match(r"virt\d+", node_id)]
    open_dimension_sites = [ttns.nodes[node_id] for node_id in virtual_sites
                            if ttns.nodes[node_id].nvirt_legs() < 3]
    for i in range(num_mol):
        virtual_site = open_dimension_sites[0]
        mol_tensor = crandn((bond_dim,bond_dim,mol_dim))
        mol_node = Node(identifier=f"mol{i}")
        ttns.add_child_to_parent(mol_node, deepcopy(mol_tensor), 0,
                                 virtual_site.identifier, virtual_site.nvirt_legs())
        # Add twin site
        mol_twin_node = Node(identifier=f"mol{i}_twin")
        ttns.add_child_to_parent(mol_twin_node, deepcopy(mol_tensor), 0,
                                 mol_node.identifier, 1)
        # Add molecule bath sites
        mol_bath_tensor = crandn((bond_dim,bond_dim,mol_bath_dim))
        for j in range(num_mol_bath):
            mol_bath_node = Node(identifier=f"mol_bath{i}_{j}")
            if j == 0:
                ttns.add_child_to_parent(mol_bath_node, deepcopy(mol_bath_tensor), 0,
                                         mol_twin_node.identifier, 1)
            elif j == num_mol_bath - 1:
                ttns.add_child_to_parent(mol_bath_node, deepcopy(mol_bath_tensor[:,0,:]), 0,
                                         f"mol_bath{i}_{j-1}", 1)
            else:
                ttns.add_child_to_parent(mol_bath_node, deepcopy(mol_bath_tensor), 0,
                                         f"mol_bath{i}_{j-1}", 1)
        # Discard the virtual site, if it has no open legs
        if virtual_site.nvirt_legs() == 3:
            del open_dimension_sites[0]
    return ttns

if __name__ == "__main__":
    ttns = generate_molecule_tree(2, 2, 4, 4)
    for node in ttns.nodes.values():
        print(node.identifier, node.children, node.parent)
    print(50*"-")
    ttns = generate_molecule_tree(2, 2, 4, 4,
                                  num_mol=4)
    for node in ttns.nodes.values():
        print(node.identifier, node.children, node.parent)
    print(50*"-")
    ttns = generate_molecule_tree(2, 2, 4, 4,
                                  num_mol=3)
    for node in ttns.nodes.values():
        print(node.identifier, node.children, node.parent)
