from copy import deepcopy

from pytreenet.core.node import Node
from pytreenet.special_ttn.mps import MatrixProductState
from pytreenet.random import crandn

def generate_mps_structure(mol_dim: int, cav_dim: int,
                           mol_bath_dim: int, cav_bath_dim: int,
                           num_mol: int = 2, num_mol_bath: int = 4,
                           num_cav_bath: int = 4) -> MatrixProductState:
    """
    Generate the HEOM MPS structure for molecules and the cavity.

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
        MatrixProductState: The MPS structure for the molecule cavity system.

    """
    mps = MatrixProductState()
    bond_dim = 2 # Not actually relevant, but 2 is nicer than 1 -\_(ツ)_/-
    cav_tensor = crandn((bond_dim,bond_dim,cav_dim))
    cavity_node = Node(identifier="cavity")
    mps.add_root(cavity_node, deepcopy(cav_tensor))
    cavity_twin_node = Node(identifier="cavity_twin")
    mps.attach_node_right_end(cavity_twin_node, deepcopy(cav_tensor))
    cav_bath_tensor = crandn((bond_dim,bond_dim,cav_bath_dim))
    for i in range(num_cav_bath):
        cav_bath_node = Node(identifier=f"cav_bath{i}")
        mps.attach_node_right_end(cav_bath_node, deepcopy(cav_bath_tensor))
    mol_id = 0
    half_num_mol = num_mol // 2
    for i in range(half_num_mol):
        mps = _attach_molecule_plus_bath_left(mps, mol_dim, mol_bath_dim,
                                            num_mol_bath, bond_dim, mol_id)
        mol_id += 1
        mps = _attach_molecule_plus_bath_right(mps, mol_dim, mol_bath_dim,
                                            num_mol_bath, bond_dim, mol_id)
        mol_id += 1
    if num_mol % 2 == 1:
        mps = _attach_molecule_plus_bath_left(mps, mol_dim, mol_bath_dim,
                                            num_mol_bath, bond_dim, mol_id)
    return mps

def _attach_molecule_plus_bath(direction: int,
                               mps: MatrixProductState,
                               mol_dim:int,
                               mol_bath_dim: int,
                               num_mol_bath: int,
                               bond_dim: int,
                               mol_ind: int) -> MatrixProductState:
    """
    Attach the sites of a molecule and its bath to an end of the MPS.

    Args:
        direction (int): Defines, if the molecule and bath are attached to the
            left (direction = -1) or right (direction = 1) end of the MPS.
        mps (MatrixProductState): The MPS to attach the molecule and bath to.
        mol_dim (int): Dimension of the molecular Hilbert space.
        mol_bath_dim (int): Dimension of the molecular bath Hilbert space.
        bond_dim (int): Bond dimension of the MPS.
        mol_ind (int): Index of the molecule to uniquely identify it.
    
    Returns:
        MatrixProductState: The modified MPS with the molecule and bath
            attached to the left end.
    """
    if direction == -1:
        add_fct = mps.attach_node_left_end
    elif direction == 1:
        add_fct = mps.attach_node_right_end
    else:
        raise ValueError("Direction must be either -1 or 1.")
    mol_tensor = crandn((bond_dim,bond_dim,mol_dim))
    mol_twin_node = Node(identifier=f"mol{mol_ind}_twin")
    add_fct(mol_twin_node, deepcopy(mol_tensor))
    mol_node = Node(identifier=f"mol{mol_ind}")
    add_fct(mol_node, deepcopy(mol_tensor))
    mol_bath_tensor = crandn((bond_dim,bond_dim,mol_bath_dim))
    for i in range(num_mol_bath):
        mol_bath_node = Node(identifier=f"mol_bath{mol_ind}_{i}")
        add_fct(mol_bath_node, deepcopy(mol_bath_tensor))
    return mps

def _attach_molecule_plus_bath_left(mps: MatrixProductState,
                                    mol_dim:int,
                                    mol_bath_dim: int,
                                    num_mol_bath: int,
                                    bond_dim: int,
                                    mol_ind: int) -> MatrixProductState:
    """
    Attach the sites of a molecule and its bath to the left end of the MPS.

    Args:
        mps (MatrixProductState): The MPS to attach the molecule and bath to.
        mol_dim (int): Dimension of the molecular Hilbert space.
        mol_bath_dim (int): Dimension of the molecular bath Hilbert space.
        bond_dim (int): Bond dimension of the MPS.
        mol_ind (int): Index of the molecule to uniquely identify it.
    
    Returns:
        MatrixProductState: The modified MPS with the molecule and bath
            attached to the left end.
    """
    return _attach_molecule_plus_bath(-1, mps, mol_dim, mol_bath_dim,
                                      num_mol_bath, bond_dim, mol_ind)

def _attach_molecule_plus_bath_right(mps: MatrixProductState,
                                     mol_dim:int,
                                     mol_bath_dim: int,
                                     num_mol_bath: int,
                                     bond_dim: int,
                                     mol_ind: int) -> MatrixProductState:
    """
    Attach the sites of a molecule and its bath to the right end of the MPS.

    Args:
        mps (MatrixProductState): The MPS to attach the molecule and bath to.
        mol_dim (int): Dimension of the molecular Hilbert space.
        mol_bath_dim (int): Dimension of the molecular bath Hilbert space.
        bond_dim (int): Bond dimension of the MPS.
        mol_ind (int): Index of the molecule to uniquely identify it.
    
    Returns:
        MatrixProductState: The modified MPS with the molecule and bath
            attached to the right end.
    """
    return _attach_molecule_plus_bath(1, mps, mol_dim, mol_bath_dim,
                                    num_mol_bath, bond_dim, mol_ind)

if __name__ == "__main__":
    mps2 = generate_mps_structure(20, 2, 20, 10)
    mps1 = generate_mps_structure(20, 2, 20, 10, num_mol=1)

    print(mps2)
    print(mps1)
