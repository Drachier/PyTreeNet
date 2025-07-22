"""
A module to provide various commonly used models for simulations.
"""
from typing import List, Tuple, Union, Dict
from warnings import warn

from numpy import ndarray, zeros, mean

from ..hamiltonian import Hamiltonian
from ..tensorproduct import TensorProduct
from ..sim_operators import (create_single_site_observables)
from ..common_operators import pauli_matrices
from ...core.ttn import TreeTensorNetwork

from .topology import Topology
from .two_site_model import (HeisenbergModel,
                                IsingModel,
                                FlippedIsingModel,
                                BoseHubbardModel,
                                HeisenbergParameters,
                                IsingParameters,
                                BoseHubbardParameters)

DEPRECIATION_WARNING = "This function to generate models is deprecated. " \
                          "Use the model classes instead."

def heisenberg_model(structure: TreeTensorNetwork | list[tuple[str,str]],
                     ext_magn: float = 0.0,
                     x_factor: float = 1.0,
                     y_factor: None | float = None,
                     z_factor: None | float = None
                     ) -> Hamiltonian:
    """
    Generates the Hamiltonian of the Heisenberg model

    .. math::
        - \sum_{<i,j>} J_x X_iX_j + J_y Y_iY_j + J_z Z_iZ_j - g \sum_i Z_i
    
    Args:
        structure (TreeTensorNetwork | list[tuple[str,str]]): The nearest
            neighbours. They can either be given directly or be inferred from
            from a given TTN.
        ext_magn (float): The external magnetic field to be applied.
            Defaults to 0.0.
        x_factor (float): The factor of the XX term.
        y_factor (float): The factor of the YY term. If None, it will be the
            same as the `x_factor`.
        z_factor (float): The factor of the ZZ term. If None, it will be the
            same as the `x_factor`.

    Returns:
        Hamiltonian: The Hamiltonian of the Heisenberg model.
    """
    warn(DEPRECIATION_WARNING, DeprecationWarning)
    params = HeisenbergParameters(x_factor=x_factor,
                                  y_factor=y_factor,
                                  z_factor=z_factor,
                                  ext_z=ext_magn)
    heisenberg_model = HeisenbergModel.from_dataclass(params)
    return heisenberg_model.generate_hamiltonian(structure)

def ising_model(ref_tree: Union[TreeTensorNetwork, List[Tuple[str, str]]],
                ext_magn: float,
                factor: float = 1.0
                ) -> Hamiltonian:
    """
    Generates the Ising model with an external magnetic field for a full
    qubit tree, i.e. every node has a physical dimension of 2.

    Args:
        ref_tree (Union[TreeTensorNetwork, List[Tuple[str, str]]]): The nearest
            neighbour identifiers. They can either be given as a TreeTensorNetwork
            object or as a list of tuples of nearest neighbours.
        ext_magn (float): The strength of the external magnetic field (Z-term).
        factor (float): The coupling factor between the nearest neighbours.
            (XX-term) Defaults to 1.0.
    
    Returns:
        Hamiltonian: The Hamiltonian of the Ising model.

    """
    warn(DEPRECIATION_WARNING, DeprecationWarning)
    params = IsingParameters(factor=factor,
                             ext_magn=ext_magn)
    model = IsingModel.from_dataclass(params)
    return model.generate_hamiltonian(ref_tree)

def ising_model_2D(grid: Union[tuple[str,int,int],list[list[str]]],
                   ext_magn: float,
                   coupling: float = 1.0
                   ) -> Hamiltonian:
    """
    Generates the 2D Ising model for a given grid structure.

    Args:
        grid (Union[tuple[str,int,int],ArrayLike]): The grid structure of the
            Ising model. Can either be a tuple with the structure of the grid
            or a 2D array with the node identifiers. If it is a tuple, the
            first element is the prefix of the node identifiers, the second
            element is the number of rows and the third element is the number
            of columns.
        ext_magn (float): The strength of the external magnetic field (Z-term).
        coupling (float): The coupling strength between the nearest neighbours.
            (XX-term) Defaults to 1.0.
    
    Returns:
        Hamiltonian: The 2D-Ising Hamiltonian

    """
    warn(DEPRECIATION_WARNING, DeprecationWarning)
    params = IsingParameters(factor=coupling,
                             ext_magn=ext_magn)
    model = IsingModel.from_dataclass(params)
    if isinstance(grid, tuple):
        num_rows, num_cols = grid[1], grid[2]
        return model.generate_2d_model(num_rows, num_cols,
                                        site_ids=grid[0])
    return model.generate_2d_model(len(grid),
                                   num_cols=len(grid[0]),
                                   site_ids=grid)

def flipped_ising_model(ref_tree: Union[TreeTensorNetwork, List[Tuple[str, str]]],
                        ext_magn: float,
                        factor: float = 1.0
                        ) -> Hamiltonian:
    """
    Generates the Ising model with an external magnetic field for a full
    qubit tree, i.e. every node has a physical dimension of 2. The Ising model
    is flipped, i.e. X and Z operators are interchanged.

    Args:
        ref_tree (Union[TreeTensorNetwork, List[Tuple[str, str]]]): The nearest
            neighbour identifiers. They can either be given as a TreeTensorNetwork
            object or as a list of tuples of nearest neighbours.
        ext_magn (float): The strength of the external magnetic field (X-term).
        factor (float): The coupling factor between the nearest neighbours.
            (ZZ-term) Defaults to 1.0.

    Returns:
        Hamiltonian: The Hamiltonian of the flipped Ising model.
    
    """
    warn(DEPRECIATION_WARNING, DeprecationWarning)
    params = IsingParameters(factor=factor,
                             ext_magn=ext_magn)
    model = FlippedIsingModel.from_dataclass(params)
    return model.generate_hamiltonian(ref_tree)

def flipped_ising_model_2D(grid: Union[tuple[str,int,int],list[list[str]]],
                            ext_magn: float,
                            coupling: float = 1.0
                            ) -> Hamiltonian:
    """
    Generates the 2D Ising model for a given grid structure. The Ising model
    is flipped, i.e. X and Z operators are interchanged.

    Args:
        grid (Union[tuple[str,int,int],list[list[str]]]): The grid structure of
            the Ising model. Can either be a tuple with the structure of the grid
            or a 2D array with the node identifiers. If it is a tuple, the first
            element is the prefix of the node identifiers, the second element is
            the number of rows and the third element is the number of columns.
        ext_magn (float): The strength of the external magnetic field (X-term).
        factor (float): The coupling factor between the nearest neighbours.
            (ZZ-term) Defaults to 1.0.

    Returns:
        Hamiltonian: The Hamiltonian of the 2D flipped Ising model.

    """
    warn(DEPRECIATION_WARNING, DeprecationWarning)
    params = IsingParameters(factor=coupling,
                             ext_magn=ext_magn)
    model = FlippedIsingModel.from_dataclass(params)
    if isinstance(grid, tuple):
        num_rows, num_cols = grid[1], grid[2]
        return model.generate_2d_model(num_rows, num_cols,
                                        site_ids=grid[0])
    return model.generate_2d_model(len(grid),
                                   num_cols=len(grid[0]),
                                   site_ids=grid)

def local_magnetisation_from_topology(topology: Topology,
                                      system_size: int,
                                      site_prefix: str = "site"
                                        ) -> Dict[str, TensorProduct]:
    """
    Generates the local magnetisation operator for a given topology.

    Args:
        topology (Topology): The topology of the system.
        system_size (int): The characteristic size of the system.
        site_prefix (str): The prefix for the site identifiers.
        Defaults to "site".
    """
    num_sites = topology.num_sites(system_size)
    structure = [f"{site_prefix}_{i}"
                 for i in range(num_sites)]
    return local_magnetisation(structure)

def local_magnetisation(structure: Union[TreeTensorNetwork,List[str]]
                        ) -> Dict[str,TensorProduct]:
    """
    Generates the local magnetisation operator for a given tree structure.

    Args:
        structure (Union[TreeTensorNetwork,List[str]]): The tree structure for
            which the local magnetisation operator should be generated. Can
            also be a list of node identifiers.
    
    Returns:
        Dict[str,TensorProduct]: The local magnetisation operators.

    """
    sigma_z = pauli_matrices()[2]
    return create_single_site_observables(sigma_z, structure)

def total_magnetisation(local_magnetisations: List[ndarray]
                        ) -> ndarray:
    """
    Computes the total magnetisation from the local magnetisations.
    
    Args:
        local_magnetisations (List[ndarray]): The local magnetisations as a
            list of arrays, where each array contains the local magnetisations
            for one site for different times.

    Returns:
        ndarray: The total magnetisation

        .. math::
            M = 1/L \sum_i^L m_i

    """
    if len(local_magnetisations) == 0:
        raise ValueError("No local magnetisations given!")
    num_sites = len(local_magnetisations)
    magn = zeros((num_sites, local_magnetisations[0].shape[0]),
                 dtype=local_magnetisations[0].dtype)
    for i in range(num_sites):
        magn[i] = local_magnetisations[i]
    return mean(magn, axis=0)

def bose_hubbard_model(
        structure: Union[TreeTensorNetwork, List[Tuple[str, str]]],
        local_dim: int = 2,
        hopping: float = 1.0,
        on_site_int: float = 1.0,
        chem_pot: float = 0.0
                    ) -> Hamiltonian:
    """
    Generates the Bose-Hubbard model Hamiltonian for a given structure.

    ..math::
        H = -t \sum_{i,j} (a_i^\dagger a_j + a_j^\dagger a_i) + U \sum_i n_i(n_i-1) - \mu \sum_i n_i

    where :math:`a_i^\dagger` and :math:`a_i` are the creation and annihilation
    operators, :math:`n_i` is the number operator, :math:`t` is the hopping
    strength, :math:`U` is the on-site interaction strength and :math:`\mu`
    is the chemical potential.

    Args:
        structure (Union[TreeTensorNetwork, List[Tuple[str, str]]]): The nearest
            neighbour identifiers. They can either be given as a TreeTensorNetwork
            obsject, and be inferred, or as a list of tuples of nearest
            neighbours.
        local_dim (int): The local to be truncated to, i.e. the maximum number
            of particles per site. Defaults to 2.
        hopping (float): The hopping strength between the nearest neighbours.
            Defaults to 1.0.
        on_site_int (float): The on-site interaction strength. Defaults to 1.0.
        chem_pot (float): The chemical potential. Defaults to 0.0.
    """
    warn(DEPRECIATION_WARNING, DeprecationWarning)
    params = BoseHubbardParameters(local_dim=local_dim,
                                   hopping=hopping,
                                   on_site_int=on_site_int,
                                   chem_pot=chem_pot)
    model = BoseHubbardModel.from_dataclass(params)
    return model.generate_hamiltonian(structure)
