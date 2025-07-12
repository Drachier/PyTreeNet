"""
A module to provide various commonly used models for simulations.
"""
from typing import List, Tuple, Union, Dict
from fractions import Fraction

from numpy import ndarray, zeros, mean, eye
from numpy.typing import ArrayLike

from .hamiltonian import Hamiltonian
from .tensorproduct import TensorProduct
from .sim_operators import (create_single_site_hamiltonian,
                            create_nearest_neighbour_hamiltonian,
                            single_site_operators)
from .common_operators import pauli_matrices, bosonic_operators
from ..core.tree_structure import TreeStructure
from ..util.ttn_exceptions import positivity_check

def ising_model(ref_tree: Union[TreeStructure, List[Tuple[str, str]]],
                ext_magn: float,
                factor: float = 1.0
                ) -> Hamiltonian:
    """
    Generates the Ising model with an external magnetic field for a full
    qubit tree, i.e. every node has a physical dimension of 2.

    Args:
        ref_tree (Union[TreeStructure, List[Tuple[str, str]]]): The nearest
            neighbour identifiers. They can either be given as a TreeStructure
            object or as a list of tuples of nearest neighbours.
        ext_magn (float): The strength of the external magnetic field (Z-term).
        factor (float): The coupling factor between the nearest neighbours.
            (XX-term) Defaults to 1.0.
    
    Returns:
        Hamiltonian: The Hamiltonian of the Ising model.

    """
    paulis = pauli_matrices()
    return _abstract_ising_model(ref_tree, ext_magn, factor,
                                 ("Z", paulis[2]), ("X", paulis[0]))

def ising_model_2D(grid: Union[tuple[str,int,int],ArrayLike],
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
    return _abstract_2D_ising(grid,
                              ext_magn,
                              coupling,
                              ("Z", pauli_matrices()[2]),
                              ("X", pauli_matrices()[0]))

def flipped_ising_model(ref_tree: Union[TreeStructure, List[Tuple[str, str]]],
                        ext_magn: float,
                        factor: float = 1.0
                        ) -> Hamiltonian:
    """
    Generates the Ising model with an external magnetic field for a full
    qubit tree, i.e. every node has a physical dimension of 2. The Ising model
    is flipped, i.e. X and Z operators are interchanged.

    Args:
        ref_tree (Union[TreeStructure, List[Tuple[str, str]]]): The nearest
            neighbour identifiers. They can either be given as a TreeStructure
            object or as a list of tuples of nearest neighbours.
        ext_magn (float): The strength of the external magnetic field (X-term).
        factor (float): The coupling factor between the nearest neighbours.
            (ZZ-term) Defaults to 1.0.

    Returns:
        Hamiltonian: The Hamiltonian of the flipped Ising model.
    
    """
    paulis = pauli_matrices()
    return _abstract_ising_model(ref_tree, ext_magn, factor,
                                 ("X", paulis[0]), ("Z", paulis[2]))

def flipped_ising_model_2D(grid: Union[tuple[str,int,int],ArrayLike],
                            ext_magn: float,
                            coupling: float = 1.0
                            ) -> Hamiltonian:
    """
    Generates the 2D Ising model for a given grid structure. The Ising model
    is flipped, i.e. X and Z operators are interchanged.

    Args:
        ref_tree (Union[TreeStructure, List[Tuple[str, str]]]): The nearest
            neighbour identifiers. They can either be given as a TreeStructure
            object or as a list of tuples of nearest neighbours.
        ext_magn (float): The strength of the external magnetic field (X-term).
        factor (float): The coupling factor between the nearest neighbours.
            (ZZ-term) Defaults to 1.0.

    Returns:
        Hamiltonian: The Hamiltonian of the 2D flipped Ising model.

    """
    return _abstract_2D_ising(grid,
                              ext_magn,
                              coupling,
                              ("X", pauli_matrices()[0]),
                              ("Z", pauli_matrices()[2]))

def _abstract_ising_model(ref_tree: Union[TreeStructure, List[Tuple[str, str]]],
                          ext_magn: float,
                          coupling: float,
                          ext_magn_op: Tuple[str,ndarray],
                          nn_op: Tuple[str,ndarray]
                          ) -> Hamiltonian:
    """
    Generates the Ising model with an external magnetic field for a full
    qubit tree, i.e. every node has a physical dimension of 2.

    Args:
        ref_tree (Union[TreeStructure, List[Tuple[str, str]]]): The nearest
            neighbour identifiers. They can either be given as a TreeStructure
            object or as a list of tuples of nearest neighbours.
        ext_magn (float): The strength of the external magnetic field.
        factor (float): The coupling factor between the nearest neighbours.
        ext_magn_op (Tuple[str,ndarray]): The operator for the external
            magnetic field.
        nn_op (Tuple[str,ndarray]): The operator for the nearest neighbour
            coupling.

    Returns:
        Hamiltonian: The Hamiltonian of the Ising model.

    """
    objs_single = _get_ham_objects(ext_magn, "ext_magn", ext_magn_op)
    ext_magn_op_id = objs_single[0]
    factor = objs_single[1]
    single_dict = objs_single[2]
    single_mapping = objs_single[3]
    # Produce the single site Hamiltonian
    # We need to prepare the identifiers for the single site Hamiltonian
    single_site_structure = _adapt_structure_for_single_site(ref_tree)
    single_site_ham = create_single_site_hamiltonian(single_site_structure,
                                                     ext_magn_op_id,
                                                     factor=factor,
                                                     conversion_dict=single_dict,
                                                     coeffs_mapping=single_mapping)
    # Produce the nearest neighbour Hamiltonian
    objs_nn = _get_ham_objects(coupling,"coupling",nn_op)
    nn_op_id = objs_nn[0]
    factor_nn = objs_nn[1]
    nn_dict = objs_nn[2]
    nn_mapping = objs_nn[3]
    nearest_neighbour_ham = create_nearest_neighbour_hamiltonian(ref_tree,
                                                                 nn_op_id,
                                                                 factor=factor_nn,
                                                                 conversion_dict=nn_dict,
                                                                 coeffs_mapping=nn_mapping)
    # Now we add all together into one Hamiltonian
    ham = Hamiltonian()
    ham.add_hamiltonian(single_site_ham)
    ham.add_hamiltonian(nearest_neighbour_ham)
    local_dim = 2
    ham.include_identities([1,local_dim])
    return ham

def _get_ham_objects(factor_value: float,
                     factor_id: str,
                     operator: Union[Tuple[str,ndarray],dict[str,ndarray]]
                    ) -> Tuple[str,tuple[Fraction,str],Dict[str,ndarray],Dict[str,float]]:
    """
    Creates the objects used as arguments for the creation of concrete
    Hamiltonians
    """
    factor = (Fraction(-1), factor_id)
    if isinstance(operator, tuple):
        operator_id = operator[0]
        conv_dict = {operator_id: operator[1]}
    elif isinstance(operator, dict):
        conv_dict = operator
        operator_id = list(conv_dict.keys())[0]
    mapping = {factor_id: factor_value}
    return operator_id, factor, conv_dict, mapping

def local_magnetisation(structure: Union[TreeStructure,List[str]],
                        with_factor: bool = True
                        ) -> Dict[str,TensorProduct]:
    """
    Generates the local magnetisation operator for a given tree structure.

    Args:
        structure (Union[TreeStructure,List[str]]): The tree structure for
            which the local magnetisation operator should be generated. Can
            also be a list of node identifiers.
        with_factor (bool): If True, the local magnetisation operator is
            provided with a factor of 1.0.
    
    Returns:
        Dict[str,TensorProduct]: The local magnetisation operators.

    """
    sigma_z = pauli_matrices()[2]
    return single_site_operators(sigma_z, structure, with_factor=with_factor)

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
        raise ValueError("No local magnetisations given.")
    num_sites = len(local_magnetisations)
    magn = zeros((num_sites, local_magnetisations[0].shape[0]),
                 dtype=local_magnetisations[0].dtype)
    for i in range(num_sites):
        magn[i] = local_magnetisations[i]
    return mean(magn, axis=0)

def _abstract_2D_ising(grid: Union[tuple[str,int,int],ArrayLike],
                        ext_magn: float,
                        coupling: float,
                        ext_magn_op: Tuple[str,ndarray],
                        nn_op: Tuple[str,ndarray]
                        ) -> Hamiltonian:
    """
    Creates a 2D Ising model for any given operators.

    Args:
        grid (Union[tuple[str,int,int],ArrayLike]): The grid structure of the
            Ising model. Can either be a tuple with the structure of the grid
            or a 2D array with the node identifiers. If it is a tuple, the
            first element is the prefix of the node identifiers, the second
            element is the number of rows and the third element is the number
            of columns.
        ext_magn (float): The strength of the external magnetic field.
        coupling (float): The coupling strength between the nearest neighbours.
        ext_magn_op (Tuple[str,ndarray]): The operator for the external magnetic
            field.
        nn_op (Tuple[str,ndarray]): The operator for the nearest neighbour
            coupling.
        
    Returns:
        Hamiltonian: The Hamiltonian of the 2D Ising model.

    """
    if isinstance(grid, tuple) and len(grid) == 3:
        grid = _grid_from_structure(grid[0], grid[1], grid[2])
    pairs = _find_nn_pairs(grid)
    return _abstract_ising_model(pairs, ext_magn, coupling, ext_magn_op, nn_op)

def _grid_from_structure(prefix: str,
                         rows: int,
                         cols: int
                         ) -> ndarray:
    """
    Creates a grid structure from a given prefix and the number of rows and
    columns.

    Args:
        prefix (str): The prefix for the node identifiers.
        rows (int): The number of rows.
        cols (int): The number of columns.
    
    Returns:
        ndarray: The grid structure.

    """
    positivity_check(rows, "rows")
    positivity_check(cols, "cols")
    grid = zeros((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            grid[i,j] = f"{prefix}{i}_{j}"
    return grid

def _find_nn_pairs(grid: ArrayLike) -> List[Tuple[str,str]]:
    """
    Finds the nearest neighbour pairs in a given grid.

    Args:
        grid (ArrayLike): The grid structure.
    
    Returns:
        List[Tuple[str,str]]: The nearest neighbour pairs.

    """
    pairs = []
    rows, cols = grid.shape
    # We only want unique pairs, so len-1
    for i in range(rows):
        for j in range(cols):
            if i < rows-1:
                pairs.append((grid[i][j],grid[i+1][j]))
            if j < cols-1:
                pairs.append((grid[i][j],grid[i][j+1]))
    return pairs

def _pairs_to_list(pairs: List[Tuple[str,str]]
                   ) -> List[str]:
    """
    Converts a list of pairs to a list of identifiers.

    Args:
        pairs (List[Tuple[str,str]]): The list of pairs to convert.
    
    Returns:
        List[str]: The list of identifiers.
    """
    identifiers = []
    for pair in pairs:
        identifiers.append(pair[0])
        identifiers.append(pair[1])
    return list(set(identifiers))  # Remove duplicates by converting to set

def _adapt_structure_for_single_site(structure: Union[TreeStructure, List[Tuple[str, str]]]
                                     ) -> TreeStructure | List[str]:
    """
    Adapts the structure for single site Hamiltonian creation.

    Args:
        structure (Union[TreeStructure, List[Tuple[str, str]]]): The structure
            to adapt. Can either be a TreeStructure object or a list of tuples
            of nearest neighbours.

    Returns:
        List[str]: The adapted structure as a list of identifiers.
    """
    if isinstance(structure, TreeStructure):
        return structure
    return _pairs_to_list(structure)

def bose_hubbard_model(
        structure: Union[TreeStructure, List[Tuple[str, str]]],
        local_dim: int = 2,
        hopping: float = 1.0,
        on_site_int: float = 1.0,
        chem_pot: float = 0.0
                    ) -> Hamiltonian:
    """
    Generates the Bose-Hubbard model Hamiltonian for a given structure.

    ..math::
        H = -t \sum_{i,j} a_i^\dagger a_j + U \sum_i n_i(n_i-1) - \mu \sum_i n_i

    where :math:`a_i^\dagger` and :math:`a_i` are the creation and annihilation
    operators, :math:`n_i` is the number operator, :math:`t` is the hopping
    strength, :math:`U` is the on-site interaction strength and :math:`\mu`
    is the chemical potential.

    Args:
        structure (Union[TreeStructure, List[Tuple[str, str]]]): The nearest
            neighbour identifiers. They can either be given as a TreeStructure
            obsject, and be inferred, or as a list of tuples of nearest
            neighbours.
        local_dim (int): The local to be truncated to, i.e. the maximum number
            of particles per site. Defaults to 2.
        hopping (float): The hopping strength between the nearest neighbours.
            Defaults to 1.0.
        on_site_int (float): The on-site interaction strength. Defaults to 1.0.
        chem_pot (float): The chemical potential. Defaults to 0.0.
    """
    if local_dim < 2:
        errstr = "The local dimension must be at least 2 for the Bose-Hubbard model!"
        raise ValueError(errstr)
    # Prepare operators
    cr, an, num = bosonic_operators(dimension=local_dim)
    ident = eye(local_dim, dtype=complex)
    num_m_eye = num - ident
    on_site_op = num @ num_m_eye
    bose_hub_ham = Hamiltonian()
    bose_hub_ham.include_identities([1, local_dim])
    # Create the chemical potential terms
    if chem_pot != 0.0:
        single_site_structure = _adapt_structure_for_single_site(structure)
        chem_ham = create_single_site_hamiltonian(single_site_structure,
                                                "number",
                                                factor=(Fraction(-1), "chem_pot"),
                                                conversion_dict={"number": num},
                                                coeffs_mapping={"chem_pot": chem_pot})
        bose_hub_ham.add_hamiltonian(chem_ham)
    # Create on-site interaction
    if on_site_int != 0.0:
        single_site_structure = _adapt_structure_for_single_site(structure)
        on_site_ham = create_single_site_hamiltonian(single_site_structure,
                                                     "on_site_op",
                                                     factor=(Fraction(-1,2), "on_site_int"),
                                                     conversion_dict={"on_site_op": on_site_op},
                                                     coeffs_mapping={"on_site_int": on_site_int})
        bose_hub_ham.add_hamiltonian(on_site_ham)
    # Create hopping terms
    if hopping != 0.0:
        nn_ham1 = create_nearest_neighbour_hamiltonian(structure,
                                                    "creation",
                                                    factor=(Fraction(-1), "hopping"),
                                                    local_operator2="annihilation")
        nn_ham2 = create_nearest_neighbour_hamiltonian(structure,
                                                    "annihilation",
                                                    factor=(Fraction(-1), "hopping"),
                                                    local_operator2="creation",
                                                    conversion_dict={"creation": cr,
                                                                     "annihilation": an},
                                                    coeffs_mapping={"hopping": hopping})
        bose_hub_ham.add_hamiltonian(nn_ham1)
        bose_hub_ham.add_hamiltonian(nn_ham2)
    return bose_hub_ham
