"""
This file implements two site interaction models for quantum systems.

A two site model contains only terms that act non-trivially on two sites or
one site.
"""
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ..hamiltonian import Hamiltonian
from ...core.ttn import TreeTensorNetwork
from ...special_ttn.star import StarTreeTensorState
from ..common_operators import (ket_i,
                                pauli_matrices,
                                bosonic_operators)
from ...util.experiment_util.sim_params import SimulationParameters
from ..sim_operators import (create_nearest_neighbour_hamiltonian,
                             create_single_site_hamiltonian)
from .abc_model import Model

@dataclass
class TwoSiteParameters(SimulationParameters):
    """
    Collects all parameters of a two site model.

    Attributes:
        interaction_range (int): The range of interaction between sites.
            Default is 1, meaning nearest neighbour interactions
    """
    interaction_range: int = 1

class TwoSiteModel(Model):
    """
    A two site model of a quantum system.

    This model contains only terms that act non-trivially on two sites or
    one site.

    Attributes:
        interaction_range (int): The range of interaction between sites.
            Default is 1, meaning nearest neighbour interactions.
    """

    def __init__(self, interaction_range: int = 1) -> None:
        """
        Initializes the two site model with a given interaction range.

        Args:
            interaction_range (int): The range of interaction between sites.
                Default is 1, meaning nearest neighbour interactions.
        """
        super().__init__()
        self.interaction_range = interaction_range

    def generate_chain_structure(self,
                                 site_ids: list[str]
                                 ) -> list[tuple[str,str]]:
        """
        Generates the chain structure for the given site IDs.
        """
        return [(site_ids[i], site_ids[i + self.interaction_range])
                for i in range(len(site_ids) - self.interaction_range)]

    def generate_t_topology_structure(self,
                                      site_ids: tuple[list[str], list[str], list[str]]
                                      ) -> list[tuple[str,str]]:
        """
        Generates the T topology structure for the given site IDs.
        """
        centre_tensor = ket_i(0,1).reshape((1,1,1,1))
        other_tensors = [ket_i(0,2).reshape((1,1,2))
                         for _ in site_ids[0][:-1]] + [ket_i(0,2).reshape((1,2))]
        tttns = StarTreeTensorState.from_tensor_lists(centre_tensor,
                                                      [other_tensors,other_tensors,other_tensors],
                                                      identifiers=list(site_ids)
                                                      )
        pairs = tttns.find_pairs_of_distance(self.interaction_range,
                                             consider_open=True)
        pairs = [tuple(pair) for pair in pairs]
        return pairs

    def generate_2d_structure(self,
                              site_ids: list[list[str]]
                              ) -> list[tuple[str, str]]:
        """
        Generates a 2D structure for the given site IDs.
        """
        pairs = []
        num_rows = len(site_ids)
        num_cols = len(site_ids[0]) if num_rows > 0 else 0
        # We only want unique pairs, so len-1
        for i in range(num_rows):
            for j in range(num_cols):
                if i < num_rows-self.interaction_range:
                    pairs.append((site_ids[i][j],site_ids[i+self.interaction_range][j]))
                if j < num_cols-self.interaction_range:
                    pairs.append((site_ids[i][j],site_ids[i][j+self.interaction_range]))
        return pairs

@dataclass
class HeisenbergParameters(TwoSiteParameters):
    """
    Parameters for the Heisenberg model.

    Attributes:
        x_factor (float): The factor for the x-component of the interaction.
        y_factor (float): The factor for the y-component of the interaction.
            If None, it is set to the same as `x_factor`.
        z_factor (float): The factor for the z-component of the interaction.
            if None, it is set to the same as `x_factor`.
        ext_x (float): The external magnetic field in the x-direction. If None
            it is set to the same as `ext_z`.
        ext_y (float): The external magnetic field in the y-direction. If None
            it is set to the same as `ext_y`.
        ext_z (float): The external magnetic field in the z-direction.
    """
    x_factor: float = 1.0
    y_factor: float | None = None
    z_factor: float | None = None
    ext_x: float | None = 0.0
    ext_y: float | None = 0.0
    ext_z: float = 0.0

class HeisenbergModel(TwoSiteModel):
    """
    Represents the Heisenberg Model

    .. math::
        - \sum_{<i,j>} J_x X_iX_j + J_y Y_iY_j + J_z Z_iZ_j - \sum_i (g_x X_i + g_y Y_i + g_z Z_i)
    
    """

    def __init__(self,
                 interaction_range: int = 1,
                 x_factor: float = 1.0,
                 y_factor: float | None = 0.0,
                 z_factor: float | None = 0.0,
                 ext_x: float | None = 0.0,
                 ext_y: float | None = 0.0,
                 ext_z: float = 0.0,
                 ) -> None:
        """
        Initialises a new HeisenbergModel object.

        Args:
            interaction_range (int): The range of interaction between sites.
                Default is 1, meaning nearest neighbour interactions.
            x_factor (float): The factor for the x-component of the interaction.
            y_factor (float): The factor for the y-component of the interaction.
                If None, it is set to the same as `x_factor`.
            z_factor (float): The factor for the z-component of the interaction.
                if None, it is set to the same as `x_factor`.
            ext_x (float): The external magnetic field in the x-direction. If None
                it is set to the same as `ext_z`.
            ext_y (float): The external magnetic field in the y-direction. If None
                it is set to the same as `ext_y`.
            ext_z (float): The external magnetic field in the z-direction. 
        """
        super().__init__(interaction_range)
        self.factors = [x_factor,y_factor,z_factor]
        self.ext_magns = [ext_x,ext_y,ext_z]

        self.pauli_symbols = ["X", "Y", "Z"]
        self.factor_prefix = "J_"
        self.ext_magn_prefix = "ext_"

    def generate_hamiltonian(self,
                             structure: TreeTensorNetwork | list[tuple[str, str]]
                             ) -> Hamiltonian:
        """
        Generates the Hamiltonian for the given structure of the quantum system.

        Uses the internal parameters of the model to generate the Hamiltonian.

        Args:
            structure (TreeTensorNetwork | list[tuple[str, ...]]): The structure
                of the quantum system.

        Returns:
            Hamiltonian: The Hamiltonian of the quantum system.
        """
        ham = Hamiltonian()
        ham.include_identities([1,2])
        factors = [(self.factor_prefix + "x", self.factors[0]),
                   (self.factor_prefix + "y", 1),
                   (self.factor_prefix + "z", 1)]
        ext_magns = [(self.ext_magn_prefix + "x", 1),
                     (self.ext_magn_prefix + "y", 1),
                     (self.ext_magn_prefix + "z", self.ext_magns[2])]
        for i in range(1,len(self.factors)):
            if self.factors[i] is None:
                factors[i] = factors[0]
            else:
                factors[i] = (factors[i][0], self.factors[i])
        for i in range(len(self.ext_magns) - 1):
            if self.ext_magns[i] is None:
                ext_magns[i] = ext_magns[2]
            else:
                ext_magns[i] = (ext_magns[i][0],self.ext_magns[i])
        paulis = pauli_matrices()
        single_site_structure = _adapt_structure_for_single_site(structure)
        for i in range(3):
            if factors[i][1] != 0.0:
                conv_dict = {self.pauli_symbols[i]: paulis[i]}
                coeff_map = {factors[i][0]: factors[i][1]}
                nn_ham = create_nearest_neighbour_hamiltonian(structure,
                                                            self.pauli_symbols[i],
                                                            (Fraction(-1),factors[i][0]),
                                                            conversion_dict=conv_dict,
                                                            coeffs_mapping=coeff_map)
                ham.add_hamiltonian(nn_ham)
            if self.ext_magns[i] != 0.0:
                conv_dict = {self.pauli_symbols[i]: paulis[i]}
                coeff_map = {ext_magns[i][0]: ext_magns[i][1]}
                ss_ham = create_single_site_hamiltonian(single_site_structure,
                                                        self.pauli_symbols[i],
                                                        factor=(Fraction(-1),ext_magns[i][0]),
                                                        conversion_dict=conv_dict,
                                                        coeffs_mapping=coeff_map)
                ham.add_hamiltonian(ss_ham)
        return ham

@dataclass
class IsingParameters(TwoSiteParameters):
    """
    Parameters used to define as Ising model.
    """
    factor: float = 1.0
    ext_magn: float = 0.0

class IsingModel(HeisenbergModel):
    """
    A class implementing the Ising model

    ..math::
        -J \sum_{<i,j>} X_i X_j -g \sum_i Z_i

    """

    def __init__(self,
                 interaction_range: int = 1,
                 factor: float = 1.0,
                 ext_magn: float = 0.0
                 ) -> None:
        """
        Initialises an IsingModel object.

        Args:
            interaction_range (int): The range of interaction between sites.
                Default is 1, meaning nearest neighbour interactions.
            factor (float): The prefactor of the two site interaction.
                Defaults to 1.0.
            ext_magn (float): The prefactor of the single site interaction.
                Defaults to 0.0.
        """
        super().__init__(interaction_range=interaction_range,
                         x_factor=factor,
                         ext_z=ext_magn)

    @property
    def factor(self) -> float:
        """
        Returns the two site interaction factor.
        """
        return self.factors[0]

    @factor.setter
    def factor(self, new_factor: float):
        """
        Sets the two site interaction.
        """
        self.factors[0] = new_factor

    @property
    def ext_magn(self) -> float:
        """
        Returns the single site factor.
        """
        return self.ext_magns[-1]

    @ext_magn.setter
    def ext_magn(self, new_magn: float):
        """
        Sets the single site factor.
        """
        self.ext_magns[-1] = new_magn

class FlippedIsingModel(HeisenbergModel):
    """
    A class implementing the flipped Ising model

    ..math::
        -J \sum_{<i,j>} Z_i Z_j -g \sum_i X_i

    """

    def __init__(self,
                 interaction_range: int = 1,
                 factor: float = 1.0,
                 ext_magn: float = 0.0
                 ) -> None:
        """
        Initialises an IsingModel object.

        Args:
            interaction_range (int): The range of interaction between sites.
                Default is 1, meaning nearest neighbour interactions.
            factor (float): The prefactor of the two site interaction.
                Defaults to 1.0.
            ext_magn (float): The prefactor of the single site interaction.
                Defaults to 0.0.
        """
        super().__init__(interaction_range=interaction_range,
                         x_factor=0.0,
                         z_factor=factor,
                         ext_x=ext_magn,
                         ext_z=0.0)

    @property
    def factor(self) -> float:
        """
        Returns the two site interaction factor.
        """
        return self.factors[-1]

    @factor.setter
    def factor(self, new_factor: float):
        """
        Sets the two site interaction.
        """
        self.factors[-1] = new_factor

    @property
    def ext_magn(self) -> float:
        """
        Returns the single site factor.
        """
        return self.ext_magns[0]

    @ext_magn.setter
    def ext_magn(self, new_magn: float):
        """
        Sets the single site factor.
        """
        self.ext_magns[0] = new_magn

@dataclass
class BoseHubbardParameters(TwoSiteParameters):
    """
    Parameters for the Bose Hubbard model.

    Attributes:
        local_dim (int): The local dimension of the system, i.e. the maximum
            number of particles per site.
        hopping (float): The hopping strength between the nearest neighbours.
        on_site_int (float): The on-site interaction strength.
        chem_pot (float): The chemical potential.
    """
    local_dim: int = 2
    hopping: float = 1.0
    on_site_int: float = 1.0
    chem_pot: float = 0.0

class BoseHubbardModel(TwoSiteModel):
    """
    A class implementing the Bose Hubbard model.

    ..math::
        H = -t \sum_{i,j} (a_i^\dagger a_j + a_j^\dagger a_i) + U \sum_i n_i(n_i-1) - \mu \sum_i n_i

    where :math:`a_i^\dagger` and :math:`a_i` are the creation and annihilation
    operators, :math:`n_i` is the number operator, :math:`t` is the hopping
    strength, :math:`U` is the on-site interaction strength and :math:`\mu`
    is the chemical potential.

    """

    def __init__(self,
                 interaction_range: int = 1,
                 local_dim: int = 2,
                 hopping: float = 1.0,
                 on_site_int: float = 1.0,
                 chem_pot: float = 0.0
                 ) -> None:
        """
        Initialises a BoseHubbardModel object.

        Args:
            local_dim (int): The local to be truncated to, i.e. the maximum number
                of particles per site. Defaults to 2.
            hopping (float): The hopping strength between the nearest neighbours.
                Defaults to 1.0.
            on_site_int (float): The on-site interaction strength. Defaults to 1.0.
            chem_pot (float): The chemical potential. Defaults to 0.0.
        """
        super().__init__(interaction_range)
        if local_dim < 2:
            errstr = "The local dimension must be at least 2 for the Bose-Hubbard model!"
            raise ValueError(errstr)
        self.local_dim = local_dim
        self.hopping = hopping
        self.on_site_int = on_site_int
        self.chem_pot = chem_pot

        # Define symbolic values for the operators
        self.creation_symb = "creation"
        self.annihilation_symb = "annihilation"
        self.number_symb = "number"
        self.on_site_op_symb = "on_site_op"

        # Define the symbolic values for the coefficients
        self.hopping_symb = "hopping"
        self.on_site_int_symb = "on_site_int"
        self.chem_pot_symb = "chem_pot"

    def symbolic_operators(self) -> list[str]:
        """
        Returns a list of symbolic operators used in the model.

        Returns:
            list[str]: A list of symbolic operators.
        """
        return [self.creation_symb,
                self.annihilation_symb,
                self.number_symb,
                self.on_site_op_symb]

    def symbolic_coefficients(self) -> list[str]:
        """
        Returns a list of symbolic coefficients used in the model.

        Returns:
            list[str]: A list of symbolic coefficients.
        """
        return [self.hopping_symb,
                self.on_site_int_symb,
                self.chem_pot_symb]

    def generate_hamiltonian(self,
                             structure: TreeTensorNetwork | list[tuple[str, str]]
                             ) -> Hamiltonian:
        """
        Generates the Hamiltonian for the given structure of the quantum system.

        Uses the internal parameters of the model to generate the Hamiltonian.

        Args:
            structure (TreeTensorNetwork | list[tuple[str, ...]]): The structure
                of the quantum system.

        Returns:
            Hamiltonian: The Hamiltonian of the quantum system.
        """
        # Prepare operators
        cr, an, num = bosonic_operators(dimension=self.local_dim)
        ident = np.eye(self.local_dim,
                       dtype=complex)
        num_m_eye = num - ident
        on_site_op = num @ num_m_eye
        bose_hub_ham = Hamiltonian()
        bose_hub_ham.include_identities([1, self.local_dim])
        single_site_structure = _adapt_structure_for_single_site(structure)
        # Create the chemical potential terms
        if self.chem_pot != 0.0:
            conv_dict = {self.number_symb: num}
            coeffs_map = {self.chem_pot_symb: self.chem_pot}
            factor = (Fraction(-1), self.chem_pot_symb)
            chem_ham = create_single_site_hamiltonian(single_site_structure,
                                                    self.number_symb,
                                                    factor=factor,
                                                    conversion_dict=conv_dict,
                                                    coeffs_mapping=coeffs_map)
            bose_hub_ham.add_hamiltonian(chem_ham)
        # Create on-site interaction
        if self.on_site_int != 0.0:
            conv_dict = {self.on_site_op_symb: on_site_op}
            coeffs_map = {self.on_site_int_symb: self.on_site_int}
            factor = (Fraction(-1,2), self.on_site_int_symb)
            on_site_ham = create_single_site_hamiltonian(single_site_structure,
                                                        self.on_site_op_symb,
                                                        factor=factor,
                                                        conversion_dict=conv_dict,
                                                        coeffs_mapping=coeffs_map)
            bose_hub_ham.add_hamiltonian(on_site_ham)
        # Create hopping terms
        if self.hopping != 0.0:
            conv_dict = {self.creation_symb: cr,
                         self.annihilation_symb: an}
            coeffs_mapping = {self.hopping_symb: self.hopping}
            factor = (Fraction(-1), self.hopping_symb)
            nn_ham1 = create_nearest_neighbour_hamiltonian(structure,
                                                        self.creation_symb,
                                                        factor=factor,
                                                        local_operator2=self.annihilation_symb)
            nn_ham2 = create_nearest_neighbour_hamiltonian(structure,
                                                        self.annihilation_symb,
                                                        factor=factor,
                                                        local_operator2=self.creation_symb,
                                                        conversion_dict=conv_dict,
                                                        coeffs_mapping=coeffs_mapping)
            bose_hub_ham.add_hamiltonian(nn_ham1)
            bose_hub_ham.add_hamiltonian(nn_ham2)
        return bose_hub_ham

def _pairs_to_list(pairs: list[tuple[str,str]]
                   ) -> list[str]:
    """
    Converts a list of pairs to a list of identifiers.

    Args:
        pairs (list[tuple[str,str]]): The list of pairs to convert.
    
    Returns:
        list[str]: The list of identifiers.
    """
    identifiers = set()
    for pair in pairs:
        identifiers.add(pair[0])
        identifiers.add(pair[1])
    return list(identifiers)  # Remove duplicates by converting to set

def _adapt_structure_for_single_site(structure: TreeTensorNetwork | list[tuple[str, str]]
                                     ) -> TreeTensorNetwork | list[str]:
    """
    Adapts the structure for single site Hamiltonian creation.

    Args:
        structure (TreeTensorNetwork | list[tuple[str, str]]): The structure
            to adapt. Can either be a TreeTensorNetwork object or a list of tuples
            of nearest neighbours.

    Returns:
        list[str]: The adapted structure as a list of identifiers.
    """
    if isinstance(structure, TreeTensorNetwork):
        return structure
    return _pairs_to_list(structure)
        