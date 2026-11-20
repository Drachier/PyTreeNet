"""
Implements the common functionality for the operator power application simulation.

This experiment applies powers of an operator (by default Ising model) to a state
using different TTNO-TTNS application algorithms.

The experiment works by sequentially applying the operator multiple times:
- Start with state |ψ⟩
- Apply H once: |ψ₁⟩ = H|ψ⟩
- Apply H to |ψ₁⟩: |ψ₂⟩ = H|ψ₁⟩ = H²|ψ⟩
- etc.

This allows us to study how truncation errors accumulate with repeated application.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import time
import os
from copy import deepcopy
from enum import Enum

from h5py import File

from pytreenet.util.experiment_util.sim_params import SimulationParameters
from pytreenet.ttns.ttns_ttno.application import (ApplicationMethod,
                                                  apply_ttno_to_ttns)
from pytreenet.random.random_special_ttns import random_ttns
from pytreenet.ttno.ttno_class import TreeTensorNetworkOperator
from pytreenet.operators.models.two_site_model import IsingModel, HeisenbergModel, FlippedIsingModel
from pytreenet.operators.models.abc_model import Model
from pytreenet.special_ttn.special_states import TTNStructure
from pytreenet.time_evolution.results import Results
from pytreenet.util.tensor_splitting import SVDParameters


class OperatorType(Enum):
    """
    Enumeration for different operator types that can be used in the simulation.
    """
    ISING = "ising"
    HEISENBERG = "heisenberg"
    FLIPPED_ISING = "flipped_ising"


@dataclass
class PowerApplicationParams(SimulationParameters):
    """
    Parameters for the operator power application simulation.

    Attributes:
        structure (TTNStructure): The structure of the TTNS.
        sys_size (int): The system size.
        phys_dim (int): The physical dimension.
        bond_dim (int): The bond dimension of the random TTNS.
        appl_method (ApplicationMethod): The application method to use.
        max_target_bond_dim (int): The maximum target bond dimension for truncation.
        min_target_bond_dim (int): The minimum target bond dimension for truncation.
        step_target_bond_dim (int): The step size for target bond dimensions.
        seed (int): The random seed.
        operator_type (OperatorType): The type of operator to use.
        max_power (int): The maximum power to apply.
    """
    structure: TTNStructure = TTNStructure.MPS
    sys_size: int = 10
    phys_dim: int = 2
    bond_dim: int = 10
    appl_method: ApplicationMethod = ApplicationMethod.DIRECT
    max_target_bond_dim: int = 10
    min_target_bond_dim: int = 1
    step_target_bond_dim: int = 1
    seed: int = 12334
    operator_type: OperatorType = OperatorType.ISING
    max_power: int = 5

    def get_model(self) -> Model:
        """
        Returns the model instance based on the operator_type.

        Returns:
            Model: The model instance.
        """
        if self.operator_type == OperatorType.ISING:
            # Hardcoded Ising model parameters
            return IsingModel(interaction_range=1, factor=1.0, ext_magn=0.5)
        elif self.operator_type == OperatorType.HEISENBERG:
            # Hardcoded Heisenberg model parameters
            return HeisenbergModel(interaction_range=1, 
                                  x_factor=1.0, y_factor=1.0, z_factor=1.0,
                                  ext_x=0.0, ext_y=0.0, ext_z=0.0)
        elif self.operator_type == OperatorType.FLIPPED_ISING:
            # Hardcoded Flipped Ising model parameters
            return FlippedIsingModel(interaction_range=1, factor=1.0, ext_magn=0.5)
        else:
            raise ValueError(f"Unknown operator type: {self.operator_type}")


RES_IDS = ("power", "bond_dim", "trunc_error", "run_time")


def init_results(params: PowerApplicationParams) -> Results:
    """
    Initializes the results object for the simulation.

    Args:
        params (PowerApplicationParams): The parameters for the simulation.

    Returns:
        Results: The initialized and empty results object.
    """
    num_res = len(list(bond_dim_range(params))) * params.max_power
    results = Results()
    res_dtypes = (int, int, float, float)
    results.initialize(dict(zip(RES_IDS, res_dtypes)),
                       num_res,
                       with_time=False)
    return results


def set_result_values(results: Results,
                      index: int,
                      power: int,
                      bond_dim: int,
                      trunc_error: float,
                      run_time: float
                      ) -> None:
    """
    Sets the result values for a specific index in the results object.

    Args:
        results (Results): The results object.
        index (int): The index to set the values for.
        power (int): The power of the operator.
        bond_dim (int): The bond dimension used in the truncation.
        trunc_error (float): The truncation error.
        run_time (float): The time taken to perform the truncation.
    """
    res_values = (power, bond_dim, trunc_error, run_time)
    for res_id, res_value in zip(RES_IDS, res_values):
        results.set_element(res_id, index, res_value)


def bond_dim_range(params: PowerApplicationParams) -> range:
    """
    Generates a range of bond dimensions to be used in the simulation.

    Args:
        params (PowerApplicationParams): The parameters for the simulation.

    Returns:
        range: A range of bond dimensions from min to max with the specified step.
    """
    return range(params.min_target_bond_dim,
                 params.max_target_bond_dim + 1,
                 params.step_target_bond_dim)


def generate_operator_ttno(model: Model,
                           structure: TTNStructure,
                           sys_size: int,
                           phys_dim: int,
                           bond_dim: int) -> TreeTensorNetworkOperator:
    """
    Generates a TTNO from the given model.

    Args:
        model (Model): The model to use for generating the Hamiltonian.
        structure (TTNStructure): The structure of the TTNS.
        sys_size (int): The system size.
        phys_dim (int): The physical dimension.
        bond_dim (int): The bond dimension.

    Returns:
        TreeTensorNetworkOperator: The generated TTNO.
    """
    # Generate the structure for the model
    if structure == TTNStructure.MPS:
        site_ids = [f"site_{i}" for i in range(sys_size)]
        pairs = model.generate_chain_structure(site_ids)
    elif structure == TTNStructure.TSTAR:
        arm1_ids = [f"arm1_{i}" for i in range(sys_size // 3)]
        arm2_ids = [f"arm2_{i}" for i in range(sys_size // 3)]
        arm3_ids = [f"arm3_{i}" for i in range(sys_size // 3)]
        # Adjust for systems that don't divide evenly by 3
        while len(arm1_ids) + len(arm2_ids) + len(arm3_ids) < sys_size:
            arm1_ids.append(f"arm1_{len(arm1_ids)}")
        site_ids = (arm1_ids, arm2_ids, arm3_ids)
        pairs = model.generate_t_topology_structure(site_ids)
    elif structure == TTNStructure.BINARY:
        # For binary tree, we use a 2D structure
        depth = sys_size
        num_sites = 2 ** depth
        # Create a 2D grid
        rows = int(num_sites ** 0.5)
        cols = (num_sites + rows - 1) // rows
        site_ids_2d = [[f"site_{i}_{j}" for j in range(cols)] for i in range(rows)]
        pairs = model.generate_2d_structure(site_ids_2d)
    elif structure == TTNStructure.FTPS:
        # For FTPS, we use a 2D structure
        side_length = int(sys_size ** 0.5)
        site_ids_2d = [[f"site_{i}_{j}" for j in range(side_length)] 
                       for i in range(side_length)]
        pairs = model.generate_2d_structure(site_ids_2d)
    else:
        raise ValueError(f"Unknown structure: {structure}")
    
    # Generate the Hamiltonian
    hamiltonian = model.generate_hamiltonian(pairs)
    
    # Get the reference tree structure from a TTNS of the same type
    ttns = random_ttns(structure, sys_size, phys_dim, bond_dim, seed=42)
    reference_tree = ttns.tree.copy()
    
    # Convert Hamiltonian to TTNO
    ttno = TreeTensorNetworkOperator.from_hamiltonian(hamiltonian,
                                                      reference_tree)
    return ttno


def run_simulation(params: PowerApplicationParams) -> Results:
    """
    Runs the operator power application simulation.

    This applies the operator sequentially multiple times:
    - Start with state |ψ⟩
    - Apply H once: |ψ₁⟩ = H|ψ⟩
    - Apply H to |ψ₁⟩: |ψ₂⟩ = H|ψ₁⟩ = H²|ψ⟩
    - etc.

    For each power and each bond dimension, we measure the truncation error
    compared to applying with a high bond dimension (exact).

    Args:
        params (PowerApplicationParams): The parameters for the simulation.

    Returns:
        Results: The results of the simulation.
    """
    # Generate random TTNS
    ttns = random_ttns(params.structure,
                       params.sys_size,
                       params.phys_dim,
                       params.bond_dim,
                       seed=params.seed)
    
    # Get the model and generate the operator TTNO
    model = params.get_model()
    operator_ttno = generate_operator_ttno(model,
                                           params.structure,
                                           params.sys_size,
                                           params.phys_dim,
                                           params.bond_dim)
    
    results = init_results(params)
    
    # Precompute exact results for each power using DIRECT method (no truncation)
    exact_states = {}
    current_exact = deepcopy(ttns)
    exact_states[0] = current_exact
    
    for power in range(1, params.max_power + 1):
        # DIRECT method doesn't truncate, so it's exact
        current_exact = apply_ttno_to_ttns(current_exact, deepcopy(operator_ttno),
                                          ApplicationMethod.DIRECT)
        exact_states[power] = deepcopy(current_exact)
    
    # Now run the actual simulations with different bond dimensions
    for power in range(1, params.max_power + 1):
        for idx, bond_dim in enumerate(bond_dim_range(params)):
            index = (power - 1) * len(list(bond_dim_range(params))) + idx
            
            svd_params = SVDParameters(max_bond_dim=bond_dim)
            
            if params.appl_method == ApplicationMethod.DIRECT:
                # DIRECT method doesn't truncate, so no parameters needed
                kwargs = {}
            elif params.appl_method == ApplicationMethod.VARIATIONAL:
                kwargs = {"num_sweeps": 2,
                          "svd_params": svd_params}
            elif params.appl_method == ApplicationMethod.ZIPUP_VARIATIONAL:
                kwargs = {"num_sweeps": 2,
                          "var_svd_params": svd_params,
                          "zipup_svd_params": svd_params}
            elif params.appl_method == ApplicationMethod.HALF_DENSITY_MATRIX_VARIATIONAL:
                kwargs = {"num_sweeps": 2,
                          "var_svd_params": svd_params,
                          "dm_svd_params": svd_params}
            elif params.appl_method == ApplicationMethod.SRC:
                kwargs = {"desired_dimension": bond_dim}
            elif params.appl_method == ApplicationMethod.DIRECT_TRUNCATE:
                kwargs = {"params": svd_params}
            elif params.appl_method == ApplicationMethod.DIRECT_TRUNCATE_RANDOM:
                svd_params.random = True
                kwargs = {"params": svd_params}
            else:
                kwargs = {"svd_params": svd_params}
            
            func = params.appl_method.get_function()
            
            # Start from the original state and apply the operator 'power' times
            current_state = deepcopy(ttns)
            start_time = time()
            
            for _ in range(power):
                current_state = func(current_state, deepcopy(operator_ttno), **kwargs)
            
            end_time = time()
            
            # Compare with exact result
            trunc_error = exact_states[power].distance(current_state, normalise=True)
            
            set_result_values(results, index, power, bond_dim, trunc_error,
                              end_time - start_time)
    
    return results


def save_results(params: PowerApplicationParams,
                 results: Results,
                 save_directory: str) -> None:
    """
    Saves the simulation results to an HDF5 file.

    Args:
        params (PowerApplicationParams): The parameters for the simulation.
        results (Results): The results of the simulation.
        save_directory (str): The directory to save the results in.
    """
    filename = params.get_hash() + ".h5"
    filepath = os.path.join(save_directory, filename)
    with File(filepath, "w") as f:
        results.save_to_h5(f)
        params.save_to_h5(f)


def run_and_save(params: PowerApplicationParams,
                 save_directory: str) -> None:
    """
    Runs the power application simulation and saves the result.

    Args:
        params (PowerApplicationParams): The parameters for the simulation.
        save_directory (str): The directory to save the results in.
    """
    results = run_simulation(params)
    save_results(params, results, save_directory)
