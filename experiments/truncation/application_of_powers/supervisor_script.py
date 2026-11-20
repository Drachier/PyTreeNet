"""
This script is used to supervise the operator power application experiments.
"""
import os
from copy import copy
from itertools import product

from pytreenet.util.experiment_util.supervisor import (Supervisor,
                                                       SIMSCRIPT_STANDARD_NAME)
from pytreenet.ttns.ttns_ttno.application import ApplicationMethod
from pytreenet.special_ttn.special_states import TTNStructure

from application_common import PowerApplicationParams, OperatorType


def generate_parameter_set() -> list[PowerApplicationParams]:
    """
    Generates a set of parameters for operator power application experiments.

    Returns:
        list[PowerApplicationParams]: A list of simulation parameters.
    """
    phys_dim = 2
    structures = (TTNStructure.MPS,
                  TTNStructure.TSTAR,
                  TTNStructure.BINARY)
    methods = (
        ApplicationMethod.DENSITY_MATRIX,
        ApplicationMethod.SRC,
        ApplicationMethod.ZIPUP,
        ApplicationMethod.DIRECT_TRUNCATE,
        ApplicationMethod.DIRECT_TRUNCATE_RANDOM,
    )
    operator_types = (OperatorType.ISING, )
    seeds = (1234, 4321)
    bond_dims = (5, 10)
    num_powers_list = (3, 5)
    
    param_set = []
    for structure, method, operator_type, seed, bond_dim, num_powers in product(
        structures, methods, operator_types, seeds, bond_dims, num_powers_list):
        
        if structure is TTNStructure.MPS:
            sys_size = 20
        elif structure is TTNStructure.BINARY:
            sys_size = 5  # depth of binary tree
        elif structure is TTNStructure.TSTAR:
            sys_size = 12  # arm length
        else:
            raise ValueError(f"Unknown structure: {structure}")
        
        params = PowerApplicationParams(
            structure=structure,
            sys_size=sys_size,
            phys_dim=phys_dim,
            bond_dim=bond_dim,
            appl_method=method,
            max_target_bond_dim=bond_dim + 1,
            min_target_bond_dim=2,
            step_target_bond_dim=1,
            seed=seed,
            operator_type=operator_type,
            num_powers=num_powers,
            max_power=num_powers
        )
        param_set.append(params)
        
        # For MPS and zipup, also test canonical zipup
        if structure is TTNStructure.MPS and method == ApplicationMethod.ZIPUP:
            params_canonical = copy(params)
            params_canonical.appl_method = ApplicationMethod.ZIPUP_CANONICAL
            param_set.append(params_canonical)
    
    return param_set


def generate_small_test_set() -> list[PowerApplicationParams]:
    """
    Generates a small set of parameters for testing.

    Returns:
        list[PowerApplicationParams]: A small list of simulation parameters.
    """
    param_set = []
    
    # MPS with Ising model
    params = PowerApplicationParams(
        structure=TTNStructure.MPS,
        sys_size=10,
        phys_dim=2,
        bond_dim=5,
        appl_method=ApplicationMethod.DIRECT,
        max_target_bond_dim=6,
        min_target_bond_dim=2,
        step_target_bond_dim=1,
        seed=1234,
        operator_type=OperatorType.ISING,
        num_powers=3,
        max_power=3
    )
    param_set.append(params)
    
    # MPS with Heisenberg model
    params = PowerApplicationParams(
        structure=TTNStructure.MPS,
        sys_size=10,
        phys_dim=2,
        bond_dim=5,
        appl_method=ApplicationMethod.SRC,
        max_target_bond_dim=6,
        min_target_bond_dim=2,
        step_target_bond_dim=1,
        seed=4321,
        operator_type=OperatorType.HEISENBERG,
        num_powers=3,
        max_power=3
    )
    param_set.append(params)
    
    return param_set


def main():
    """
    Main function to set up and run the supervisor for operator power application experiments.
    """
    # Use small test set for now
    param_set = generate_small_test_set()
    
    # The simulation script lies in the same directory as this script
    sim_script_path = os.path.join(os.path.dirname(__file__),
                                   SIMSCRIPT_STANDARD_NAME)
    supervisor = Supervisor.from_commandline(param_set,
                                             sim_script_path)
    supervisor.timeout = 1 * 60 * 60  # 1 hour
    supervisor.run_simulations()


if __name__ == "__main__":
    main()
