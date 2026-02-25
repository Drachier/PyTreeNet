"""
This script is used to supervise the addition comparison experiments.
"""
import os
from itertools import product

from pytreenet.util.experiment_util.supervisor import (Supervisor,
                                                       SIMSCRIPT_STANDARD_NAME)
from pytreenet.core.addition.addition import AdditionMethod
from pytreenet.special_ttn.special_states import TTNStructure

from sim_script import AdditionComparisonParams

def generate_parameter_set() -> list[AdditionComparisonParams]:
    """
    Generates a set of parameters for addition comparison experiments.
    
    Returns:
        list[AdditionComparisonParams]: A list of parameters for the simulations.
    """
    low = -0.5
    high = 1.0
    
    # Define structures and their corresponding system sizes and bond dimension ranges
    structure_configs = [
        (TTNStructure.MPS, 100, 5, 100, 5),      # structure, sys_size, min_bd, max_bd, step_bd
        (TTNStructure.FTPS, 10, 5, 50, 5),
        (TTNStructure.BINARY, 6, 5, 50, 5),
        (TTNStructure.TSTAR, 33, 5, 50, 5)
    ]
    
    # Define addition methods to compare
    methods = [
        AdditionMethod.DIRECT_TRUNCATE,
        AdditionMethod.DENSITY_MATRIX,
        AdditionMethod.HALF_DENSITY_MATRIX,
        AdditionMethod.SRC
    ]
    
    # Number of times to add the TTNS to itself
    num_additions_list = [2, 5, 10
                          ]
    
    # Multiple seeds for statistical confidence
    seeds = [1234, 4321,
             5678, 8765, 9012
             ]
    
    # Physical dimension
    phys_dim = 2
    
    param_set = []
    
    for (structure, sys_size, min_bd, max_bd, step_bd), method, num_additions, seed in product(
        structure_configs, methods, num_additions_list, seeds
    ):
        params = AdditionComparisonParams(
            structure=structure,
            sys_size=sys_size,
            phys_dim=phys_dim,
            init_bond_dim=max_bd,
            max_bond_dim=max_bd,
            min_bond_dim=min_bd,
            step_bond_dim=step_bd,
            addition_method=method,
            num_additions=num_additions,
            seed=seed,
            distr_low=low,
            distr_high=high
        )
        param_set.append(params)
    
    return param_set

def main():
    """
    Main function to set up and run the supervisor for addition comparison experiments.
    """
    param_set = generate_parameter_set()
    
    print(f"Generated {len(param_set)} parameter combinations")
    
    # The simulation script lies in the same directory as this script
    sim_script_path = os.path.join(os.path.dirname(__file__),
                                   SIMSCRIPT_STANDARD_NAME)
    
    supervisor = Supervisor.from_commandline(param_set,
                                             sim_script_path)
    supervisor.timeout = 2 * 60 * 60  # 2 hours per simulation
    supervisor.run_simulations()

if __name__ == "__main__":
    main()
