"""
This script is used to supervise the truncation comparison experiments.
"""
import os
from copy import copy
from itertools import product

from pytreenet.util.experiment_util.supervisor import (Supervisor,
                                                       SIMSCRIPT_STANDARD_NAME)
from pytreenet.ttns.ttns_ttno.application import ApplicationMethod
from pytreenet.special_ttn.special_states import TTNStructure

from sim_script import ApplicationParams

def generate_parameter_set() -> list[ApplicationParams]:
    """
    Generates a set of parameters for time evolution experiments.
    
    Returns:
        list[TruncationParams]: A list of tuples containing simulation and time
        evolution parameters.
    """
    low = -0.5
    high = 1.0
    structures = (TTNStructure.TSTAR, TTNStructure.MPS,
                  TTNStructure.BINARY, TTNStructure.FTPS)
    methods = (ApplicationMethod.DENSITY_MATRIX,
               ApplicationMethod.HALF_DENSITY_MATRIX,
               ApplicationMethod.SRC,
               ApplicationMethod.ZIPUP,
               ApplicationMethod.VARIATIONAL,
               ApplicationMethod.ZIPUP_VARIATIONAL,
               ApplicationMethod.HALF_DENSITY_MATRIX_VARIATIONAL,
               ApplicationMethod.DIRECT_TRUNCATE
               )
    seeds = (1234, 4321, 32974, 238934, 239401)
    bond_dims = (40, 50, 60)
    param_set = []
    for structure, method, bond_dim, seed in product(structures, methods, bond_dims, seeds):
        if structure is TTNStructure.MPS:
            sys_size = 50
        elif structure is TTNStructure.BINARY:
            sys_size = 7
        elif structure is TTNStructure.FTPS:
            sys_size = 8
        elif structure is TTNStructure.TSTAR:
            sys_size = 20
        else:
            raise ValueError(f"Unknown structure: {structure}")
        params = ApplicationParams(
            structure=structure,
            sys_size=sys_size,
            phys_dim=5,
            bond_dim=bond_dim,
            appl_method=method,
            max_target_bond_dim=bond_dim + 1,
            min_target_bond_dim=2,
            step_target_bond_dim=2,
            seed=seed,
            distr_low=low,
            distr_high=high
        )
        param_set.append(params)
    return param_set

def main():
    """
    Main function to set up and run the supervisor for time evolution experiments.
    """
    param_set = generate_parameter_set()
    # The simulation script lies in the same directory as this script
    sim_script_path = os.path.join(os.path.dirname(__file__),
                                   SIMSCRIPT_STANDARD_NAME)
    supervisor = Supervisor.from_commandline(param_set,
                                             sim_script_path)
    supervisor.timeout = 10 * 60 * 60  # 10 hours
    supervisor.run_simulations()

if __name__ == "__main__":
    main()
