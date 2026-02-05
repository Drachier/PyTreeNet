"""
This script is used to supervise the addition experiment
"""
import os
from itertools import product

from pytreenet.util.experiment_util.supervisor import (Supervisor,
                                                       SIMSCRIPT_STANDARD_NAME)
from pytreenet.core.addition.addition import AdditionMethod
from pytreenet.special_ttn.special_states import TTNStructure

from sim_script import AdditionParams

def generate_parameter_set() -> list[AdditionParams]:
    """
    Generates a set of parameters for time evolution experiments.
    
    Returns:
        list[AdditionParams]: A list of tuples containing simulation and time
        evolution parameters.
    """
    param_set = []
    structures = [TTNStructure.MPS,
                  TTNStructure.BINARY,
                  TTNStructure.FTPS,
                  TTNStructure.TSTAR]
    sys_sizes = [100, 6, 10, 35]
    num_adds = [2, 5, 10]
    addition_methods = [AdditionMethod.DIRECT_TRUNCATE,
                        AdditionMethod.DENSITY_MATRIX]
    bond_dims = [10, 20, 50, 75, 100]
    seeds = [12334, 4321, 98765]
    for struct_size, num_add, add_method, bond_dim, seed in product(zip(structures, sys_sizes),
                                                                    num_adds,
                                                                    addition_methods,
                                                                    bond_dims,
                                                                    seeds):
        structure, size = struct_size
        params = AdditionParams(structure=structure,
                                sys_size=size,
                                phys_dim=4,
                                bond_dim=bond_dim,
                                num_additions=num_add,
                                seed=seed,
                                addition_method=add_method,
                                min_target_bond_dim=5,
                                max_target_bond_dim=bond_dim,
                                step_target_bond_dim=5)
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
