"""
This script is used to supervise the 1TDVP simulations.
"""

"""
This script is used to supervise the random addition comparison experiments.
"""
import os
from itertools import product

from pytreenet.util.experiment_util.supervisor import (Supervisor,
                                                       SIMSCRIPT_STANDARD_NAME)
from pytreenet.special_ttn.special_states import TTNStructure

from sim_script import SimParams1TDVP, Order

def generate_parameter_set() -> list[SimParams1TDVP]:
    """
    Generates a set of parameters for random addition comparison experiments.

    Returns:
        list[SimParams1TDVP]: A list of parameters for the simulations.
    """

    # Define structures and their corresponding system sizes and bond dimension ranges
    structure_configs = [
        (TTNStructure.MPS, 14, 5, 100, 5),      # structure, sys_size, min_bd, max_bd, step_bd
        (TTNStructure.FTPS, 4, 5, 50, 5),
        (TTNStructure.BINARY, 3, 5, 50, 5),
        (TTNStructure.TSTAR, 4, 5, 50, 5)
    ]

    # The two orders
    orders = [Order.FIRST, Order.SECOND]

    param_set = []

    for struct_congif, order in product(structure_configs, orders):
        for bond_dim in range(struct_congif[2], struct_congif[3] + 1, struct_congif[4]):
            params = SimParams1TDVP(
                structure=struct_congif[0],
                system_size=struct_congif[1],
                ext_magn=0.5,
                time_step_size=0.1,
                bond_dim=bond_dim,
                order=order
            )
            param_set.append(params)

    # Parameters for the time step dependence
    time_steps = [10**(-1*i) for i in range(1, 10)]  # 0.1, 0.01, 0.001, 0.0001
    for time_step in time_steps:
        for struct_congif, order in product(structure_configs, orders):
            params = SimParams1TDVP(
                structure=struct_congif[0],
                system_size=struct_congif[1],
                ext_magn=0.5,
                time_step_size=time_step,
                bond_dim=struct_congif[3],  # Max bond dimension for time step dependence
                order=order
            )
            param_set.append(params)        

    return param_set

def main():
    """
    Main function to set up and run the supervisor for random addition comparison experiments.
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
