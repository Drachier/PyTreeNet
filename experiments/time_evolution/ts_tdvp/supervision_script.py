"""
This script is used to supervise the 2TDVP simulations.
"""
import os
from itertools import product

from pytreenet.util.experiment_util.supervisor import (Supervisor,
                                                       SIMSCRIPT_STANDARD_NAME)
from pytreenet.special_ttn.special_states import TTNStructure

from sim_script import SimParams2TDVP

def generate_parameter_set() -> list[SimParams2TDVP]:
    """
    Generates a set of parameters for two-site TDVP experiments.

    Returns:
        list[SimParams2TDVP]: A list of parameters for the simulations.
    """

    # Define structures and their corresponding system sizes and bond dimension ranges
    structure_configs = [
        (TTNStructure.MPS, 14, 5, 100, 5),      # structure, sys_size, min_bd, max_bd, step_bd
        (TTNStructure.FTPS, 4, 5, 50, 5),
        (TTNStructure.BINARY, 3, 5, 50, 5),
        (TTNStructure.TSTAR, 4, 5, 50, 5)
    ]

    # Tolerance values to test
    rtol_values = [1e-6, 1e-8, 1e-10]
    atol_values = [1e-6, 1e-8, 1e-10]

    param_set = []

    # Parameters for bond dimension dependence with fixed tolerances
    for struct_congif in structure_configs:
        for bond_dim in range(struct_congif[2], struct_congif[3] + 1, struct_congif[4]):
            params = SimParams2TDVP(
                structure=struct_congif[0],
                system_size=struct_congif[1],
                ext_magn=0.5,
                time_step_size=0.1,
                bond_dim=bond_dim,
                rtol=1e-10,
                atol=1e-10
            )
            param_set.append(params)

    # Parameters for rtol dependence (with fixed atol and bond_dim)
    for struct_congif in structure_configs:
        for rtol in rtol_values:
            params = SimParams2TDVP(
                structure=struct_congif[0],
                system_size=struct_congif[1],
                ext_magn=0.5,
                time_step_size=0.1,
                bond_dim=struct_congif[3],  # Max bond dimension
                rtol=rtol,
                atol=1e-10
            )
            param_set.append(params)

    # Parameters for atol dependence (with fixed rtol and bond_dim)
    for struct_congif in structure_configs:
        for atol in atol_values:
            params = SimParams2TDVP(
                structure=struct_congif[0],
                system_size=struct_congif[1],
                ext_magn=0.5,
                time_step_size=0.1,
                bond_dim=struct_congif[3],  # Max bond dimension
                rtol=1e-10,
                atol=atol
            )
            param_set.append(params)

    # Parameters for combined tolerance dependence
    for struct_congif in structure_configs:
        for rtol, atol in product(rtol_values, atol_values):
            params = SimParams2TDVP(
                structure=struct_congif[0],
                system_size=struct_congif[1],
                ext_magn=0.5,
                time_step_size=0.1,
                bond_dim=struct_congif[3],  # Max bond dimension
                rtol=rtol,
                atol=atol
            )
            param_set.append(params)

    return param_set

def main():
    """
    Main function to set up and run the supervisor for two-site TDVP experiments.
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
