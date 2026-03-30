"""
This script is used to supervise the integrator comparison simulations.
"""
import os
from itertools import product

from pytreenet.util.experiment_util.supervisor import (Supervisor,
                                                       SIMSCRIPT_STANDARD_NAME)
from pytreenet.special_ttn.special_states import TTNStructure

from sim_script import SimParams2TDVP, Integrator

def generate_parameter_set() -> list[SimParams2TDVP]:
    """
    Generates a set of parameters for integrator comparison experiments.

    Returns:
        list[SimParams2TDVP]: A list of parameters for the simulations.
    """

    # Define structures and their corresponding system sizes and bond dimension ranges
    structure_configs = [
        (TTNStructure.MPS, 5, 5, 100, 5),      # structure, sys_size, min_bd, max_bd, step_bd
        #(TTNStructure.FTPS, 4, 5, 50, 5),
        #(TTNStructure.BINARY, 3, 5, 50, 5),
        #(TTNStructure.TSTAR, 4, 5, 50, 5)
    ]

    # Tolerance values to test
    rel_tols = [10**(-i) for i in range(9, 11)]
    total_tols = [10**(-i) for i in range(9, 11)]
    integrators = [Integrator.TWO_SITE_TDVP, Integrator.BUG]

    ext_magn = 0.5
    
    param_set = []
    # Parameters for varying bond dimension
    for integrator, (structure, sys_size, min_bd, max_bd, step_bd) in product(integrators,
                                                                              structure_configs):
        for bond_dim in range(min_bd, max_bd + 1, step_bd):
            param_set.append(SimParams2TDVP(
                system_size=sys_size,
                structure=structure,
                max_bond_dim=bond_dim,
                rel_tol=1e-10,
                total_tol=1e-10,
                integrator=integrator,
                ext_magn=ext_magn
            ))

    # Parameters for varying relative tolerance
    for integrator, (structure, sys_size, _, max_bd, _) in product(integrators,
                                                                    structure_configs):
        for rel_tol in rel_tols:
            param_set.append(SimParams2TDVP(
                system_size=sys_size,
                structure=structure,
                max_bond_dim=max_bd,
                rel_tol=rel_tol,
                total_tol=1e-10,
                integrator=integrator,
                ext_magn=ext_magn
            ))
    
    # Parameters for varying total tolerance
    for integrator, (structure, sys_size, _, max_bd, _) in product(integrators,
                                                                    structure_configs):
        for total_tol in total_tols:
            param_set.append(SimParams2TDVP(
                system_size=sys_size,
                structure=structure,
                max_bond_dim=max_bd,
                rel_tol=1e-10,
                total_tol=total_tol,
                integrator=integrator,
                ext_magn=ext_magn
            ))
    return param_set

def main():
    """
    Main function to set up and run the supervisor for integrator experiments.
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
