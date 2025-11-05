"""
This script is used to supervise the truncation comparison experiments.
"""
import os
from itertools import product

from pytreenet.util.experiment_util.supervisor import (Supervisor,
                                                       SIMSCRIPT_STANDARD_NAME)
from pytreenet.ttns.ttns_ttno.application import ApplicationMethod
from pytreenet.special_ttn.special_states import TTNStructure

from sim_script import CircuitSimParams

def generate_parameter_set() -> list[CircuitSimParams]:
    """
    Generates a set of parameters for time evolution experiments.
    
    Returns:
        list[TruncationParams]: A list of tuples containing simulation and time
        evolution parameters.
    """
    structures = (TTNStructure.TSTAR,
                  TTNStructure.MPS,
                  TTNStructure.BINARY)
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
    min_bd = 5
    max_bd = 50
    param_set = []
    for structure, method, seed in product(structures, methods, seeds):
        params = CircuitSimParams(
            ttn_structure=structure,
            appl_method=method,
            min_bond_dim=min_bd,
            max_bond_dim=max_bd,
            bond_dim_step=5,
            seed=seed,
            num_circuit_repeats=1
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
