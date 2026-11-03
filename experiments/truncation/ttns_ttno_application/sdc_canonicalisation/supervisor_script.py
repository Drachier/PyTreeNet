"""
This script is used to supervise the SDC canonicalisation comparison
experiments for both the uniform and the Gaussian random elements.
"""
import os
from itertools import product

from pytreenet.util.experiment_util.supervisor import (Supervisor,
                                                       SIMSCRIPT_STANDARD_NAME)
from pytreenet.ttns.ttns_ttno.application import ApplicationMethod
from pytreenet.special_ttn.special_states import TTNStructure

from canonicalisation_common import (ElementDistribution,
                                     SDCCanonicalisationParams)

def generate_parameter_set() -> list[SDCCanonicalisationParams]:
    """
    Generates a set of parameters for the SDC canonicalisation comparison
    experiments.

    Returns:
        list[SDCCanonicalisationParams]: A list of simulation parameters.
    """
    phys_dim = 3
    structure = TTNStructure.MPS
    sys_size = 50
    distributions = (ElementDistribution.UNIFORM,
                     ElementDistribution.GAUSSIAN)
    seeds = (1234, 4321, 32974, 238934, 239401)
    bond_dims = (10, 20, 40, 50)
    param_set = []
    for distribution, bond_dim, seed in product(distributions,
                                                bond_dims,
                                                seeds):
        params = SDCCanonicalisationParams(
            structure=structure,
            sys_size=sys_size,
            phys_dim=phys_dim,
            bond_dim=bond_dim,
            appl_method=ApplicationMethod.HALF_DENSITY_MATRIX,
            max_target_bond_dim=bond_dim + 1,
            min_target_bond_dim=2,
            step_target_bond_dim=2,
            seed=seed,
            distribution=distribution
        )
        param_set.append(params)
    return param_set

def main():
    """
    Main function to set up and run the supervisor for the SDC
    canonicalisation comparison experiments.
    """
    param_set = generate_parameter_set()
    # The simulation script lies in the same directory as this script
    sim_script_path = os.path.join(os.path.dirname(__file__),
                                   SIMSCRIPT_STANDARD_NAME)
    supervisor = Supervisor.from_commandline(param_set,
                                             sim_script_path)
    supervisor.timeout = 5 * 60 * 60  # 5 hours
    supervisor.run_simulations()

if __name__ == "__main__":
    main()
