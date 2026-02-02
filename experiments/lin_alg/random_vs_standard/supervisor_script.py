"""
The supervisor script for running the random vs standard SVD experiment.
"""

import os
from copy import copy
from itertools import product

from pytreenet.util.experiment_util.supervisor import (Supervisor,
                                                       SIMSCRIPT_STANDARD_NAME)

from sim_script import (RandomVsStandardParams,
                        SVDType)

def generate_parameter_set() -> list[RandomVsStandardParams]:
    """
    Generates a set of parameters for time evolution experiments.
    
    Returns:
        list[RandomVsStandardParams]: A list of tuples containing simulation and time
        evolution parameters.
    """
    low = -0.5
    high = 1.0
    svd_types = [SVDType.STANDARD,
                 SVDType.RANDOMIZED]
    dimensions = [1e2,1e3,1e4,1e5]
    seeds = [669564823, 556548421]
    parameter_set = []
    for dim, svd_type, seed in product(dimensions, svd_types, seeds):
        params = RandomVsStandardParams()
        params.dimension = int(dim)
        params.low = low
        params.high = high
        params.svd_type = svd_type
        params.seed = seed
        parameter_set.append(params)
        params.rank_max = params.dimension
        params.rank_step = params.rank_max // 100
        params.rank_min = params.rank_step
    return parameter_set

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
