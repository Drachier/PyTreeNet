"""
This script is used to supervise the truncation comparison experiments.
"""
import os
from itertools import product

from pytreenet.util.experiment_util.supervisor import (Supervisor,
                                                       SIMSCRIPT_STANDARD_NAME)

from sim_script import (TwoDimParams,
                        ModelKind,
                        TTNOFinder,
                        TTNStructure)

def generate_parameter_set() -> list[TwoDimParams]:
    """
    Generates a set of parameters for time evolution experiments.
    
    Returns:
        list[TruncationParams]: A list of tuples containing simulation and time
        evolution parameters.
    """
    param_set: list[TwoDimParams] = []
    sys_sizes = range(2, 30)
    model_kinds = [ModelKind.ISING, ModelKind.XXZ]
    ttn_structures = [TTNStructure.MPS,
                      TTNStructure.BINARY,
                      TTNStructure.FTPS]
    finders = [TTNOFinder.SGE,
               TTNOFinder.SGE_PURE,
               TTNOFinder.BIPARTITE]
    for sys_size, model_kind, ttn_structure, finder in product(sys_sizes,
                                                               model_kinds,
                                                               ttn_structures,
                                                               finders):
        params = TwoDimParams()
        params.sys_size = sys_size
        params.model = model_kind
        params.ttn_structure = ttn_structure
        params.finder = finder
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
    supervisor.timeout = 5 * 60 * 60  # 5 hours
    supervisor.run_simulations()

if __name__ == "__main__":
    main()
