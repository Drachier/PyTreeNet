"""
This script is used to supervise the truncation comparison experiments.
"""
import os
from itertools import product

from pytreenet.util.experiment_util.supervisor import (Supervisor,
                                                       SIMSCRIPT_STANDARD_NAME)

from sim_script import (AllToAllSimParams,
                        DimensionalityType,
                        TTNOFinder,
                        TTNStructure)

def generate_parameter_set() -> list[AllToAllSimParams]:
    """
    Generates a set of parameters for time evolution experiments.
    
    Returns:
        list[TruncationParams]: A list of tuples containing simulation and time
        evolution parameters.
    """
    param_set: list[AllToAllSimParams] = []
    methods = [TTNOFinder.SGE,
               TTNOFinder.SGE_PURE,
               TTNOFinder.BIPARTITE]
    # Params for system size depencence sim 1D
    sys_sizes = range(5, 200, 10)
    ttn_structures = [TTNStructure.MPS,
                      TTNStructure.BINARY]
    num_operators = (1,3,6)
    for sys_size, ttn_structure, method, num_op in product(sys_sizes,
                                                            ttn_structures,
                                                            methods,
                                                            num_operators):
        params = AllToAllSimParams()
        params.system_size = sys_size
        params.dimensionality = DimensionalityType.ONE_D
        params.ttn_structure = ttn_structure
        params.method = method
        params.num_operators = num_op
        param_set.append(params)
    # Params for system size depencence sim 2D
    sys_sizes = range(2, 20)
    ttn_structures = [TTNStructure.MPS,
                      TTNStructure.BINARY,
                      TTNStructure.FTPS]
    num_operators = (1,3,6)
    for sys_size, ttn_structure, method, num_op in product(sys_sizes,
                                                            ttn_structures,
                                                            methods,
                                                            num_operators):
        params = AllToAllSimParams()
        params.system_size = sys_size
        params.dimensionality = DimensionalityType.TWO_D
        params.ttn_structure = ttn_structure
        params.method = method
        params.num_operators = num_op
        param_set.append(params)
    # Operator Number Dependence 1D
    sys_sizes = (20, 50, 100)
    ttn_structures = [TTNStructure.MPS,
                        TTNStructure.BINARY]
    num_operators = range(1, 21, 2)
    for sys_size, ttn_structure, method, num_op in product(sys_sizes,
                                                            ttn_structures,
                                                            methods,
                                                            num_operators):
        params = AllToAllSimParams()
        params.system_size = sys_size
        params.dimensionality = DimensionalityType.ONE_D
        params.ttn_structure = ttn_structure
        params.method = method
        params.num_operators = num_op
        param_set.append(params)
    # Operator Number Dependence 2D
    sys_sizes = (5, 10)
    ttn_structures = [TTNStructure.MPS,
                      TTNStructure.BINARY,
                      TTNStructure.FTPS]
    num_operators = range(1, 21, 2)
    for sys_size, ttn_structure, method, num_op in product(sys_sizes,
                                                            ttn_structures,
                                                            methods,
                                                            num_operators):
        params = AllToAllSimParams()
        params.system_size = sys_size
        params.dimensionality = DimensionalityType.TWO_D
        params.ttn_structure = ttn_structure
        params.method = method
        params.num_operators = num_op
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
