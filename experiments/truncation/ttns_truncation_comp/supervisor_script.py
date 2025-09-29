"""
This script is used to supervise the truncation comparison experiments.
"""
import os
from copy import copy

from pytreenet.util.experiment_util.supervisor import (Supervisor,
                                                       SIMSCRIPT_STANDARD_NAME)
from pytreenet.core.truncation import TruncationMethod
from pytreenet.special_ttn.special_states import TTNStructure

from sim_script import TruncationParams

def generate_parameter_set() -> list[TruncationParams]:
    """
    Generates a set of parameters for time evolution experiments.
    
    Returns:
        list[TruncationParams]: A list of tuples containing simulation and time
        evolution parameters.
    """
    low = -0.5
    high = 1.0
    param_set = []
    for structure in (TTNStructure.MPS, TTNStructure.TSTAR, TTNStructure.BINARY, TTNStructure.FTPS):
        if structure in {TTNStructure.MPS, TTNStructure.BINARY}:
            sys_size = 3
        else:
            sys_size = 2
        for method in (TruncationMethod.RECURSIVE, TruncationMethod.SVD, TruncationMethod.VARIATIONAL):
            params = TruncationParams(
                structure=structure,
                sys_size=sys_size,
                phys_dim=2,
                bond_dim=4 if structure != TTNStructure.MPS else 2,
                trunc_method=method,
                random_trunc=False,
                max_target_bond_dim=4,
                seed=1234,
                distr_low=low,
                distr_high=high
            )
            param_set.append(params)
            if method in {TruncationMethod.SVD, TruncationMethod.RECURSIVE}:
                copy_params = copy(params)
                copy_params.random_trunc = True
                param_set.append(copy_params)
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
