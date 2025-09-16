"""
This script is used to superwise the time evolution experiments in PyTreeNet.
"""
import os
from itertools import product

from pytreenet.util.experiment_util.supervisor import (Supervisor,
                                                       SIMSCRIPT_STANDARD_NAME)
from pytreenet.operators.models.topology import Topology
from pytreenet.special_ttn.special_states import TTNStructure
from pytreenet.time_evolution.time_evo_enum import TimeEvoAlg
from pytreenet.time_evolution.time_evolution import TimeEvoMode

from sim_script import TotalParameters

def generate_parameter_set() -> list[TotalParameters]:
    """
    Generates a set of parameters for time evolution experiments.
    
    Returns:
        list[TotalParameters]: A list of tuples containing simulation and time evolution parameters.
    """
    time_step = 0.01
    ext_magn = 0.1
    final_time = 1.0
    topology = Topology.CHAIN
    ttn_structure = [TTNStructure.MPS, TTNStructure.BINARY]
    num_sites = [5,10,15,20]
    interaction_length = [2]

    time_evo_modes = [TimeEvoMode.CHEBYSHEV,
                      TimeEvoMode.EXPM,
                      TimeEvoMode.RK45,
                      TimeEvoMode.RK23,
                      TimeEvoMode.BDF,
                      TimeEvoMode.DOP853]
    evo_alg = TimeEvoAlg.BUG
    maximum_bond_dim = [1,2,5,10,20,25,50,75,100]
    rel_svalue = 1e-15
    abs_svalue = 1e-15

    atol=1e-8
    rtol=1e-8

    # Generate a list of simulation parameters with different configurations
    iterator = product(num_sites,
                       interaction_length,
                       time_evo_modes,
                       maximum_bond_dim,
                       ttn_structure)
    sim_params_list = []
    for ns, il, time_evo_mode, max_bd, ttn_str in iterator:
        sim_params = TotalParameters(ttns_structure=ttn_str,
                                     topology=topology,
                                     system_size=ns,
                                     ext_magn=ext_magn,
                                     interaction_range=il,
                                     time_evo_method=time_evo_mode,
                                     time_evo_algorithm=evo_alg,
                                     time_step_size=time_step,
                                     final_time=final_time,
                                     max_bond_dim=max_bd,
                                     rel_svalue=rel_svalue,
                                     abs_svalue=abs_svalue,
                                     atol=atol,
                                     rtol=rtol)
        sim_params_list.append(sim_params)
    return sim_params_list

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
