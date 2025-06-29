"""
This module contains the current parameter sets for time evolution experiments
in PyTreeNet.
"""
from itertools import product

from experiments.time_evolution.closed_system.sim_script import (SimulationParameters,
                        TimeEvolutionParameters,
                        Topology,
                        TTNStructure,
                        TimeEvoMode,
                        TimeEvoAlg)

def generate_parameter_set():
    """
    Generates a set of parameters for time evolution experiments.
    
    Returns:
        list: A list of tuples containing simulation and time evolution parameters.
    """
    time_step = 0.01
    ext_magn = 0.1
    final_time = 1.0
    topology = Topology.CHAIN
    ttn_structure = [TTNStructure.MPS, TTNStructure.BINARY]
    num_sites = [5,10,15,20]
    interaction_length = [2,3,4]

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
        sim_params = SimulationParameters(ttn_str,
                                          topology,
                                          ns,
                                          il,
                                          ext_magn)
        time_evo_params = TimeEvolutionParameters(time_evo_mode,
                                                  evo_alg,
                                                  time_step,
                                                  final_time,
                                                  max_bond_dim=max_bd,
                                                  rel_svalue=rel_svalue,
                                                  abs_svalue=abs_svalue,
                                                  atol=atol,
                                                  rtol=rtol)
        sim_params_list.append((sim_params, time_evo_params))
    return sim_params_list
