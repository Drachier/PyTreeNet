import math
from typing import Optional
from pytreenet.time_evolution.time_evo_enum import TimeEvoAlg
from pytreenet.time_evolution.time_evolution import TimeEvoMode, TimeEvoMethod
from pytreenet.util.tensor_splitting import SVDParameters
from experiments.time_evolution.open_system_bug.sim_script import (SimulationParameters,
                                                                    TimeEvolutionParameters,
                                                                    TTNStructure,
                                                                    TimeStepLevel)


def generate_timestep_evaluation_grid(time_step: float,
                                     evaluation_time: int) -> list[tuple[TimeStepLevel,
                                                                         float,
                                                                         int]]:
    """Generate three timestep/evaluation pairs labeled by TimeStepLevel."""
    # Define the three fineness levels
    levels = [TimeStepLevel.COARSE, TimeStepLevel.MEDIUM, TimeStepLevel.FINE]
    # Start with the initial grid
    ts = time_step
    et = evaluation_time
    grid = [(ts, et)]
    # Refine twice by halving time step and doubling evaluation window
    for _ in range(2):
        ts /= 2
        et *= 2
        grid.append((ts, et))
    # Combine levels with grid entries
    combinations = [(lvl, t, e) for lvl, (t, e) in zip(levels, grid)]
    return combinations

def generate_parameter(ttn_structure: TTNStructure,
                       depth: Optional[int] = None,
                       svd_prams: SVDParameters = SVDParameters()) -> list:
    """
    Generate complete parameter combinations for a specific (structure, depth) pair.

    Args:
        ttn_structure: Specific TTN structure (MPS, BINARY, or SYMMETRIC)
        depth: Specific depth value
        svd_prams: SVD parameters for tensor decomposition 
        (includes truncation_level and optional label)

    Returns:
        list: Complete parameter combinations ready for simulation
    """
    # length: Number of spins in the chain
    num_site = 4
    # ext_magn: External magnetic field strength in z-direction
    ext_magn = 0.5
    # coupling: Nearest-neighbor coupling strength for XX interactions
    coupling = 1.0
    # relaxation_rate: Local amplitude damping (relaxation) rate
    relaxation_rate = 0.1
    # dephasing_rate: Local dephasing rate
    dephasing_rate = 0.1
    # Compute time-step evaluation grid
    final_time = 2
    time_step = 0.1
    evaluation_time = 1

    timestep_evaluation = generate_timestep_evaluation_grid(time_step,
                                                           evaluation_time)

    if depth is None and ttn_structure != TTNStructure.MPS:
        depth = max(1, math.ceil(math.log2(num_site)))
        print(f"Depth not specified, using maximum depth: {depth}")

    init_bond_dim = 4

    time_evo_modes = [TimeEvoMode(TimeEvoMethod.RK45, {'atol': 1e-6, 'rtol': 1e-6}),
                      TimeEvoMode(TimeEvoMethod.FASTEST)]
                      #TimeEvoMode(TimeEvoMethod.RK45, {'atol': 1e-6, 'rtol': 1e-6})]
    evo_algs = [TimeEvoAlg.SRBUG,
                TimeEvoAlg.PRBUG,]
                #TimeEvoAlg.FPBUG,
                #TimeEvoAlg.SPBUG]

    # Extract local parameters
    maximum_bond_dim = svd_prams.max_bond_dim
    rel_svalue = svd_prams.rel_tol
    abs_svalue = svd_prams.total_tol
    renorm = svd_prams.renorm
    sum_trunc = svd_prams.sum_trunc
    sum_renorm = svd_prams.sum_renorm
    # Use truncation_level from SVD parameters
    truncation_level = svd_prams.truncation_level

    # Generate complete parameter combinations
    complete_params_list = []

    for time_evo_mode in time_evo_modes:
        for evo_alg in evo_algs:
            for time_step_level, ts, ev_time in timestep_evaluation:
                sim_params = SimulationParameters(ttn_structure,
                                                 num_site,
                                                 coupling,
                                                 ext_magn,
                                                 relaxation_rate,
                                                 dephasing_rate,
                                                 init_bond_dim,
                                                 depth)

                time_evo_params = TimeEvolutionParameters(time_evo_mode,
                                                        evo_alg,
                                                        ts,
                                                        ev_time,
                                                        final_time,
                                                        max_bond_dim=maximum_bond_dim,
                                                        rel_svalue=rel_svalue,
                                                        abs_svalue=abs_svalue,
                                                        renorm=renorm,
                                                        sum_trunc=sum_trunc,
                                                        sum_renorm=sum_renorm,
                                                        truncation_level=truncation_level,
                                                        time_step_level=time_step_level)

                complete_params_list.append((sim_params, time_evo_params))

    return complete_params_list
