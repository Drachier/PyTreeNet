"""
The simulation script comparing integrator performance.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import os
import time

from h5py import File

from pytreenet.operators.models.two_site_model import (IsingModel,
                                                       IsingParameters)
from pytreenet.special_ttn.special_states import (generate_zero_state,
                                                  TTNStructure,
                                                  STANDARD_NODE_PREFIX,
                                                  Topology)
from pytreenet.time_evolution.tdvp_algorithms.secondordertwosite import (SecondOrderTwoSiteTDVP,
                                                                         TwoSiteTDVPConfig)
from pytreenet.time_evolution.tdvp_algorithms.firstordertwosite import FirstOrderTwoSiteTDVP
from pytreenet.time_evolution.bug import BUG, BUGConfig
from pytreenet.ttno.ttno_class import TTNO
from pytreenet.time_evolution.time_evolution import TimeEvoMode
from pytreenet.operators.models.eval_ops import (local_magnetisation_from_topology)
from pytreenet.time_evolution.results import Results
from pytreenet.time_evolution.exact_time_evolution import (ExactTimeEvolution)
from pytreenet.util.experiment_util.script_util import script_main
from pytreenet.operators.exact_operators import exact_local_magnetisation
from pytreenet.util.tensor_splitting import SVDParameters


class Integrator(Enum):
    """
    Supported integrators for the simulation.
    """
    FO_TWO_SITE_TDVP = "fo_2tdvp"
    TWO_SITE_TDVP = "2tdvp"
    BUG = "bug"

@dataclass
class SimParams2TDVP(IsingParameters, SVDParameters):
    """
    Parameters for two-site TDVP simulations.
    """
    rtol: float = 1e-10
    atol: float = 1e-10
    time_step_size: float = 0.01
    system_size: int = 5
    structure: TTNStructure = TTNStructure.MPS
    integrator: Integrator = Integrator.TWO_SITE_TDVP

STRUCT_TO_TOP_MAP = {TTNStructure.MPS: Topology.CHAIN,
                     TTNStructure.BINARY: Topology.CHAIN,
                     TTNStructure.TSTAR: Topology.TTOPOLOGY,
                     TTNStructure.FTPS: Topology.SQUARE}

def run_simulation(params: SimParams2TDVP) -> tuple[Results, float]:
    """
    Runs a simulation of the Ising Model with the given parameters.
    """
    top = STRUCT_TO_TOP_MAP[params.structure]
    init_state = generate_zero_state(params.system_size,
                                     params.structure,
                                     node_prefix=STANDARD_NODE_PREFIX,
                                     bond_dim=2,
                                     topology=top
                                     )
    model = IsingModel(params.interaction_range,
                       factor=params.factor,
                       ext_magn=params.ext_magn)
    ham = model.generate_by_topology(top,
                                     params.system_size,
                                     site_id_prefix=STANDARD_NODE_PREFIX)
    ttno = TTNO.from_hamiltonian(ham, init_state)
    ops = local_magnetisation_from_topology(top,
                                            params.system_size,
                                            site_prefix=STANDARD_NODE_PREFIX)
    if params.integrator is Integrator.TWO_SITE_TDVP:
        cnfg = TwoSiteTDVPConfig(max_bond_dim=params.max_bond_dim,
                                 rel_tol=params.rel_tol,
                                 total_tol=params.total_tol,
                                 time_evo_mode=TimeEvoMode.RK45)
        time_evo = SecondOrderTwoSiteTDVP(init_state,
                                          ttno,
                                          params.time_step_size,
                                          1,
                                          ops,
                                          config=cnfg,
                                          solver_options={"rtol": params.rtol,
                                                          "atol": params.atol})
    elif params.integrator is Integrator.FO_TWO_SITE_TDVP:
        cnfg = TwoSiteTDVPConfig(max_bond_dim=params.max_bond_dim,
                                 rel_tol=params.rel_tol,
                                 total_tol=params.total_tol,
                                 time_evo_mode=TimeEvoMode.RK45)
        time_evo = FirstOrderTwoSiteTDVP(init_state,
                                          ttno,
                                          params.time_step_size,
                                          1,
                                          ops,
                                          config=cnfg,
                                          solver_options={"rtol": params.rtol,
                                                          "atol": params.atol})
    elif params.integrator is Integrator.BUG:
        cnfg = BUGConfig(max_bond_dim=params.max_bond_dim,
                         rel_tol=params.rel_tol,
                         total_tol=params.total_tol,
                         time_evo_mode=TimeEvoMode.RK45)
        time_evo = BUG(init_state,
                       ttno,
                       params.time_step_size,
                       1,
                       ops,
                       config=cnfg,
                       solver_options={"rtol": params.rtol,
                                       "atol": params.atol})
    else:
        raise ValueError(f"Unknown integrator {params.integrator}.")
    start = time.time()
    time_evo.run(pgbar=False)
    end = time.time()
    return time_evo.results, end - start

def run_reference(params: SimParams2TDVP) -> Results:
    """
    Runs the reference simulation of the Ising Model with the given parameters.
    """
    top = STRUCT_TO_TOP_MAP[params.structure]
    init_state = generate_zero_state(params.system_size,
                                     params.structure,
                                     node_prefix=STANDARD_NODE_PREFIX,
                                     bond_dim=2,
                                     topology=top
                                     )
    model = IsingModel(params.interaction_range,
                       factor=params.factor,
                       ext_magn=params.ext_magn)
    ham = model.generate_by_topology(top,
                                     params.system_size,
                                     site_id_prefix=STANDARD_NODE_PREFIX)
    ttno = TTNO.from_hamiltonian(ham, init_state)
    init_vec, contraction_order = init_state.completely_contract_tree()
    ham_mat, _ = ttno.as_matrix(order=contraction_order)
    node_ids = [f"{STANDARD_NODE_PREFIX}_{i}"
                for i in range(init_vec.ndim)
                if init_vec.shape[i] > 1]
    init_vec = init_vec.reshape(-1)
    ops = exact_local_magnetisation(node_ids)
    exact_evo = ExactTimeEvolution(init_vec,
                                   ham_mat,
                                   params.time_step_size,
                                   1,
                                   ops)
    exact_evo.run(pgbar=False)
    return exact_evo.results

def compare_results(res: Results, ref: Results) -> Results:
    """
    Compares the results of the TDVP simulation with the reference simulation.
    """
    ref_magn = ref.average_results(STANDARD_NODE_PREFIX + "_", realise=True)
    res_magn = res.average_results(STANDARD_NODE_PREFIX, realise=True)
    err_res = Results()
    ops = {"ref_magn": float,
           "res_magn": float,
           "error": float}
    err_res.initialize(ops,
                       res.results_length() - 1)
    for i, t in enumerate(res.times()):
        err_res.set_time(i, t)
        err_res.set_element("ref_magn", i, ref_magn[i])
        err_res.set_element("res_magn", i, res_magn[i])
        err_res.set_element("error", i, abs(ref_magn[i] - res_magn[i]))
    return err_res

def main(params: SimParams2TDVP,
         save_path: str) -> None:
    """
    The main function for the simulation script.
    """
    res, res_time = run_simulation(params)
    ref = run_reference(params)
    err_res = compare_results(res, ref)
    filename = params.get_hash() + ".h5"
    filepath = os.path.join(save_path, filename)
    with File(filepath, "w") as f:
        err_res.save_to_h5(f)
        params.save_to_h5(f)
        f.attrs["simulation_time"] = res_time

if __name__ == "__main__":
    script_main(main,
                SimParams2TDVP)
