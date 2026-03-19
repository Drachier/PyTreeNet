"""
The simluation script comparing the two single site TDVP's performance.
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
from pytreenet.time_evolution.tdvp_algorithms.firstorderonesite import (FirstOrderOneSiteTDVP, OneSiteTDVP)
from pytreenet.time_evolution.tdvp_algorithms.secondorderonesite import (SecondOrderOneSiteTDVP)
from pytreenet.ttno.ttno_class import TTNO
from pytreenet.operators.models.eval_ops import (local_magnetisation_from_topology)
from pytreenet.time_evolution.results import Results
from pytreenet.time_evolution.exact_time_evolution import (ExactTimeEvolution)
from pytreenet.util.experiment_util.script_util import script_main
from pytreenet.operators.exact_operators import exact_local_magnetisation

class Order(Enum):
    """
    The TDVP order.
    """
    FIRST = 1
    SECOND = 2

    def get_class(self) -> type[OneSiteTDVP]:
        if self is self.FIRST:
            return FirstOrderOneSiteTDVP
        else:
            return SecondOrderOneSiteTDVP

@dataclass
class SimParams1TDVP(IsingParameters):
    """
    
    """
    bond_dim: int = 1
    time_step_size: float = 0.01
    system_size: int = 5
    order: Order = Order.FIRST
    structure: TTNStructure = TTNStructure.MPS

STRUCT_TO_TOP_MAP = {TTNStructure.MPS: Topology.CHAIN,
                     TTNStructure.BINARY: Topology.CHAIN,
                     TTNStructure.TSTAR: Topology.TTOPOLOGY,
                     TTNStructure.FTPS: Topology.SQUARE}

def run_simulation(params: SimParams1TDVP) -> tuple[Results, float]:
    """
    Runs a simulation of the Ising Model with the given parameters.
    """
    top = STRUCT_TO_TOP_MAP[params.structure]
    init_state = generate_zero_state(params.system_size,
                                     params.structure,
                                     node_prefix=STANDARD_NODE_PREFIX,
                                     bond_dim=params.bond_dim,
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
    cls = params.order.get_class()
    time_evo = cls(init_state,
                   ttno,
                   params.time_step_size,
                   1,
                   ops,
                   solver_options={"rtol": 1e-10,
                                   "atol": 1e-10})
    start = time.time()
    time_evo.run(pgbar=False)
    end = time.time()
    return time_evo.results, end - start

def run_reference(params: SimParams1TDVP) -> Results:
    """
    Runs the reference simulation of the Ising Model with the given parameters.
    """
    top = STRUCT_TO_TOP_MAP[params.structure]
    init_state = generate_zero_state(params.system_size,
                                     params.structure,
                                     node_prefix=STANDARD_NODE_PREFIX,
                                     bond_dim=params.bond_dim,
                                     topology=top
                                     )
    model = IsingModel(params.interaction_range,
                       factor=params.factor,
                       ext_magn=params.ext_magn)
    ham = model.generate_by_topology(top,
                                     params.system_size,
                                     site_id_prefix=STANDARD_NODE_PREFIX)
    ttno = TTNO.from_hamiltonian(ham, init_state)
    init_vec, ord = init_state.completely_contract_tree()
    ham_mat, _ = ttno.as_matrix(order=ord)
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
    for i, time in enumerate(res.times()):
        err_res.set_time(i, time)
        err_res.set_element("ref_magn", i, ref_magn[i])
        err_res.set_element("res_magn", i, res_magn[i])
        err_res.set_element("error", i, abs(ref_magn[i] - res_magn[i]))
    return err_res

def main(params: SimParams1TDVP,
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
                SimParams1TDVP)
