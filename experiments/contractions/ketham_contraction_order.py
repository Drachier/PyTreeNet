"""
Measures the memory saved by splitting the environment blocks around the
Hamiltonian tensor in `contract_ket_ham_with_envs`.

Contracting a neighbour block into the ket tensor replaces the virtual leg of
dimension chi to that neighbour by two legs of dimensions m and chi, with m the
Hamiltonian bond dimension towards that neighbour. Contracting every block
before the Hamiltonian tensor therefore builds an intermediate carrying the
product of all Hamiltonian bond dimensions, while splitting the neighbours into
a group contracted before and a group contracted after it leaves only one of
the two partial products alive at a time.

Both orders are run on balanced binary trees under two-site TDVP, where a
merged pair of degree-three nodes has four neighbours and the Hamiltonian bonds
towards them grow near the root. The peak is taken from `tracemalloc`, which is
started after the state and the operator are built, so it measures the working
memory of the evolution alone.

The default run over trees of 4, 8 and 16 sites takes about one and a half
minutes. Pass --depths to change the trees, for example `--depths 6` for the 32
site case, which takes about six minutes.
"""
from __future__ import annotations

from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from unittest.mock import patch
import os
import tracemalloc

import matplotlib.pyplot as plt
from numpy.typing import NDArray
import numpy as np

from pytreenet.contractions import state_operator_contraction
from pytreenet.operators.models.two_site_model import FlippedIsingModel
from pytreenet.operators.tensorproduct import TensorProduct
from pytreenet.operators.common_operators import pauli_matrices
from pytreenet.random.random_special_ttns import random_binary_state
from pytreenet.time_evolution.tdvp_algorithms.secondordertwosite import (
    SecondOrderTwoSiteTDVP, TwoSiteTDVPConfig)
from pytreenet.time_evolution.time_evolution import TimeEvoMode
from pytreenet.ttno.ttno_class import TTNO
from pytreenet.util.plotting.configuration import (config_matplotlib_to_latex,
                                                   set_size, DocumentStyle)

MEGA = 1024 ** 2

@dataclass
class Measurement:
    """
    One run of the time evolution.

    Attributes:
        peak (float): The peak memory traced during the run, in MiB.
        runtime (float): The wall time of the run, in seconds.
        observables (NDArray[np.complex128]): The measured results, kept to
            check that the two orders agree.
    """
    peak: float
    runtime: float
    observables: NDArray[np.complex128]

def physical_sites(ttns) -> list[str]:
    """
    The identifiers of the nodes carrying a non-trivial physical leg.

    The internal nodes of a binary state have an open leg of dimension one, so
    only the leaves are sites of the model.

    Args:
        ttns (TreeTensorNetworkState): The state to inspect.

    Returns:
        list[str]: The site identifiers, sorted.
    """
    return sorted(nid for nid, node in ttns.nodes.items()
                  if ttns.tensors[nid].shape[node.open_legs[0]] > 1)

def build(depth: int, bond_dim: int):
    """
    Builds a binary tree state and a TTNO with growing Hamiltonian bonds.

    The sites are the leaves, threaded by a transverse field and by nearest and
    next-nearest neighbour ZZ couplings. Terms crossing the bonds near the root
    accumulate, so the Hamiltonian bond dimensions there grow with the depth.

    Args:
        depth (int): The depth of the binary tree.
        bond_dim (int): The bond dimension of the state.

    Returns:
        tuple: The state and the TTNO.
    """
    ttns = random_binary_state(depth, 2, bond_dim)
    sites = physical_sites(ttns)
    nearest = FlippedIsingModel(interaction_range=1, factor=1.0, ext_magn=0.5)
    next_nearest = FlippedIsingModel(interaction_range=2, factor=0.5,
                                     ext_magn=0.0)
    ham = nearest.generate_hamiltonian(
        nearest.generate_chain_structure(sites))
    ham.add_hamiltonian(next_nearest.generate_hamiltonian(
        next_nearest.generate_chain_structure(sites)))
    ham.include_identities([1, 2])
    return ttns, TTNO.from_hamiltonian(ham, ttns)

def ham_bond_dims(ttno: TTNO) -> dict[int, int]:
    """
    How often every Hamiltonian bond dimension occurs in the TTNO.

    Args:
        ttno (TTNO): The operator to inspect.

    Returns:
        dict[int, int]: The bond dimensions and their multiplicities.
    """
    dims: dict[int, int] = {}
    for nid, node in ttno.nodes.items():
        for neighbour in node.neighbouring_nodes():
            dim = ttno.tensors[nid].shape[node.neighbour_index(neighbour)]
            dims[dim] = dims.get(dim, 0) + 1
    return dict(sorted(dims.items()))

def run(ttns,
        ttno: TTNO,
        bond_dim: int,
        time_step_size: float,
        final_time: float,
        split: bool
        ) -> Measurement:
    """
    Runs two-site TDVP on a binary tree under one of the two orders.

    The state is copied, so that both orders start from the same one and their
    observables can be compared.

    Args:
        ttns (TreeTensorNetworkState): The initial state.
        ttno (TTNO): The Hamiltonian.
        bond_dim (int): The bond dimension cap.
        time_step_size (float): The size of one time step.
        final_time (float): The time until which to evolve.
        split (bool): Whether the blocks are split around the Hamiltonian
            tensor. If False, every block is contracted first, which is the
            order the split replaces.

    Returns:
        Measurement: The peak memory, the wall time and the observable.
    """
    ttns = deepcopy(ttns)
    sites = physical_sites(ttns)
    # One local observable, so that the two orders can be compared.
    operators = [TensorProduct({sites[0]: pauli_matrices()[2]})]
    config = TwoSiteTDVPConfig(max_bond_dim=bond_dim,
                               time_evo_mode=TimeEvoMode.RK45)
    time_evo = SecondOrderTwoSiteTDVP(ttns, ttno,
                                      time_step_size, final_time,
                                      operators,
                                      config=config,
                                      solver_options={"rtol": 1e-8,
                                                      "atol": 1e-8})
    def keep_all(ham_dims, **kwargs):
        # Every block in front of the Hamiltonian is the order the split
        # improves upon, so the baseline is the split that keeps all of them.
        del kwargs
        return tuple(range(len(ham_dims)))
    tracemalloc.start()
    tracemalloc.reset_peak()
    start = perf_counter()
    if split:
        time_evo.run(pgbar=False)
    else:
        with patch.object(state_operator_contraction,
                          "_neighbours_before_ham", keep_all):
            time_evo.run(pgbar=False)
    runtime = perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return Measurement(peak / MEGA, runtime,
                       np.asarray(time_evo.results.operator_results()))

def plot(depths: list[int],
         split: list[Measurement],
         unsplit: list[Measurement],
         path: str) -> None:
    """
    Plots the peak memory of both orders against the tree depth.

    Args:
        depths (list[int]): The tree depths measured.
        split (list[Measurement]): The runs with the blocks split.
        unsplit (list[Measurement]): The runs with every block in front.
        path (str): Where to save the figure.
    """
    config_matplotlib_to_latex()
    fig, axs = plt.subplots(1, 2,
                            figsize=set_size(DocumentStyle.THESIS,
                                             subplots=(1, 2)))
    sites = [2 ** (depth - 1) for depth in depths]
    axs[0].plot(sites, [run.peak for run in unsplit], "o-",
                label="all blocks first")
    axs[0].plot(sites, [run.peak for run in split], "s-",
                label="split around $H$")
    axs[0].set_xlabel("sites")
    axs[0].set_ylabel("peak memory / MiB")
    axs[0].set_yscale("log")
    axs[0].legend()
    ratio = [before.peak / after.peak
             for before, after in zip(unsplit, split)]
    axs[1].plot(sites, ratio, "o-")
    axs[1].axhline(1.0, color="grey", linestyle=":")
    axs[1].set_xlabel("sites")
    axs[1].set_ylabel("peak memory saved")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    print(f"figure written to {path}")

def main() -> None:
    """
    Runs both orders on a few trees and reports the saving.
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--depths", type=int, nargs="+", default=[3, 4, 5],
                        help="the depths of the binary trees")
    parser.add_argument("--bond-dim", type=int, default=12,
                        help="the bond dimension of the state and the cap")
    parser.add_argument("--time-step-size", type=float, default=0.05)
    parser.add_argument("--final-time", type=float, default=0.1)
    args = parser.parse_args()
    split, unsplit = [], []
    print(f"{'sites':>6} {'ham bonds':>24} {'peak split':>11} "
          f"{'peak all first':>15} {'saved':>7} {'time split':>11} "
          f"{'time all first':>15} {'rel diff':>10}")
    for depth in args.depths:
        ttns, ttno = build(depth, args.bond_dim)
        bonds = ham_bond_dims(ttno)
        with_split = run(ttns, ttno, args.bond_dim, args.time_step_size,
                         args.final_time, split=True)
        without = run(ttns, ttno, args.bond_dim, args.time_step_size,
                      args.final_time, split=False)
        split.append(with_split)
        unsplit.append(without)
        # Reordering only changes the association of the floating point
        # operations, so the observables have to agree to round-off.
        scale = np.abs(without.observables).max()
        rel_diff = np.abs(with_split.observables
                          - without.observables).max() / scale
        print(f"{2 ** (depth - 1):>6} {str(bonds):>24} "
              f"{with_split.peak:>10.1f}M {without.peak:>14.1f}M "
              f"{without.peak / with_split.peak:>6.1f}x "
              f"{with_split.runtime:>10.1f}s {without.runtime:>14.1f}s "
              f"{rel_diff:>10.1e}")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "ketham_contraction_order.pdf")
    plot(args.depths, split, unsplit, path)

if __name__ == "__main__":
    main()
