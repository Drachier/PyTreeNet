"""
This file is used to plot the time evolution experiment's results.
"""
from typing import Any, Self
import os
import json
from copy import copy

from h5py import File
import numpy as np
import matplotlib.pyplot as plt

from pytreenet.time_evolution.results import Results
from pytreenet.operators.models import total_magnetisation

from sim_script import FILENAME_PREFIX, NODE_PREFIX

METADATA_LOG_FILE = "metadata.json"
STANDARD_ATTR_FILTER = {"topology": ["chain"],
                        "strength": [0.1],
                        "init_bond_dim": [2],
                        "time_evo_algorithm": ["bug"],
                        "time_step_size": [0.01],
                        "final_time": [1.0],
                        "rel_svalue": [1e-15],
                        "abs_svalue": [1e-15],
                        "status": ["success"]}

def filter_metadata_log(metadata: dict[str, Any],
                        attrs_filter: dict[str, list[Any]],
                        ) -> list[str]:
    """
    Filters the metadata log based on the given attributes.

    Args:
        metadata (dict[str, Any]): The metadata log to filter.
        attrs_filter (dict[str, list[Any]]): A dictionary where keys are
            attribute names and values are lists of acceptable values for those
            attributes.
        
    Returns:
        list[str]: A list of parameter hashes that match the filter criteria.
    """
    attrs_filter = {key: [str(value) for value in lst]
                        for key, lst in (attrs_filter or {}).items()}
    hashes = []
    for param_hash, result_data in metadata.items():
        valid = True
        for attr, val_lst in attrs_filter.items():
            if attr not in result_data:
                valid = False
            if str(result_data[attr]) not in val_lst:
                valid = False
        if valid:
            hashes.append(param_hash)
    return hashes

def load_results(dir_path: str,
                    obs_ids: list[str] | None = None,
                    attrs_filter: dict[str, list[Any]] | None = None,
                    metadata: dict[str, Any] | None = None
                    ) -> Self:
    """
    Loads the results filtering via the metadata log file.

    Args:
        dir_path (str): The directory path where the metadata log file is
            located.
        obs_ids (list[str] | None): A list of observation IDs to filter the
            results. If None, all observation IDs are included.
        attrs_filter (dict[str, list[Any]]] | None): A dictionary to filter the
            results based on specific attributes. Only files with attributes
            that match the values in the lists will be included. If None, no
            filtering is applied.
        
    
    Returns:
        Loader: An instance of the Loader class with the loaded results.
    """
    if metadata is None:
        metadata = load_metadata_log(dir_path)
    results = {}
    hashes = filter_metadata_log(metadata, attrs_filter or {})
    for param_hash in hashes:
        result_path = os.path.join(dir_path, FILENAME_PREFIX + f"{param_hash}.h5")
        result = Results.load_from_h5(result_path,
                                    loaded_ops=obs_ids)
        results[param_hash] = result
    return results

def load_metadata_log(dir_path: str) -> dict[str, Any]:
    """
    Loads the metadata log file from the given directory path.

    Args:
        dir_path (str): The directory path where the metadata log file is
            located.

    Returns:
        dict[str, Any]: A dictionary containing the metadata from the log file.
    """
    meta_data_log_path = os.path.join(dir_path, METADATA_LOG_FILE)
    if not os.path.exists(meta_data_log_path):
        raise FileNotFoundError(f"Metadata log file not found: {meta_data_log_path}")
    with open(meta_data_log_path, "r") as f:
        metadata = json.load(f)
    return metadata

def load_runtimes(dir_path: str,
                  attrs_filter: dict[str, list[Any]] | None = None,
                  metadata: dict[str, Any] | None = None,
                  ) -> dict[str, float]:
    """
    Loads the runtimes from the metadata log file in the given directory path.

    Args:
        dir_path (str): The directory path where the metadata log file is
            located.
        attrs_filter (dict[str, list[Any]] | None): A dictionary to filter the
            results based on specific attributes. Only files with attributes
            that match the values in the lists will be included. If None, no
            filtering is applied.
        metadata (dict[str, Any] | None): A dictionary containing the metadata
            from the log file. If None, it will be loaded from the directory.
    
    Returns:
        dict[str, float]: A dictionary where keys are parameter hashes and
            values are the corresponding runtimes in seconds.
    """
    if metadata is None:
        metadata = load_metadata_log(dir_path)
    hashes = filter_metadata_log(metadata, attrs_filter or {})
    runtimes = {}
    for param_hash in hashes:
        with File(os.path.join(dir_path, FILENAME_PREFIX + f"{param_hash}.h5"), "r") as f:
            if "elapsed_time" in f.attrs:
                elapsed_time = f.attrs["elapsed_time"]
                if isinstance(elapsed_time, (int, float)):
                    runtimes[param_hash] = elapsed_time
                else:
                    raise ValueError(f"Invalid elapsed_time type for {param_hash}: {type(elapsed_time)}")
            else:
                raise KeyError(f"elapsed_time attribute not found for {param_hash}")
    return runtimes

def total_magn_from_results(results: Results,
                            observable_ids: list[str]) -> float:
    """
    Computes the total magnetisation from the results.

    Args:
        results (Results): The results object containing the simulation data.
        observable_ids (list[str]): A list of observable IDs to compute the
            total magnetisation from.

    Returns:
        float: The total magnetisation.
    """
    local_magns = [results.operator_result(obs_id)
                   for obs_id in observable_ids]
    return total_magnetisation(local_magns)

LINESTYLE_MAP = {"mps": "-",
                    "binary": "--"}

MARKER_MAP = {"mps": "o",
                    "binary": "x"}

COLOR_MAP = {"expm": "blue",
             "chebyshev": "green",
             "RK45": "orange",
             "RK23": "purple",
             "BDF": "yellow",
             "DOP853": "grey"}

def plot_delta_to_expm_vs_time(dir_path: str):
    """
    Plots the delta to expm vs time for the results in the given directory.

    Args:
        dir_path (str): The directory path where the results are stored.
    """
    num_sites = 15
    obs_ids = [NODE_PREFIX + str(i) for i in range(num_sites)]
    obs_ids = obs_ids + ["times"]
    metadata = load_metadata_log(dir_path)
    attrs_filter = copy(STANDARD_ATTR_FILTER)
    attrs_filter.update({"status": ["success"],
                    "max_bond_dim": [20],
                    "num_sites": [num_sites],
                    "interaction_length": [2],
                    "time_evo_method": ["expm"]})
    exact_results = load_results(dir_path,
                                obs_ids=obs_ids,
                                attrs_filter=attrs_filter,
                                metadata=metadata)
    exmp_total_magn = {metadata[md_hash]["ttns_structure"]:
                       total_magn_from_results(result, obs_ids[:-1])
                       for md_hash, result in exact_results.items()}
    attrs_filter["time_evo_method"] = ["chebyshev", "RK45", "RK23", "BDF", "DOP853"]
    other_data = load_results(dir_path,
                                obs_ids=obs_ids,
                                attrs_filter=attrs_filter,
                                metadata=metadata)
    other_total_magn = {(metadata[md_hash]["time_evo_method"],
                          metadata[md_hash]["ttns_structure"]):
                        total_magn_from_results(result, obs_ids[:-1])
                        for md_hash, result in other_data.items()}
    delta = {(evo_method, ttn_structure):
             np.abs(exmp_total_magn[ttn_structure] -
                    other_total_magn[(evo_method, ttn_structure)])
             for (evo_method, ttn_structure) in other_total_magn.keys()}
    times = other_data[next(iter(other_data))].times()
    plt.figure(figsize=(10, 6))
    for (evo_method, ttn_structure), deltas in delta.items():
        plt.semilogy(times[:-1], deltas[:-1], label=f"{evo_method} - {ttn_structure}",
                     ls=LINESTYLE_MAP.get(ttn_structure, "-"))
    plt.xlabel("Time $t$")
    plt.ylabel("$\Delta$ to expm")
    plt.legend()
    plt.show()

def plot_runtime_vs_num_sites(dir_path: str):
    """
    Plots the runtime vs number of sites for the results in the given directory.

    Args:
        dir_path (str): The directory path where the results are stored.
    """
    metadata = load_metadata_log(dir_path)
    attrs_filter = copy(STANDARD_ATTR_FILTER)
    methods = ["expm", "chebyshev", "RK45", "RK23", "BDF", "DOP853"]
    state_structures = ["mps", "binary"]
    attrs_filter.update({"status": ["success"],
                        "max_bond_dim": [20],
                        "interaction_length": [2]})
    runtimes = {}
    for method in methods:
        for state_structure in state_structures:
            attrs_filter["time_evo_method"] = [method]
            attrs_filter["ttns_structure"] = [state_structure]
            filtered_runtimes = load_runtimes(dir_path, attrs_filter,
                                              metadata=metadata)
            filtered_runtimes = [(metadata[md_hash]["num_sites"],rt)
                                 for md_hash, rt in filtered_runtimes.items()]
            if filtered_runtimes:
                runtimes[(method, state_structure)] = filtered_runtimes
    plt.figure(figsize=(10, 6))
    for (method, state_structure), rt_data in runtimes.items():
        rt_data = sorted(rt_data, key=lambda x: x[0])
        num_sites, times = zip(*rt_data)
        plt.plot(num_sites, times, label=f"{method} - {state_structure}",
                 marker=MARKER_MAP.get(state_structure, "."),
                 ls=LINESTYLE_MAP.get(state_structure, "-"),
                 color=COLOR_MAP.get(method, "black"))
    plt.xlabel("Number of Sites")
    plt.ylabel("Runtime (seconds)")
    plt.yscale("log")
    plt.legend()
    plt.xlim(left=5)
    plt.show()

def plot_runtime_vs_max_bd(dir_path: str):
    """
    Plots the runtime vs maximum bond dimension for the results in the given directory.

    Args:
        dir_path (str): The directory path where the results are stored.
    """
    metadata = load_metadata_log(dir_path)
    attrs_filter = copy(STANDARD_ATTR_FILTER)
    methods = ["expm", "chebyshev", "RK45", "RK23", "BDF", "DOP853"]
    state_structures = ["mps", "binary"]
    attrs_filter.update({"status": ["success"],
                        "num_sites": [15],
                        "interaction_length": [2]})
    runtimes = {}
    for method in methods:
        for state_structure in state_structures:
            attrs_filter["time_evo_method"] = [method]
            attrs_filter["ttns_structure"] = [state_structure]
            filtered_runtimes = load_runtimes(dir_path, attrs_filter,
                                              metadata=metadata)
            filtered_runtimes = [(metadata[md_hash]["max_bond_dim"],rt)
                                 for md_hash, rt in filtered_runtimes.items()]
            if filtered_runtimes:
                runtimes[(method, state_structure)] = filtered_runtimes
    plt.figure(figsize=(10, 6))
    for (method, state_structure), rt_data in runtimes.items():
        rt_data = sorted(rt_data, key=lambda x: x[0])
        max_bd, times = zip(*rt_data)
        plt.plot(max_bd, times, label=f"{method} - {state_structure}",
                 marker=MARKER_MAP.get(state_structure, "."),
                 ls=LINESTYLE_MAP.get(state_structure, "-"),
                 color=COLOR_MAP.get(method, "black"))
    plt.xlabel("Maximum Bond Dimension")
    plt.ylabel("Runtime (seconds)")
    plt.yscale("log")
    plt.legend()
    plt.xlim(left=5)
    plt.show()

def plot_runtime_bar_chart(dir_path: str):
    """
    Plots a bar chart of the runtimes for the results in the given directory.

    Args:
        dir_path (str): The directory path where the results are stored.
    """
    metadata = load_metadata_log(dir_path)
    attrs_filter = copy(STANDARD_ATTR_FILTER)
    attrs_filter.update({"status": ["success"],
                        "max_bond_dim": [20],
                        "num_sites": [15],
                        "interaction_length": [2]})
    runtimes = load_runtimes(dir_path, attrs_filter,
                             metadata=metadata)
    methods = ["expm", "chebyshev", "RK45", "RK23", "BDF", "DOP853"]
    mps_results = [rt for md_hash, rt in runtimes.items()
                   if metadata[md_hash]["ttns_structure"] == "mps"
                   and metadata[md_hash]["time_evo_method"] in methods]
    binary_results = [rt for md_hash, rt in runtimes.items()
                      if metadata[md_hash]["ttns_structure"] == "binary"
                      and metadata[md_hash]["time_evo_method"] in methods]
    x = np.arange(len(methods))
    bar_width = 0.35
    _, ax = plt.subplots()

    # Plot bars to the left and right of the central ticks
    ax.bar(x - bar_width/2,
                   mps_results,
                   width=bar_width,
                   color='green')
    ax.bar(x + bar_width/2,
                   binary_results,
                   width=bar_width,
                   color='blue')
    ax.set_xticks(x, methods)
    ax.set_xlabel("Time Evolution Method")
    ax.set_yscale("log")
    ax.set_ylabel("Runtime (seconds)")
    ax.plot([],[], color='green', label='MPS')
    ax.plot([],[], color='blue', label='Binary')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    DIR_PATH="/work/ge24fum/diss_data/pytreenet/time_evo_benchmark"
    plot_delta_to_expm_vs_time(DIR_PATH)
