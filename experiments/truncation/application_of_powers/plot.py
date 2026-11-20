"""
This module provides plotting functionality for the operator power application experiments.

For each tree structure, creates two plots:
1. Bond dimension vs truncation error
2. Bond dimension vs runtime

The method determines the color and linestyle, while the power determines the marker style.
The legend is shown only on the runtime plot.
"""
import os
from glob import glob
from typing import Dict, List
import argparse

import matplotlib.pyplot as plt
import numpy as np
from h5py import File

from pytreenet.time_evolution.results import Results
from pytreenet.ttns.ttns_ttno.application import ApplicationMethod
from pytreenet.special_ttn.special_states import TTNStructure

from application_common import PowerApplicationParams


# Define color and linestyle mappings for ApplicationMethod
# Using a consistent set of colors and linestyles
METHOD_COLORS = {
    ApplicationMethod.DIRECT: "tab:blue",
    ApplicationMethod.DENSITY_MATRIX: "tab:orange",
    ApplicationMethod.HALF_DENSITY_MATRIX: "tab:green",
    ApplicationMethod.SRC: "tab:red",
    ApplicationMethod.ZIPUP: "tab:purple",
    ApplicationMethod.ZIPUP_CANONICAL: "tab:brown",
    ApplicationMethod.VARIATIONAL: "tab:pink",
    ApplicationMethod.ZIPUP_VARIATIONAL: "tab:gray",
    ApplicationMethod.HALF_DENSITY_MATRIX_VARIATIONAL: "tab:olive",
    ApplicationMethod.DIRECT_TRUNCATE: "tab:cyan",
    ApplicationMethod.DIRECT_TRUNCATE_RANDOM: "black",
}

METHOD_LINESTYLES = {
    ApplicationMethod.DIRECT: "-",
    ApplicationMethod.DENSITY_MATRIX: "--",
    ApplicationMethod.HALF_DENSITY_MATRIX: "-.",
    ApplicationMethod.SRC: ":",
    ApplicationMethod.ZIPUP: (0, (3, 1, 1, 1)),  # dot-dash
    ApplicationMethod.ZIPUP_CANONICAL: (0, (5, 1)),  # long dash
    ApplicationMethod.VARIATIONAL: (0, (1, 1)),  # dotted
    ApplicationMethod.ZIPUP_VARIATIONAL: (0, (3, 1, 1, 1, 1, 1)),  # longer pattern
    ApplicationMethod.HALF_DENSITY_MATRIX_VARIATIONAL: (0, (2, 2)),  # dash with longer spacing
    ApplicationMethod.DIRECT_TRUNCATE: (0, (4, 2)),
    ApplicationMethod.DIRECT_TRUNCATE_RANDOM: (0, (1, 2, 1, 2)),
}

# List of available marker styles for different powers
# Using standard matplotlib markers
AVAILABLE_MARKERS = ["o", "s", "D", "^", "v", "*", "P", "X", "d", "h", "p", "H", "+", "x", "|", "_"]


def create_power_marker_mapping(powers: List[int]) -> Dict[int, str]:
    """
    Creates a mapping from power values to marker styles.
    
    Args:
        powers (List[int]): List of power values to create markers for.
    
    Returns:
        Dict[int, str]: Mapping from power value to marker style.
    """
    power_to_marker = {}
    sorted_powers = sorted(powers)
    for i, power in enumerate(sorted_powers):
        # Cycle through available markers if we have more powers than markers
        marker_index = i % len(AVAILABLE_MARKERS)
        power_to_marker[power] = AVAILABLE_MARKERS[marker_index]
    return power_to_marker


def load_all_results(result_directory: str) -> Dict[str, Dict]:
    """
    Loads all results from the given directory.

    Args:
        result_directory (str): The directory containing the result files.

    Returns:
        Dict[str, Dict]: A dictionary mapping hash to a dictionary containing
            params and results.
    """
    result_files = glob(os.path.join(result_directory, "*.h5"))
    results_dict = {}
    
    for filepath in result_files:
        with File(filepath, "r") as f:
            params = PowerApplicationParams.load_from_h5(f)
            results = Results.load_from_h5(f)
            results_dict[params.get_hash()] = {
                "params": params,
                "results": results
            }
    
    return results_dict


def get_structure_data(results_dict: Dict[str, Dict], 
                       structure: TTNStructure) -> List[Dict]:
    """
    Filters results for a specific structure.

    Args:
        results_dict (Dict[str, Dict]): Dictionary of all results.
        structure (TTNStructure): The structure to filter by.

    Returns:
        List[Dict]: List of data dictionaries for the specified structure.
    """
    structure_data = []
    for hash_val, data in results_dict.items():
        if data["params"].structure == structure:
            structure_data.append(data)
    return structure_data


def create_plot_for_structure(structure_data: List[Dict],
                              structure: TTNStructure,
                              save_directory: str = None) -> None:
    """
    Creates two plots (error vs bond dim and runtime vs bond dim) for a given structure.

    Args:
        structure_data (List[Dict]): List of result data for the structure.
        structure (TTNStructure): The tree structure.
        save_directory (str, optional): Directory to save plots.
    """
    if not structure_data:
        print(f"No data for structure {structure.value}")
        return

    # Collect all unique methods and powers
    all_methods = set()
    all_powers = set()
    for data in structure_data:
        params = data["params"]
        results = data["results"]
        all_methods.add(params.appl_method)
        all_powers.update(results.results["power"])

    if not all_methods or not all_powers:
        print(f"No methods or powers found for structure {structure.value}")
        return

    # Sort methods and powers for consistent ordering
    sorted_methods = sorted(all_methods, key=lambda m: m.value)
    sorted_powers = sorted(all_powers)

    # Create mapping from power to marker based on the actual powers present
    power_to_marker = create_power_marker_mapping(sorted_powers)

    # Create a single figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # First plot: Bond dimension vs Error (log scale)
    for data in structure_data:
        params = data["params"]
        results = data["results"]
        
        method = params.appl_method
        operator_type = params.operator_type
        
        bond_dims = results.results["bond_dim"]
        trunc_errors = results.results["trunc_error"]
        powers = results.results["power"]
        
        # Group by power
        unique_powers = sorted(set(powers))
        for power in unique_powers:
            power_indices = [i for i, p in enumerate(powers) if p == power]
            power_bond_dims = [bond_dims[i] for i in power_indices]
            power_errors = [trunc_errors[i] for i in power_indices]
            
            # Sort by bond dimension
            sorted_indices = np.argsort(power_bond_dims)
            power_bond_dims = np.array(power_bond_dims)[sorted_indices]
            power_errors = np.array(power_errors)[sorted_indices]
            
            # Get color, linestyle, and marker
            color = METHOD_COLORS.get(method, "black")
            linestyle = METHOD_LINESTYLES.get(method, "-")
            marker = power_to_marker.get(power, "o")
            
            # Create label only for the first power to avoid duplicate legend entries
            # We'll add labels to the runtime plot only
            label = None
            
            ax1.semilogy(power_bond_dims, power_errors, 
                        linestyle=linestyle, color=color, marker=marker,
                        label=label, markersize=8)

    # Second plot: Bond dimension vs Runtime (linear scale)
    # Track which method-power combinations we've already added to legend
    legend_entries = {}
    for data in structure_data:
        params = data["params"]
        results = data["results"]
        
        method = params.appl_method
        operator_type = params.operator_type
        
        bond_dims = results.results["bond_dim"]
        runtimes = results.results["run_time"]
        powers = results.results["power"]
        
        # Group by power
        unique_powers = sorted(set(powers))
        for power in unique_powers:
            power_indices = [i for i, p in enumerate(powers) if p == power]
            power_bond_dims = [bond_dims[i] for i in power_indices]
            power_runtimes = [runtimes[i] for i in power_indices]
            
            # Sort by bond dimension
            sorted_indices = np.argsort(power_bond_dims)
            power_bond_dims = np.array(power_bond_dims)[sorted_indices]
            power_runtimes = np.array(power_runtimes)[sorted_indices]
            
            # Get color, linestyle, and marker
            color = METHOD_COLORS.get(method, "black")
            linestyle = METHOD_LINESTYLES.get(method, "-")
            marker = power_to_marker.get(power, "o")
            
            # Create label for legend
            # Format: "method (power)"
            label_key = (method.value, power)
            if label_key not in legend_entries:
                label = f"{method.value} (power={power})"
                legend_entries[label_key] = True
            else:
                label = None
            
            ax2.plot(power_bond_dims, power_runtimes,
                     linestyle=linestyle, color=color, marker=marker,
                     label=label, markersize=8)

    # Configure first plot (Error)
    ax1.set_xlabel("Bond Dimension")
    ax1.set_ylabel("Truncation Error")
    ax1.set_title(f"{structure.value.upper()}: Error vs Bond Dimension")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    # Configure second plot (Runtime) with legend
    ax2.set_xlabel("Bond Dimension")
    ax2.set_ylabel("Runtime (seconds)")
    ax2.set_title(f"{structure.value.upper()}: Runtime vs Bond Dimension")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    
    # Add legend to runtime plot
    if legend_entries:
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Save or show
    if save_directory:
        os.makedirs(save_directory, exist_ok=True)
        error_path = os.path.join(save_directory, f"{structure.value}_error_vs_bonddim.png")
        runtime_path = os.path.join(save_directory, f"{structure.value}_runtime_vs_bonddim.png")
        
        # Save both plots separately
        fig.savefig(error_path, bbox_inches='tight', dpi=300)
        fig.savefig(runtime_path, bbox_inches='tight', dpi=300)
    
    plt.close(fig)


def create_all_plots(results_dict: Dict[str, Dict],
                     save_directory: str = None) -> None:
    """
    Creates plots for all tree structures.

    Args:
        results_dict (Dict[str, Dict]): Dictionary of all results.
        save_directory (str, optional): Directory to save plots.
    """
    # Get all unique structures from the results
    all_structures = set()
    for hash_val, data in results_dict.items():
        all_structures.add(data["params"].structure)
    
    # Sort structures for consistent ordering
    sorted_structures = sorted(all_structures, key=lambda s: s.value)
    
    # Create plots for each structure
    for structure in sorted_structures:
        structure_data = get_structure_data(results_dict, structure)
        create_plot_for_structure(structure_data, structure, save_directory)


def main():
    """
    Main function to run all plots.
    """
    
    parser = argparse.ArgumentParser(
        description="Plot results for operator power application experiments"
    )
    parser.add_argument(
        "result_directory", 
        type=str, 
        help="Directory containing the result files"
    )
    parser.add_argument(
        "--save", 
        type=str, 
        default=None,
        help="Directory to save plots"
    )
    
    args = parser.parse_args()
    
    results_dict = load_all_results(args.result_directory)
    
    if not results_dict:
        print("No results found in the specified directory.")
        return
    
    print(f"Loaded {len(results_dict)} result files.")
    
    # Create all plots
    create_all_plots(results_dict, args.save)
    
    print("Plots generated successfully.")


if __name__ == "__main__":
    main()
