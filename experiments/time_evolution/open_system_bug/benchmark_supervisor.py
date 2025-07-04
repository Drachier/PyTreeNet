import os
import json
import subprocess
import logging
import h5py
from time import time
from enum import Enum
import pandas as pd
import hashlib
from numpy import array, savez, load, abs, ndarray
from IPython.display import display, clear_output
from typing import Union, Dict, Tuple, List
from experiments.time_evolution.open_system_bug.sim_script import (get_param_hash, 
                                                                   CURRENT_PARAM_FILENAME, 
                                                                   INITIAL_STATE_ZERO)
from pytreenet.special_ttn.binary import PHYS_PREFIX
from experiments.time_evolution.open_system_bug.parameters import generate_parameter
from pytreenet.util.tensor_splitting import SVDParameters
from experiments.time_evolution.open_system_bug.sim_script import TTNStructure
from experiments.time_evolution.open_system_bug.exact_time_evolution.TFI_1D_Qutip import TFI_1D_Qutip
from pytreenet.time_evolution.results import Results
import matplotlib.pyplot as plt
from collections import defaultdict


class Status(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"

LOG_FILE_NAME = "benchmark_log.txt"

def params_to_dict(params):
    """Convert parameter tuple to dictionary."""
    sim_params, time_evo_params = params
    dictionary = sim_params.to_dict()
    dictionary.update(time_evo_params.to_dict())
    return dictionary

def setup_benchmark_directory(benchmark_dir):
    """Setup benchmark directory and load existing results."""
    os.makedirs(benchmark_dir, exist_ok=True)
    
    metadata_file = os.path.join(benchmark_dir, "benchmark_results.json")
    existing_results = {}
    
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️  Corrupted metadata file found..")
            existing_results = {}
    
    # Setup logging
    log_path = os.path.join(benchmark_dir, LOG_FILE_NAME)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'  # Append to existing log
    )
    
    return existing_results, metadata_file

def get_or_compute_exact(sim_params, time_evo_params, benchmark_dir):
    # Build physical-only hash from key sim and time-evo fields
    phys_hash_dict = {"num_sites": sim_params.num_sites,
                    "coupling": sim_params.coupling,
                    "ext_magn": sim_params.ext_magn,
                    "relaxation_rate": sim_params.relaxation_rate,
                    "dephasing_rate": sim_params.dephasing_rate,
                    "final_time": time_evo_params.final_time}

    phys_hash = hashlib.md5(json.dumps(phys_hash_dict, sort_keys=True).encode()).hexdigest()
    exact_file = os.path.join(benchmark_dir, f"{phys_hash}_exact.npz")
    # load if exists
    if os.path.exists(exact_file):
        data = load(exact_file)
        logging.info(f" Loaded exact solution from cache: {exact_file}")
        return data['magns'], data['energy']
    # otherwise compute exact via QuTiP
    tfi = TFI_1D_Qutip(L=sim_params.num_sites,
                        coupling=sim_params.coupling,
                        ext_magn=sim_params.ext_magn,
                        periodic=False,
                        use_single_precision=False)

    initial_state = tfi.uniform_product_state(INITIAL_STATE_ZERO)
    jump_ops = tfi.build_jump_operators(sim_params.relaxation_rate, sim_params.dephasing_rate)
    e_ops = [sum(tfi.sz_ops)/sim_params.num_sites, tfi.hamiltonian]
    result, _ = tfi.evolve_system_lindblad(initial_state=initial_state,
                                            final_time=time_evo_params.final_time,
                                            time_step_size=time_evo_params.time_step_size,
                                            evaluation_time=time_evo_params.evaluation_time,
                                            jump_operators=jump_ops,
                                            e_ops=e_ops,
                                            options={'method':'bdf','atol':time_evo_params.atol,'rtol':time_evo_params.rtol,'nsteps':1000})

    exact_magns = array(result.expect[:-1])
    exact_energy = array(result.expect[-1])
    
    # Fix shape mismatch: flatten magnetization if needed
    if exact_magns.ndim > 1:
        exact_magns = exact_magns.flatten()
    
    # Save the computed exact solution to cache
    savez(exact_file, magns=exact_magns, energy=exact_energy)
    logging.info(f" Saved exact solution to cache: {exact_file}")
    
    return exact_magns, exact_energy

def run_single_benchmark(parameters, 
                         benchmark_dir, 
                         existing_results, 
                         metadata_file, 
                         ask_overwrite=False):
    """
    Run a single benchmark simulation.

    Args:
        parameters: (sim_params, time_evo_params) tuple
        benchmark_dir: Directory to save all results
        existing_results: Dict of existing benchmark results
        metadata_file: Path to metadata file
        ask_overwrite: Whether to ask for skip/overwrite (default: True)
    
    Returns:
        Tuple[Status, float]: Status enum and elapsed time
    """
    param_hash = get_param_hash(parameters[0], parameters[1])

    # Check if this benchmark already exists
    if param_hash in existing_results:        
        existing_elapsed_time = existing_results[param_hash].get('elapsed_time', 0.0)
        if ask_overwrite:
            choice = input("Choose: [s]kip, [o]verwrite, or [q]uit: ").lower().strip()
            if choice == 'q':
                return Status.SKIPPED, existing_elapsed_time
            elif choice == 's' or choice == '':
                return Status.SKIPPED, existing_elapsed_time
            elif choice != 'o':
                return Status.SKIPPED, existing_elapsed_time
        else:
            return Status.SKIPPED, existing_elapsed_time

    # Run simulation
    start_time = time()
    status = Status.FAILED

    try:
        # Construct the path relative to the project root to avoid working directory issues
        script_path = os.path.join(os.path.dirname(__file__), "sim_script.py")

        # Write parameters for subprocess
        with open(os.path.join(benchmark_dir, CURRENT_PARAM_FILENAME), "w", encoding="utf-8") as f:
            dictionary = params_to_dict(parameters)
            json.dump(dictionary, f, indent=4)
        
        result = subprocess.run(
            ["python", script_path, benchmark_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8', 
            errors='replace',  # Replace problematic characters instead of failing
            check=False,
            timeout=3600)  # 1 hour timeout

        # Log subprocess output regardless of success/failure
        if result.stdout.strip():
            logging.info(f"Subprocess stdout for {param_hash[:12]}:\n{result.stdout}")
        if result.stderr.strip():
            logging.warning(f"Subprocess stderr for {param_hash[:12]}:\n{result.stderr}")

        if result.returncode == 0:
            status = Status.SUCCESS
        else:
            status = Status.FAILED
            logging.error(f"Simulation {param_hash[:12]} failed with return code {result.returncode}")
            
    except subprocess.TimeoutExpired:
        status = Status.TIMEOUT
        logging.error(f"Simulation {param_hash[:12]} timed out after 1 hour")

    end_time = time()
    elapsed_time = end_time - start_time

    # Compute exact solution and error arrays only if simulation succeeded
    sim_params, time_evo_params = parameters

    if status == Status.SUCCESS:
        # Load simulation results first to ensure they exist
        loader = BenchmarkResultLoader(benchmark_dir)
        sim = loader.load_results(param_hash)

        # Compute magnetization from individual nodes
        node_id = PHYS_PREFIX
        bug_magns = sum(sim.results.get(f'{node_id}{i}', 0) for i in range(sim_params.num_sites)) / sim_params.num_sites
        bug_energy = sim.results.get('energy', array([]))

        # Get or compute exact solution
        exact_magns, exact_energy = get_or_compute_exact(sim_params, time_evo_params, benchmark_dir)

        # Check shapes before computing errors
        assert bug_magns.shape == exact_magns.shape
        assert bug_energy.shape == exact_energy.shape

        # Compute error arrays
        mag_error_array = abs(bug_magns - exact_magns)
        energy_error_array = abs(bug_energy - exact_energy)

        # Store error arrays in HDF5 file
        save_file_path = os.path.join(benchmark_dir, f"simulation_{param_hash}.h5")
        with h5py.File(save_file_path, "a") as file:  # Open in append mode
            # Remove existing error datasets if they exist
            if 'mag_error' in file:
                del file['mag_error']
            if 'energy_error' in file:
                del file['energy_error']

            # Add new error datasets
            file.create_dataset('mag_error', data=mag_error_array)
            file.create_dataset('energy_error', data=energy_error_array)

    # Update results
    dictionary = params_to_dict(parameters)
    dictionary["status"] = status.value
    dictionary["elapsed_time"] = elapsed_time
    dictionary["benchmark_hash"] = param_hash

    existing_results[param_hash] = dictionary

    # Save updated results
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(existing_results, f, indent=4)

    return status, elapsed_time

def run_benchmarks(benchmark_dir: str, 
                    ttn_structure: TTNStructure, 
                    svd_params_list: Union[list[SVDParameters], SVDParameters], 
                    depth: int =None, 
                    ask_overwrite: bool = False):
    """
    Run benchmarks for a specific TTN structure with different SVD parameters.

    Args:
        benchmark_dir: Unified directory for all benchmark results
        ttn_structure: TTNStructure enum (MPS, BINARY, etc.)
        svd_params_list: List of SVDParameters to test
        depth: Optional depth parameter
        ask_overwrite: Whether to ask for skip/overwrite if the parameters already exist.

    Returns:
        dict: Summary of results with DataFrame
    """
    # Setup benchmark directory
    existing_results, metadata_file = setup_benchmark_directory(benchmark_dir)

    # Generate all parameter combinations
    all_parameters = []
    for svd_params in svd_params_list:
        params = generate_parameter(ttn_structure=ttn_structure,
                                    depth=depth,
                                    svd_prams=svd_params)
        all_parameters.extend(params)

    # Create DataFrame for live progress tracking
    columns = ['Prameters Hash', 'Algorithm', 'time_step_level', 'truncation_level', 'Status', 'CPU_time']
    df = pd.DataFrame(columns = columns)
    print(f"🔧 TTNDO Structure: {ttn_structure.name} (depth = {depth})")
    display(df)

    # Run benchmarks
    results_summary = {'successful': 0,
                        'failed': 0,
                        'timeout': 0,
                        'skipped': 0,
                        'structure': ttn_structure.name,
                        'dataframe': df}

    for i , parameters in enumerate(all_parameters, 1):
        sim_params, time_evo_params = parameters
        param_hash = get_param_hash(sim_params, time_evo_params)

        # Prepare row data before running benchmark
        row_data = {'Prameters Hash': param_hash[:12],
                    'Algorithm': time_evo_params.time_evo_algorithm.value if time_evo_params.time_evo_algorithm else 'None',
                    'time_step_level': time_evo_params.time_step_level.value if time_evo_params.time_step_level else 'None',
                    'truncation_level': time_evo_params.truncation_level.value if time_evo_params.truncation_level else 'None',
                    'Status': 'Running...',
                    'CPU_time': 0.0}

        # Add row to DataFrame
        df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)

        # Update display
        clear_output(wait=True)
        print(f"🔧 TTNDO Structure: {ttn_structure.name} (depth = {depth})")
        print(f"Running {len(all_parameters)} simulations")
        display(df)

        # Run the benchmark
        status, elapsed_time = run_single_benchmark(parameters, 
                                                    benchmark_dir, 
                                                    existing_results, 
                                                    metadata_file, 
                                                    ask_overwrite)

        # Update the row with final results
        df.loc[df.index[-1], 'Status'] = status.value
        df.loc[df.index[-1], 'CPU_time'] = f"{elapsed_time:.2f}s"

        # Update display with final results
        clear_output(wait=True)
        print(f"🔧 TTNDO Structure: {ttn_structure.name} (depth = {depth})")
        display(df)

        # Update counters
        if status == Status.SUCCESS:
            results_summary['successful'] += 1
        elif status == Status.TIMEOUT:
            results_summary['timeout'] += 1
        elif status == Status.SKIPPED:
            results_summary['skipped'] += 1
        else:
            results_summary['failed'] += 1

class BenchmarkResultLoader:
    """
    Loade benchmark results from JSON metadata and HDF5 files.
    """
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.metadata_file = os.path.join(results_dir, "benchmark_results.json")
        self._metadata = None

    def _load_metadata(self):
        """Load metadata once and cache it."""
        if self._metadata is None:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    self._metadata = json.load(f)
            else:
                self._metadata = {}
        return self._metadata

    def get_available_hashes(self) -> List[str]:
        """Get list of available parameter hashes."""
        metadata = self._load_metadata()
        available_hashes = []
        for param_hash in metadata.keys():
            h5_file = os.path.join(self.results_dir, f"simulation_{param_hash}.h5")
            if os.path.exists(h5_file):
                available_hashes.append(param_hash)
        return available_hashes

    def load_parameters(self, param_hash: str) -> Dict:
        """Load parameter metadata for a given hash."""
        metadata = self._load_metadata()
        if param_hash not in metadata:
            raise ValueError(f"Hash {param_hash} not found in metadata")
        return metadata[param_hash]

    def load_results(self, param_hash: str):
        """Load simulation results from HDF5 file."""
        h5_file = os.path.join(self.results_dir, f"simulation_{param_hash}.h5")
        if not os.path.exists(h5_file):
            raise ValueError(f"HDF5 file not found: {h5_file}")

        with h5py.File(h5_file, 'r') as f:
            # Create a simple results object
            results = Results()
            results.results = {}

            # Load all datasets (skip groups)
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    results.results[key] = f[key][:]
            
            # Add file attributes as metadata
            results.file_attributes = dict(f.attrs)
            
            return results

    def load_error_arrays(self, param_hash: str) -> Tuple[ndarray, ndarray]:
        """
        Load error arrays from HDF5 file.
        
        Returns:
            Tuple[ndarray, ndarray]: (magnetization_error, energy_error)
        """
        h5_file = os.path.join(self.results_dir, f"simulation_{param_hash}.h5")
        if not os.path.exists(h5_file):
            raise ValueError(f"HDF5 file not found: {h5_file}")
                
        with h5py.File(h5_file, 'r') as f:
            if 'mag_error' not in f or 'energy_error' not in f:
                raise ValueError(f"Error arrays not found in {h5_file}")
            
            mag_error = f['mag_error'][:]
            energy_error = f['energy_error'][:]
            
            return mag_error, energy_error

    def load_combined(self, param_hash: str) -> Tuple[Dict, any]:
        """Load both parameters and results for a given hash."""
        return self.load_parameters(param_hash), self.load_results(param_hash)

    def filter_and_load_results(self, filter_params: Dict) -> List[Tuple[Dict, any]]:
        """
        Filter results based on parameter criteria and return all matching (params, results) pairs.
        
        Args:
            filter_params: Dictionary containing parameter criteria to filter by.
                          e.g. {"ttns_structure": "mps", "truncation_level": "fast"}
                          or {"status": "success", "time_step_level": "coarse"}
        
        Returns:
            List[Tuple[Dict, any]]: List of (parameters, results) pairs that match the filter criteria
        """
        metadata = self._load_metadata()
        matching_results = []
        
        # Find all hashes that match the filter criteria
        for param_hash, params in metadata.items():
            # Check if all filter criteria are satisfied
            matches = True
            for key, value in filter_params.items():
                if key not in params or params[key] != value:
                    matches = False
                    break
            
            if matches:
                # Load the results for this hash if HDF5 file exists
                h5_file = os.path.join(self.results_dir, f"simulation_{param_hash}.h5")
                if os.path.exists(h5_file):
                    try:
                        results = self.load_results(param_hash)
                        matching_results.append((params, results))
                    except (ValueError, OSError, h5py.HDF5Error) as e:
                        print(f"⚠️  Warning: Could not load results for hash {param_hash[:12]}: {e}")
                        continue
        
        return matching_results

    def get_matching_hashes(self, filter_params: Dict) -> List[str]:
        """
        Get list of parameter hashes that match the filter criteria.
        
        Args:
            filter_params: Dictionary containing parameter criteria to filter by.
        
        Returns:
            List[str]: List of parameter hashes that match the criteria
        """
        metadata = self._load_metadata()
        matching_hashes = []
        
        for param_hash, params in metadata.items():
            # Check if all filter criteria are satisfied
            matches = True
            for key, value in filter_params.items():
                if key not in params or params[key] != value:
                    matches = False
                    break
            
            if matches:
                # Verify HDF5 file exists
                h5_file = os.path.join(self.results_dir, f"simulation_{param_hash}.h5")
                if os.path.exists(h5_file):
                    matching_hashes.append(param_hash)
        
        return matching_hashes
    

def load_specific_result(param_hash: str, results_dir) -> Tuple[Dict, any]:
    """Load a specific result by parameter hash."""
    loader = BenchmarkResultLoader(results_dir)
    return loader.load_combined(param_hash)

def load_filtered_results(filter_params: Dict, results_dir: str) -> List[Tuple[Dict, any]]:
    """
    Load all results that match the specified parameter criteria.
    
    Args:
        filter_params: Dictionary containing parameter criteria to filter by.
                       retutns all results if None.
        results_dir: Directory containing benchmark results
    
    Returns:
        List[Tuple[Dict, any]]: List of (parameters, results) pairs that match the filter criteria
    """
    if filter_params is None:
        filter_params = {}

    loader = BenchmarkResultLoader(results_dir)
    return loader.filter_and_load_results(filter_params)


def plot_benchmark_errors(filter_criteria, benchmark_dir, save_path=None, figsize=(15, 10)):
    """
    Create error vs time plots grouped by algorithm+method combinations.
    
    Args:
        filter_criteria: Dict to filter results (e.g., {"max_bond_dim": 8})
        benchmark_dir: Directory containing benchmark results
        save_path: Optional path to save the plots
        figsize: Figure size (width, height)
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Load filtered results
    filtered_results = load_filtered_results(filter_criteria, benchmark_dir)
    
    if not filtered_results:
        print(f"❌ No results found for criteria: {filter_criteria}")
        return None
    
    print(f"📊 Found {len(filtered_results)} results matching criteria: {filter_criteria}")
    
    # Extract TTN structure and depth from first result (they should be the same for all)
    first_params = filtered_results[0][0]
    ttn_structure = first_params.get('ttns_structure', 'unknown')
    depth = first_params.get('depth', 'unknown')
    
    # Group results by (algorithm, method)
    groups = defaultdict(list)
    loader = BenchmarkResultLoader(benchmark_dir)
    
    for params, results in filtered_results:
        algorithm = params.get('time_evo_algorithm', 'unknown')
        method = params.get('time_evo_method', 'unknown')
        group_key = (algorithm, method)
        
        # Load error arrays for this result
        try:
            param_hash = params['benchmark_hash']
            mag_error, energy_error = loader.load_error_arrays(param_hash)
            
            # Get time array and norm data
            times = results.results.get('times', None)
            norm = results.results.get('norm', results.results.get('Identity', None))
            
            if times is not None and norm is not None:
                # Calculate norm deviation
                norm_error = abs(norm - 1)
                
                # Store all data for this simulation
                sim_data = {
                    'params': params,
                    'times': times,
                    'energy_error': energy_error,
                    'mag_error': mag_error,
                    'norm_error': norm_error,
                    'truncation_level': params.get('truncation_level', 'unknown'),
                    'time_step_level': params.get('time_step_level', 'unknown')
                }
                groups[group_key].append(sim_data)
                
        except Exception as e:
            print(f"⚠️  Warning: Could not load error data for hash {params.get('benchmark_hash', 'unknown')[:12]}: {e}")
            continue
    
    if not groups:
        print("❌ No valid data found after loading error arrays")
        return None
    
    # Define color schemes
    truncation_colors = {
        'fast': 'blue',
        'balanced': 'green', 
        'rigorous': 'red',
        'unknown': 'gray'
    }
    
    # Define line styles for time step levels
    time_step_linestyles = {
        'coarse': '-',      # solid line
        'medium': '--',     # dashed line
        'fine': ':',        # dotted line
        'unknown': '-.'     # dash-dot line
    }
    
    # Create figure with subplots
    n_groups = len(groups)
    fig, axes = plt.subplots(n_groups, 3, figsize=figsize, squeeze=False)
    fig.suptitle(f'TTNDO Structure: {ttn_structure} / depth = {depth}', fontsize=16, fontweight='bold')
    
    # Plot titles for the three error types
    error_titles = ['Energy Error', 'Total Magnetization Error', 'Norm Deviation |norm - 1|']
    error_keys = ['energy_error', 'mag_error', 'norm_error']
    
    # Plot each group
    for group_idx, ((algorithm, method), sims) in enumerate(groups.items()):
        group_title = f"{algorithm} with {method} ODE-solver"
        
        # Plot each error type
        for error_idx, (error_key, error_title) in enumerate(zip(error_keys, error_titles)):
            ax = axes[group_idx, error_idx]
            
            # Plot each simulation in this group
            for sim in sims:
                trunc_level = sim['truncation_level']
                time_level = sim['time_step_level']
                
                # Get base color and line style
                color = truncation_colors.get(trunc_level, 'gray')
                linestyle = time_step_linestyles.get(time_level, '-.')
                
                # Plot the error (no label to avoid individual legends)
                times = sim['times']
                error_data = sim[error_key]
                
                # Handle potential shape mismatches
                if len(times) != len(error_data):
                    min_len = min(len(times), len(error_data))
                    times = times[:min_len]
                    error_data = error_data[:min_len]
                
                ax.plot(times, error_data, color=color, linestyle=linestyle, 
                       linewidth=2, alpha=0.8)
            
            # Customize subplot
            ax.set_xlabel('Time')
            ax.set_ylabel(error_title)
            ax.set_title(group_title)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')  # Use log scale for better error visualization
    
    # Create a custom legend for the color scheme
    legend_elements = []
    for trunc, color in truncation_colors.items():
        if trunc != 'unknown':
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=4, label=f'Truncation level: {trunc}'))
    
    legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='-', lw=2, label='Time_step level: coarse'))
    legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='--', lw=2, label='Time_step level: medium'))
    legend_elements.append(plt.Line2D([0], [0], color='black', linestyle=':', lw=2, label='Time_step level: fine'))
    
    # Add main legend outside the plot area
    fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.2, 0.5), fontsize=14)
    
    # Adjust layout with extra space for legend
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved to: {save_path}")
    
    plt.show()
