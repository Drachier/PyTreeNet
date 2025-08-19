import hashlib
import json
import logging
import os
import subprocess
from enum import Enum
from time import time
from typing import Dict, List, Tuple, Any
import h5py
import pandas as pd
from numpy import array, savez, load, ndarray
from IPython.display import display, clear_output

from experiments.time_evolution.open_system_bug.exact_time_evolution.TFI_1D_Qutip import TFI_1D_Qutip
from experiments.time_evolution.open_system_bug.sim_script import (CURRENT_PARAM_FILENAME,
                                                                   INITIAL_STATE_ZERO,
                                                                   get_param_hash)
from pytreenet.special_ttn.binary import PHYS_PREFIX
from pytreenet.time_evolution.results import Results


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
            print("⚠️  Corrupted metadata file found..")
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
    # Apply SPBUG-specific parameter modifications to match actual simulation
    actual_time_step = time_evo_params.time_step_size
    actual_eval_time = time_evo_params.evaluation_time
    
    # Check if we're dealing with SPBUG algorithm
    from pytreenet.time_evolution.time_evo_enum import TimeEvoAlg
    if hasattr(time_evo_params, 'time_evo_algorithm') and time_evo_params.time_evo_algorithm == TimeEvoAlg.SPBUG:
        # Apply SPBUG modifications: double time step, halve evaluation time
        actual_time_step = time_evo_params.time_step_size * 2
        actual_eval_time = int(time_evo_params.evaluation_time / 2)
    
    # Build hash from key sim and time-evo fields including time discretization
    # Use the actual parameters that the algorithm will use
    phys_hash_dict = {"num_sites": sim_params.num_sites,
                    "coupling": sim_params.coupling,
                    "ext_magn": sim_params.ext_magn,
                    "relaxation_rate": sim_params.relaxation_rate,
                    "dephasing_rate": sim_params.dephasing_rate,
                    "final_time": time_evo_params.final_time,
                    "time_step_size": actual_time_step,
                    "evaluation_time": actual_eval_time}

    phys_hash = hashlib.md5(json.dumps(phys_hash_dict, sort_keys=True).encode()).hexdigest()
    exact_file = os.path.join(benchmark_dir, f"{phys_hash}_exact.npz")
    # load if exists
    if os.path.exists(exact_file):
        data = load(exact_file)
        logging.info(" Loaded exact solution from cache: %s", exact_file)
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
    result, _ = tfi.evolve_system_lindblad(
        initial_state=initial_state,
        final_time=time_evo_params.final_time,
        time_step_size=actual_time_step,
        evaluation_time=actual_eval_time,
        jump_operators=jump_ops,
        e_ops=e_ops,
        options={'method':'bdf',
                 'atol':1e-6,
                 'rtol':1e-6,
                 'nsteps':1000})

    exact_magns = array(result.expect[:-1])
    exact_energy = array(result.expect[-1])

    # Fix shape mismatch: flatten magnetization if needed
    if exact_magns.ndim > 1:
        exact_magns = exact_magns.flatten()

    # Save the computed exact solution to cache
    savez(exact_file, magns=exact_magns, energy=exact_energy)
    logging.info(" Saved exact solution to cache: %s", exact_file)

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
        existing_status = existing_results[param_hash].get('status', '')

        if ask_overwrite:
            choice = input("Choose: [s]kip, [o]verwrite, or [q]uit: ").lower().strip()
            if choice == 'q':
                return Status.SKIPPED, existing_elapsed_time
            elif choice == 's' or choice == '':
                return Status.SKIPPED, existing_elapsed_time
            elif choice != 'o':
                return Status.SKIPPED, existing_elapsed_time
        else:
            # Only skip if the existing simulation was successful
            if existing_status == Status.SUCCESS.value:
                return Status.SKIPPED, existing_elapsed_time
            # If it failed or timed out, re-run it

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
            errors='replace',
            check=False,
            timeout=5400)  # 1.5 hour timeout

        # Log subprocess output regardless of success/failure
        if result.stdout.strip():
            logging.info("Subprocess stdout for %s:\n%s", param_hash[:12], result.stdout)
        if result.stderr.strip():
            logging.warning("Subprocess stderr for %s:\n%s", param_hash[:12], result.stderr)

        if result.returncode == 0:
            status = Status.SUCCESS
        else:
            status = Status.FAILED
            logging.error("Simulation %s failed with return code %d", param_hash[:12], result.returncode)

    except subprocess.TimeoutExpired:
        status = Status.TIMEOUT
        logging.error("Simulation %s timed out after 1 hour", param_hash[:12])

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
        print(f"Shapes - Bug: {bug_magns.shape}, Exact: {exact_magns.shape}")
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

def run_benchmarks(parameters,
                   benchmark_dir: str,
                   ask_overwrite: bool = False):
    """
    Run benchmarks for a specific TTN structure with different time step levels.

    Args:
        parameters: Simulation parameters to use for the benchmarks
        benchmark_dir: Unified directory for all benchmark results
        ask_overwrite: Whether to ask for skip/overwrite if the parameters already exist.

    Returns:
        dict: Summary of results with DataFrame
    """
    # Setup benchmark directory
    existing_results, metadata_file = setup_benchmark_directory(benchmark_dir)

    # Create DataFrame for live progress tracking
    columns = ['Parameters Hash',
               'Algorithm',
               'TTNDO structure',
               'Time Step',
               'Max Bond Dim',
               'Max Product Dim',
               'Status',
               'CPU_time']
    df = pd.DataFrame(columns = columns)
    display(df)

    # Run benchmarks
    results_summary = {'successful': 0,
                        'failed': 0,
                        'timeout': 0,
                        'skipped': 0,
                        'dataframe': df}

    for _, parameters in enumerate(parameters, 1):
        sim_params, time_evo_params = parameters
        param_hash = get_param_hash(sim_params, time_evo_params)

        # Prepare row data before running benchmark
        algorithm = time_evo_params.time_evo_algorithm.value if time_evo_params.time_evo_algorithm else 'Unknown'
        method = time_evo_params.time_evo_mode.method.value if hasattr(time_evo_params, 'time_evo_mode') and time_evo_params.time_evo_mode else 'Unknown'
        
        # Get TTNDO structure from sim_params
        ttndo_structure = getattr(sim_params, 'ttns_structure', 'Unknown')
        if hasattr(ttndo_structure, 'value'):
            ttndo_structure = ttndo_structure.value

        row_data = {'Parameters Hash': param_hash[:12],
                    'Algorithm': f"{algorithm} / {method}",
                    'TTNDO structure': ttndo_structure,
                    'Time Step': f"{time_evo_params.time_step_size:.3f}",
                    'Max Bond Dim': time_evo_params.max_bond_dim,
                    'Max Product Dim': time_evo_params.max_product_dim,
                    'Status': 'Running...',
                    'CPU_time': 0.0}

        # Add row to DataFrame
        df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)

        # Update display
        clear_output(wait=True)
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
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
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
            for key, value in f.items():
                if isinstance(value, h5py.Dataset):
                    results.results[key] = value[:]

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

            # Check if the objects are datasets and convert them to arrays
            mag_error_obj = f['mag_error']
            energy_error_obj = f['energy_error']

            if not isinstance(mag_error_obj, h5py.Dataset) or not isinstance(energy_error_obj, h5py.Dataset):
                raise ValueError(f"mag_error or energy_error is not a dataset in {h5_file}")

            # Convert to numpy arrays
            mag_error = array(mag_error_obj)
            energy_error = array(energy_error_obj)

            return mag_error, energy_error

    def load_combined(self, param_hash: str) -> Tuple[Dict, Any]:
        """Load both parameters and results for a given hash."""
        return self.load_parameters(param_hash), self.load_results(param_hash)

    def filter_and_load_results(self, filter_params: Dict) -> List[Tuple[Dict, Any]]:
        """
        Filter results based on parameter criteria and return all matching (params, results) pairs.
        
        Args:
            filter_params: Dictionary containing parameter criteria to filter by.
                          e.g. {"status": "success", "time_step_level": "coarse"}
        
        Returns:
            List[Tuple[Dict, Any]]: List of (parameters, results) 
            pairs that match the filter criteria
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
                    results = self.load_results(param_hash)
                    matching_results.append((params, results))

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

def load_specific_result(param_hash: str, results_dir) -> Tuple[Dict, Any]:
    """Load a specific result by parameter hash."""
    loader = BenchmarkResultLoader(results_dir)
    return loader.load_combined(param_hash)

def load_filtered_results(filter_params: Dict, results_dir: str) -> List[Tuple[Dict, Any]]:
    """
    Load all results that match the specified parameter criteria.

    Args:
        filter_params: Dictionary containing parameter criteria to filter by.
                       retutns all results if None.
        results_dir: Directory containing benchmark results

    Returns:
        List[Tuple[Dict, Any]]: List of (parameters, results) pairs that match the filter criteria
    """
    if filter_params is None:
        filter_params = {}

    loader = BenchmarkResultLoader(results_dir)
    return loader.filter_and_load_results(filter_params)
