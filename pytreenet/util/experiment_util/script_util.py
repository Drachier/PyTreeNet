"""
This module implements utilities for the script file used in pytreenet experiments.
"""
import sys
import os
import traceback
from time import time
from typing import Callable

from .supervisor import CURRENTPARAMFILE_STANDARD_NAME
from .sim_params import SimulationParameters

def script_main(runner_function: Callable,
                parameter_class: type[SimulationParameters],
                current_param_name: str = CURRENTPARAMFILE_STANDARD_NAME):
    """
    Main function to run the script with the given parameter file name.
    """
    if len(sys.argv) < 2:
        print("Usage: python sim_script.py <save_directory>")
        sys.exit(1)
    try:
        save_directory = sys.argv[1]
        param_path = os.path.join(save_directory, current_param_name)
        sim_params = parameter_class.load_from_json(param_path)
        start = time()
        runtime = runner_function(sim_params, save_directory)
        end = time()
        if runtime is None:
            runtime = end - start
            print(f"Total runtime: {runtime:.2f} seconds")
        else:
            print(f"Simulation completed in {runtime:.2f} seconds.")
        sys.exit(0)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
