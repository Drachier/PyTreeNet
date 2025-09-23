"""
This script is used to supervise the truncation comparison experiments.
"""
import os

from pytreenet.util.experiment_util.supervisor import (Supervisor,
                                                       SIMSCRIPT_STANDARD_NAME)
from pytreenet.core.truncation import TruncationMethod
from pytreenet.special_ttn.special_states import TTNStructure

from sim_script import TruncationParams

def generate_parameter_set() -> list[TruncationParams]:
    """
    Generates a set of parameters for time evolution experiments.
    
    Returns:
        list[TruncationParams]: A list of tuples containing simulation and time
        evolution parameters.
    """
    return []

def main():
    """
    Main function to set up and run the supervisor for time evolution experiments.
    """
    param_set = generate_parameter_set()
    # The simulation script lies in the same directory as this script
    sim_script_path = os.path.join(os.path.dirname(__file__),
                                   SIMSCRIPT_STANDARD_NAME)
    supervisor = Supervisor.from_commandline(param_set,
                                             sim_script_path)
    supervisor.timeout = 10 * 60 * 60  # 10 hours
    supervisor.run_simulations()

if __name__ == "__main__":
    main()
