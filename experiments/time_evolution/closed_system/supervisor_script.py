"""
This script is used to superwise the time evolution experiments in PyTreeNet.
"""
import sys
import os
import json
import subprocess
import logging
from enum import Enum

from sim_script import (get_param_hash,
                         CURRENT_PARAM_FILENAME)
from parameters import generate_parameter_set

LOG_FILE_NAME = "log.txt"

class Status(Enum):
    """
    Enumeration for the status of a simulation.
    """
    SUCCESS = "success"
    FAILED = "failed"
    RUNNING = "running"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

def params_to_dict(params):
    sim_params = params[0]
    time_evo_params = params[1]
    dictionary = sim_params.to_dict()
    dictionary.update(time_evo_params.to_dict())
    return dictionary

if len(sys.argv) < 2:
    print("Usage: python supervisor_script.py <save_directory> [--skip-existing]")
    sys.exit(1)

save_directory = sys.argv[1]
os.makedirs(save_directory, exist_ok=True)
SKIP_EXISTING = "--skip-existing" in sys.argv

INDEX_FILE = os.path.join(save_directory, "metadata.json")
if SKIP_EXISTING and os.path.exists(INDEX_FILE):
    with open(INDEX_FILE, "r") as f:
        param_index = json.load(f)
else:
    param_index = {}

# Configure logging
log_path = os.path.join(save_directory, LOG_FILE_NAME)
logging.basicConfig(
    filename=log_path,
    level=logging.INFO
)

# You will want to adjust parameters in the parameters.py file
PARAMETER_SET = generate_parameter_set()
for i, parameters in enumerate(PARAMETER_SET):

    PARAM_HASH = get_param_hash(parameters[0], parameters[1])
    if SKIP_EXISTING and PARAM_HASH in param_index:
        logging.info("### Skipping existing simulation for parameters %d", i)
        continue

    logging.info("### Running simulation %d with hash %s", i, PARAM_HASH)

    with open(os.path.join(save_directory, CURRENT_PARAM_FILENAME), "w") as f:
        dictionary = params_to_dict(parameters)
        json.dump(dictionary, f, indent=4)

    status = Status.UNKNOWN
    try:
        result = subprocess.run(
            ["python", "experiments/time_evolution/sim_script.py", save_directory],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=3600  # 1 hour timeout
        )
        if result.returncode == 0:
            status = Status.SUCCESS
        else:
            status = Status.FAILED
    except subprocess.TimeoutExpired:
        status = Status.TIMEOUT

    logging.info("Return code: %s", status.value)
    if result.stdout:
        logging.info("STDOUT:\n%s", result.stdout)
    if result.stderr:
        logging.error("STDERR:\n%s", result.stderr)

    # Update the index file with the status
    dictionary = params_to_dict(parameters)
    dictionary["status"] = status.value
    param_index[PARAM_HASH] = dictionary
    with open(INDEX_FILE, "w") as f:
        json.dump(param_index, f, indent=4)
