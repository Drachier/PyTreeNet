"""
This module implements a supervisor class that can run multiple instances of experiments.
"""
from __future__ import annotations
import subprocess
import argparse
import os
from typing import Any, Self
from enum import Enum
import re
import json
import logging
import sys

from .sim_params import SimulationParameters
from .metadata_file import METADATAFILE_STANDARD_NAME
from .status_enum import Status

LOGFILE_STANDARD_NAME = "log.txt"
CURRENTPARAMFILE_STANDARD_NAME = "current_parameters.json"
SIMSCRIPT_STANDARD_NAME = "sim_script.py"

class Skipper(Enum):
    """
    Enumeration for the skipping status of a simulation.

    This is used to determine if a simulation should be skipped based on
    the existing metadata.

    Attributes:
        NONE: No skipping, all simulations will be run.
        EXISTING: Skips simulations that have already been run and exists in
            the metadata. It does not matter if it failed or not.
        SUCCESS: Skips simulations that have been successfully run.
        TIMEOUT: Skips simulations that have timed out and those that were
            successfully run.
    """
    NONE = "none"
    EXISTING = "existing"
    SUCCESS = "success"
    TIMEOUT = "timeout"

class Supervisor:
    """
    Supervisor class to run a given script from the terminal with ease.

    Attributes:
        sim_parameters (list[SimulationParameters]): A list of simulation
            parameters to run.
        save_directory (str): The directory where the results will be saved.
        simulation_script_path (str): The path to the script that will be
            executed for each simulation.
        skip_existing (bool): If True, existing results will be skipped.
        timeout (int): The timeout for each simulation in seconds.
        logfile_name (str): The filename of the logfile.
        metadatafile_name (str): The filename of the metadata `json` file.
    """

    def __init__(self,
                 sim_parameters: list[SimulationParameters] | SimulationParameters,
                 save_directory: str,
                 simulation_script_path: str,
                 skip_existing: Skipper = Skipper.NONE,
                 timeout: int = 3600,
                 python_executable: str = "python",
                 logfile_name: str = LOGFILE_STANDARD_NAME,
                 metadatafile_name: str = METADATAFILE_STANDARD_NAME
                 ) -> None:
        """
        Initialises a Supervisor object.

        Args:
            sim_parameters (list[SimulationParameters] | SimulationParameters):
                A list of simulation parameters or a single simulation
                parameter object.
            save_directory (str): The directory where the results will be
                saved.
            simulation_script_path (str): The path to the script that will
                be executed for each simulation.
            skip_existing (Skipper): The skipping strategy to use for existing
                results. Defaults to `Skipper.EXISTING`.
            timeout (int): The timeout for each simulation in seconds.
                Defaults to `3600` seconds (1 hour).
            python_executable (str): The Python executable to use for running
                the simulation script. Defaults to `"python"`.
            logfile_name (str): The filename of the logfile.
                Defaults to `"log.txt"`.
            metadatafile_name (str): The filename of the metadata
                `json` file. Defaults to `"metadata.json"`.
        """
        if not isinstance(sim_parameters, list):
            sim_parameters = [sim_parameters]
        self.sim_parameters = sim_parameters
        prepare_save_directory(save_directory)
        self.save_directory = save_directory
        check_script_path(simulation_script_path)
        self.simulation_script_path = simulation_script_path
        self.skip_existing = skip_existing
        self.timeout = timeout
        self.logfile_name = handle_file_name(logfile_name, ".txt")
        self.metadatafile_name = handle_file_name(metadatafile_name, ".json")
        self.python_executable = python_executable

    @classmethod
    def from_commandline(cls,
                         parameter_set: list[SimulationParameters] | SimulationParameters,
                         simulation_script_path: str
                         ) -> Self:
        """
        Creates a Supervisor object from the command line arguments.

        Args:
            parameter_set (list[SimulationParameters] | SimulationParameters):
                A list of simulation parameters or a single simulation
                parameter object.
            simulation_script_path (str): The path to the script that will
                be executed for each simulation.
        
        Returns:
            Supervisor: An instance of the Supervisor class.
        """
        descr = "Run simulations with given parameters."
        parser = argparse.ArgumentParser(description=descr)
        parser.add_argument("save_directory",
                            type=str,
                            help="Directory to save results.")
        parser.add_argument("--skip-existing",
                            action="store_true",
                            default=False,
                            help="Skip existing results.")
        parser.add_argument("--skip-success",
                            action="store_true",
                            default=False,
                            help="Skip successful results.")
        parser.add_argument("--skip-timeout",
                            action="store_true",
                            default=False,
                            help="Skip timed out results.")
        python_executable = sys.executable
        args = parser.parse_args()
        if args.skip_timeout:
            skip_existing = Skipper.TIMEOUT
        elif args.skip_success:
            skip_existing = Skipper.SUCCESS
        elif args.skip_existing:
            skip_existing = Skipper.EXISTING
        else:
            skip_existing = Skipper.NONE
        return cls(parameter_set,
                   args.save_directory,
                   simulation_script_path,
                   skip_existing=skip_existing,
                   python_executable=python_executable)

    def create_metadata_file_path(self) -> str:
        """
        Returns the path to the metadata file.

        Returns:
            str: The path to the metadata file.
        """
        return os.path.join(self.save_directory,
                            self.metadatafile_name)

    def load_metadata_index(self) -> dict[str, Any]:
        """
        Loads the metadata from the metadata file.

        Returns:
            dict[str, Any]: The metadata dictionary.
        """
        metadata_file_path = self.create_metadata_file_path()
        if not os.path.exists(metadata_file_path):
            return {}
        with open(metadata_file_path, 'r') as f:
            return json.load(f)

    def configure_logging(self) -> None:
        """
        Configures the logging for the supervisor.
        """
        log_path = os.path.join(self.save_directory,
                                self.logfile_name)
        logging.basicConfig(filename=log_path,
                            level=logging.INFO)

    def save_metadata_index(self,
                            metadata_index: dict[str, Any]
                            ) -> None:
        """
        Saves the metadata index to the metadata file.

        Args:
            metadata_index (dict[str, Any]): The metadata index to save.
        """
        metadata_file_path = self.create_metadata_file_path()
        with open(metadata_file_path, 'w') as file:
            json.dump(metadata_index, file, indent=4)

    def run_simulations(self) -> None:
        """
        Runs the simulations with the given parameters.

        This will create file with the current parameters for each simulation
        and log the output of the simulation runs.
        """
        self.configure_logging()
        metadata_index = self.load_metadata_index()
        num_sims = len(self.sim_parameters)
        logging.info("Starting simulations with %d parameter sets",
                     num_sims)
        for i, sim_param in enumerate(self.sim_parameters):
            logging.info("### Running simulation %d/%d", i + 1, num_sims)
            runner = SingleParameterRunner(sim_param,
                                           self.save_directory,
                                           self.simulation_script_path,
                                           timeout=self.timeout,
                                           skip_existing=self.skip_existing,
                                           python_executable=self.python_executable)
            runner.run(metadata_index)
        self.save_metadata_index(metadata_index)   

class SingleParameterRunner:
    """
    A class to run a single set of simulation parameters.
    
    This will also handle the logging and metadata management for this one
    simulation.

    Attributes:
        sim_parameters (SimulationParameters): The simulation parameters
            to run.
        save_directory (str): The directory where the results will be saved.
        simulation_script_path (str): The path to the script that will be
            executed for the simulation.
        timeout (int): The timeout for the simulation in seconds.
        skip_existing (bool): If True, existing results will be skipped.
        currentparam_file_name (str): The filename of the current parameters
            file. Defaults to `"current_parameters.json"`.
        status (Status): The status of the simulation run.
        hash (str): A hash of the simulation parameters for easy identification.
    """

    def __init__(self,
                 sim_parameters: SimulationParameters,
                 save_directory: str,
                 simulation_script_path: str,
                 timeout: int = 3600,
                 skip_existing: Skipper = Skipper.NONE,
                 python_executable: str = "python",
                 currentparam_file_name: str = CURRENTPARAMFILE_STANDARD_NAME
                 ) -> None:
        """
        Initialises a SingleParameterRunner object.

        Args:
            sim_parameters (SimulationParameters): The simulation parameters
                to run.
            save_directory (str): The directory where the results will be
                saved.
            simulation_script_path (str): The path to the script that will
                be executed for the simulation.
            timeout (int): The timeout for the simulation in seconds.
                Defaults to `3600` seconds (1 hour).
            skip_existing (Skipper): The skipping strategy to use for existing
                results. Defaults to `Skipper.EXISTING`.
            python_executable (str): The Python executable to use for running
                the simulation script. Defaults to `"python"`.
            currentparam_file_name (str): The filename of the current
                parameters file. Defaults to `"current_parameters.json"`.
                From this the simulation script can read the parameters
                to run.
        """
        self.sim_parameters = sim_parameters
        self.save_directory = save_directory
        self.simulation_script_path = simulation_script_path
        self.timeout = timeout
        self.skip_existing = skip_existing
        self.hash = sim_parameters.get_hash()
        self.python_executable = python_executable
        self.currentparam_file_name = handle_file_name(currentparam_file_name,
                                                       ".json")
        self.status = Status.UNKNOWN

    def skip(self,
             metadata_index: dict[str, dict]
             ) -> bool:
        """
        Checks if this simulation should be skipped.
        
        This would happen if the simulation has already been run and
        `skip_existing` is set to `True`.
        """
        if self.skip_existing == Skipper.EXISTING:
            return self.hash in metadata_index
        elif self.skip_existing == Skipper.SUCCESS:
            return self.hash in metadata_index and metadata_index[self.hash]["status"] == Status.SUCCESS.value
        elif self.skip_existing == Skipper.TIMEOUT:
            return self.hash in metadata_index and (metadata_index[self.hash]["status"] == Status.SUCCESS.value or
                                                    metadata_index[self.hash]["status"] == Status.TIMEOUT.value)
        return False

    def current_parameters_file_path(self) -> str:
        """
        Returns the path to the current parameters file.

        Returns:
            str: The path to the current parameters file.
        """
        return os.path.join(self.save_directory,
                            self.currentparam_file_name)

    def save_current_parameters(self) -> None:
        """
        Saves the current simulation parameters to a json file.
        """
        current_parameters_path = self.current_parameters_file_path()
        with open(current_parameters_path, "w") as f:
            dictionary = self.sim_parameters.to_json_dict()
            json.dump(dictionary, f, indent=4)

    def log_output(self, result: subprocess.CompletedProcess) -> None:
        """
        Loggs the output of the simulation run to the log file.

        Args:
            result (subprocess.CompletedProcess): The result of the
                simulation run.
        """
        logging.info("Return code: %s", self.status.value)
        if result.stdout:
            logging.info("STDOUT:\n%s", result.stdout)
        if result.stderr:
            logging.error("STDERR:\n%s", result.stderr)

    def execute_simulation(self) -> Status:
        """
        Executes the simulation with the given parameters.

        Returns:
            Status: The status of the simulation run.
        """
        try:
            result = subprocess.run(
                [self.python_executable, self.simulation_script_path, self.save_directory],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=self.timeout
            )
            if result.returncode == 0:
                self.status = Status.SUCCESS
            else:
                self.status = Status.FAILED
            self.log_output(result)
        except subprocess.TimeoutExpired:
            self.status = Status.TIMEOUT
        return self.status

    def update_metadata_index(self,
                              metadata_index: dict[str, dict]
                             ) -> None:
        """
        Updates the metadata index with the current simulation parameters
        and status.

        Args:
            metadata_index (dict[str, dict]): The metadata index to update.
        """
        dictionary = self.sim_parameters.to_json_dict()
        dictionary["status"] = self.status.value
        metadata_index[self.hash] = dictionary

    def run(self,
            metadata_index: dict[str, dict]
            ) -> None:
        """
        Runs the simulation with the given parameters.

        Args:
            metadata_index (dict[str, dict]): The metadata index to update
                after the simulation run.
        """
        if self.skip(metadata_index):
            logging.info("### Skipping existing simulation with hash %s", self.hash)
            return
        logging.info("### Running simulation with hash %s", self.hash)
        self.save_current_parameters()
        self.status = self.execute_simulation()
        self.update_metadata_index(metadata_index)

def handle_file_name(name: str,
                     file_end: str
                     ) -> str:
    """
    Deals with the supplied log file name.
    """
    if re.match(r".+" +  file_end + r"$", name):
        return name
    return name + file_end

def prepare_save_directory(save_directory: str
                           ) -> None:
    """
    Prepares the save directory by creating it if it does not exist.

    Args:
        save_directory (str): The directory where the results will be saved.
    """
    os.makedirs(save_directory, exist_ok=True)

def check_script_path(simulation_script_path: str
                      ) -> None:
    """
    Checks if the simulation script path is valid.
    
    Raises:
        ValueError: If the script path does not end with '.py'.
        FileNotFoundError: If the script does not exist.
    """
    if not simulation_script_path.endswith('.py'):
        errstr = f"{simulation_script_path} does not end with '.py'!"
        raise ValueError(errstr)
    if not os.path.exists(simulation_script_path):
        errstr = f"{simulation_script_path} does not exist!"
        raise FileNotFoundError(errstr)
