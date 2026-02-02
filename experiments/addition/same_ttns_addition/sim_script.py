"""
The simulation script to add a TTNS to itself.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import time
import os
from copy import deepcopy

from h5py import File

from pytreenet.util.experiment_util.sim_params import SimulationParameters
from pytreenet.util.experiment_util.script_util import script_main
from pytreenet.core.truncation import (TruncationMethod,
                                       truncate_ttns)
from pytreenet.random.random_matrices import RandomDistribution
from pytreenet.special_ttn.special_states import TTNStructure
from pytreenet.random.random_special_ttns import random_ttns
from pytreenet.time_evolution.results import Results
from pytreenet.util.tensor_splitting import SVDParameters
