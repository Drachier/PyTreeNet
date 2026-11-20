"""
Implements the simulation script for the operator power application experiment.

This script applies powers of an operator (Ising model by default) to a state
using different TTNO-TTNS application algorithms.
"""
from __future__ import annotations
import os
import sys

# Make the shared code in the parent folder importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytreenet.util.experiment_util.script_util import script_main

from application_common import (PowerApplicationParams,
                                run_and_save)

if __name__ == "__main__":
    script_main(run_and_save,
                PowerApplicationParams)
