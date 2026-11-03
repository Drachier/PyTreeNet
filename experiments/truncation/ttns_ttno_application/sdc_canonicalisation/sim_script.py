"""
Implements the simulation script for the SDC canonicalisation comparison
for MPS. The random element distribution is switched via the `distribution`
field of the simulation parameters.
"""
from __future__ import annotations
import os
import sys

# Make the shared code in the parent folder importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pytreenet.util.experiment_util.script_util import script_main

from canonicalisation_common import (SDCCanonicalisationParams,
                                     run_and_save)

if __name__ == "__main__":
    script_main(run_and_save,
                SDCCanonicalisationParams)
