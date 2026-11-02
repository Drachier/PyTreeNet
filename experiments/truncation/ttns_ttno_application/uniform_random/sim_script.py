"""
Implements the simulation script for the truncation comparison of different
truncation methods for TTNS with uniformly distributed random elements.
"""
from __future__ import annotations
from dataclasses import dataclass
import os
import sys

# Make the shared code in the parent folder importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytreenet.util.experiment_util.script_util import script_main
from pytreenet.random.random_matrices import RandomDistribution

from application_common import (ApplicationParams,
                                run_and_save)

@dataclass
class UniformApplicationParams(ApplicationParams):
    """
    Parameters for the TTNO-TTNS application simulation with uniformly
    distributed random elements.
    """
    distr_low: float = -1.0
    distr_high: float = 1.0

    def random_generation_kwargs(self) -> dict:
        """
        Returns the keyword arguments for the random TTNS/TTNO generation.
        The elements are drawn from a uniform distribution on the interval
        [distr_low, distr_high].

        Returns:
            dict: Keyword arguments passed on to `crandn`.
        """
        return {"seed": self.seed,
                "distribution": RandomDistribution.UNIFORM,
                "low": self.distr_low,
                "high": self.distr_high}

if __name__ == "__main__":
    script_main(run_and_save,
                UniformApplicationParams)
