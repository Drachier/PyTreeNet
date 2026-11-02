"""
Implements the simulation script for the truncation comparison of different
truncation methods for TTNS with Gaussian random elements.
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
class GaussianApplicationParams(ApplicationParams):
    """
    Parameters for the TTNO-TTNS application simulation with Gaussian random
    elements.
    """
    distr_scale: float = 1.0

    def random_generation_kwargs(self) -> dict:
        """
        Returns the keyword arguments for the random TTNS/TTNO generation.
        The elements are drawn from a zero mean (complex) Gaussian distribution
        with standard deviation `distr_scale`.

        Returns:
            dict: Keyword arguments passed on to `crandn`.
        """
        return {"seed": self.seed,
                "distribution": RandomDistribution.NORMAL,
                "scale": self.distr_scale}

if __name__ == "__main__":
    script_main(run_and_save,
                GaussianApplicationParams)
