"""
Plots the results from the SDC canonicalisation comparison experiments for
both the uniform and the Gaussian random elements.
"""
import os
import sys

# Make the shared code in the parent folder importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot_common import standard_plot_main

from canonicalisation_common import SDCCanonicalisationParams

if __name__ == "__main__":
    standard_plot_main(SDCCanonicalisationParams)
