"""
Plots the results from the truncation comparison experiments with uniformly
distributed random elements.
"""
import os
import sys

# Make the shared code in the parent folder importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plot_common import standard_plot_main

from sim_script import UniformApplicationParams

if __name__ == "__main__":
    standard_plot_main(UniformApplicationParams)
