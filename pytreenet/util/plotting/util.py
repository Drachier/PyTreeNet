"""
Plotting utilities for PyTreeNet that do not fit elsewhere.
"""
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def compute_alphas(n: int,
                   min_alpha: float = 0.2,
                   max_alpha: float = 1.0
                   ) -> list[float]:
    """
    Compute a list of alpha values for plotting.

    These alpha values represent the transparency of the lines in a plot.

    Args:
        n (int): Number of alpha values to compute.
        min_alpha (float, optional): Minimum alpha value. Defaults to 0.2.
        max_alpha (float, optional): Maximum alpha value. Defaults to 1.0.
    
    Returns:
        list[float]: List of alpha values.
    """
    if n <= 1:
        return [max_alpha]
    alphas = np.linspace(min_alpha, max_alpha, n)
    return alphas.tolist()

def save_figure(fig: Figure | None,
                filename: str | None = None,
                clear_figure: bool = True):
    """
    Save the figure to a file.

    Args:
        fig (Figure): The matplotlib figure to save.
        filename (str | None, optional): The filename to save the figure as.
            If None, the figure will not be saved, but shown instead.
            Defaults to None.
        clear_figure (bool, optional): Whether to clear the figure after
            saving. Defaults to True.
    """
    if fig is None:
        return
    if filename is not None:
        fileend = re.search(r"\.(\w+)$", filename)
        if fileend is None:
            raise ValueError("Filename must have an extension.")
        fig.savefig(filename, format=fileend.group(1))
    else:
        plt.show()
    if clear_figure:
        plt.clf()
