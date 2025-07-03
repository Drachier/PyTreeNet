"""
This module contains helpful tools for plotting results.
"""

import matplotlib.pyplot as plt

def set_size(width: float,
             fraction: float = 1,
             subplots: tuple[int, int] = (1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.

    Source: https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Args:
        width (float): Document width in points.
        fraction (float, optional): Fraction of the width which you wish the
            figure to occupy
        subplots (tuple[int,int]): The number of rows and columns of subplots.
    
    Returns:
        fig_dim (tuple[float,float]): Dimensions of figure in inches,
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)

def config_matplotlib_to_latex():
    """
    Get the matplotlib configuration to use LaTeX for rendering text.
    """
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8
    }
    plt.rcParams.update(tex_fonts)
