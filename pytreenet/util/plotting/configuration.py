"""
This module provides functions to configure the plotting.
"""

from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes
from matplotlib.figure import Figure

class DocumentStyle(Enum):
    """
    Enum for document styles to use with matplotlib.
    """
    THESIS = "thesis"
    ARTICLE = "article"

def set_size(width: float | str | DocumentStyle,
             fraction: float = 1,
             subplots: tuple[int, int] = (1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.

    Source: https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Args:
        width (float | str | DocumentStyle): Document width in points or a
            string describing the width.
        fraction (float, optional): Fraction of the width which you wish the
            figure to occupy
        subplots (tuple[int,int]): The number of rows and columns of subplots.
    
    Returns:
        fig_dim (tuple[float,float]): Dimensions of figure in inches,
    """
    # Width of figure (in pts)
    if isinstance(width, str):
        width = DocumentStyle(width)
    if width == DocumentStyle.THESIS:
        width_pt = 483.6969
    elif not isinstance(width, DocumentStyle):
        width_pt = width
    fig_width_pt = width_pt * fraction
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

def config_matplotlib_to_latex(style: DocumentStyle | str = DocumentStyle.THESIS):
    """
    Get the matplotlib configuration to use LaTeX for rendering text.
    """
    if isinstance(style, str):
        style = DocumentStyle(style)
    if style == DocumentStyle.THESIS:
        tex_fonts = {
            # Use Helvetica-like sans-serif font (TeX Gyre Heros ≈ Helvetica)
            "font.family": "sans-serif",
            "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial"],

            # Use 10pt font size to match the TUM default
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,

            # Match LaTeX-like appearance without using LaTeX
            "text.usetex": False,  # Optional: True if you have LaTeX + tgheros installed

            # Optional: match TUM's Helvetica color/spacing style
            "axes.titlepad": 8,
            "figure.dpi": 150,
        }
    elif style == DocumentStyle.ARTICLE:
        tex_fonts = {
            # Use LaTeX to write all text
            #"text.usetex": True,
            "font.family": "serif",
            # Use 10pt font in plots, to match 10pt font in document
            "axes.labelsize": 10,
            "font.size": 10,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8
        }
    else:
        errstr = f"Unknown style {style}!"
        raise ValueError(errstr)
    plt.rcParams.update(tex_fonts)

def figure_from_style(style: DocumentStyle | str = DocumentStyle.THESIS,
                      fraction: float = 1,
                      subplots: tuple[int, int] = (1, 1)
                      ) -> tuple[Figure, Axes]:
    """
    Create a figure and axes with the specified style and size.

    Args:
        style (DocumentStyle | str, optional): The style to use for the figure.
            Defaults to DocumentStyle.THESIS.
        fraction (float, optional): The fraction of the width to use for the
            figure. Defaults to 1.
        subplots (tuple[int, int], optional): The number of rows and columns
            of subplots. Defaults to (1, 1).

    Returns:
        tuple[Figure, Axes]: The created figure and axes.
    """
    if isinstance(style, str):
        style = DocumentStyle(style)
    config_matplotlib_to_latex(style=style)
    fig_dim = set_size(width=style,
                       fraction=fraction,
                       subplots=subplots)
    fig, ax = plt.subplots(subplots[0], subplots[1],
                           figsize=fig_dim)
    return fig, ax

def figure_double_plot(style: DocumentStyle | str = DocumentStyle.THESIS,
                      fraction: float = 1,
                      axis: Axes | None = None
                      ) -> tuple[Figure | None, Axes]:
    """
    Creates a figure and axes for two plots next to each other of the same
    size.

    Args:
        style (DocumentStyle | str, optional): The style to use for the figure.
            Defaults to DocumentStyle.THESIS.
        fraction (float, optional): The fraction of the width to use for the
            figure. Defaults to 1.
        axis (Axes | None, optional): The axes to use for the plot. If None,
            new axes will be created. Defaults to None.

    Returns:
        tuple[Figure | None, Axes]: The created figure and axes.
    """
    if axis is None:
        fig, axis = figure_from_style(style=style,
                                       fraction=fraction,
                                       subplots=(1, 2))
    else:
        fig = None
        if not isinstance(axis, np.ndarray) or axis.shape != (2,):
            errstr = "The provided axes must be a 1x2 array!"
            raise ValueError(errstr)
    return fig, axis
