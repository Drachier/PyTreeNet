"""
This module contains helpful tools for plotting results.
"""
from enum import Enum

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import numpy as np

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

def save_figure(fig: Figure,
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
    if filename is not None:
        fig.savefig(filename, format="pdf")
    else:
        plt.show()
    if clear_figure:
        plt.clf()

class Matriciser:
    """
    Handles result data to be turned into a matrix that can be used for
    plotting.
    """

    def __init__(self, data: list[tuple[int, int, float]] | None = None):
        """
        Initialize the Matriciser with a list of tuples.

        Each tuple should contain two integers and a float, representing
        the row index, column index, and value respectively.
        
        Args:
            data (list[tuple[int, int, float]]): The data to be converted
                into a matrix. The first integer describes the values of 
                corresponding to rows, the second integer describes the
                values corresponding to columns, and the float is the value
                at that position in the matrix.
        """
        if data is None:
            data = []
        self._data = data
        self._row_elements = []
        self._col_elements = []

    def _row_and_col_init(self) -> bool:
        """
        Determines if the row and column elements have been initialized.
        """
        return bool(self._row_elements and self._col_elements)

    def add_data_point(self, row: int, col: int, value: float):
        """
        Add a data point to the Matriciser.

        Args:
            row (int): The row index of the data point.
            col (int): The column index of the data point.
            value (float): The value at the specified row and column.
        """
        self._data.append((row, col, value))
        if self._row_and_col_init():
            if row not in self._row_elements:
                self._row_elements.append(row)
            if col not in self._col_elements:
                self._col_elements.append(col)

    def get_row_elements(self) -> list[int]:
        """
        Get the unique row elements from the data.

        Returns:
            list[int]: A sorted list of unique row elements.
        """
        if not self._row_and_col_init():
            return sorted(set(row_val for row_val, _, _ in self._data))
        return self._row_elements

    def row_map(self) -> dict[int,int]:
        """
        A mapping from row values to their indices in the matrix.
        """
        return {val: idx for idx, val in enumerate(self.get_row_elements())}

    def get_col_elements(self) -> list[int]:
        """
        Get the unique column elements from the data.

        Returns:
            list[int]: A sorted list of unique column elements.
        """
        if not self._row_and_col_init():
            return sorted(set(col_val for _, col_val, _ in self._data))
        return self._col_elements

    def col_map(self) -> dict[int,int]:
        """
        A mapping from column values to their indices in the matrix.
        
        Returns:
            dict[int,int]: A dictionary mapping column values to their indices.
        """
        return {val: idx for idx, val in enumerate(self.get_col_elements())}

    def data_min(self) -> float:
        """
        Get the minimum value from the data.

        Returns:
            float: The minimum value in the data.
        """
        return min(value for _, _, value in self._data)

    def data_max(self) -> float:
        """
        Get the maximum value from the data.

        Returns:
            float: The maximum value in the data.
        """
        return max(value for _, _, value in self._data)

    def set_y_tick_labels(self, ax: Axes):
        """
        Set the y-tick labels of the given axes to the row elements.

        Args:
            ax (Axes): The matplotlib Axes object to set the y-tick labels on.
        """
        if not ax.images:
            errstr = "No image found in the axes. Please run `ax.imshow()` first."
            raise RuntimeError(errstr)
        row_map = self.row_map()
        y_ticks = [int(round(t)) for t in ax.get_yticks()]
        y_tick_labels = [str(row_map[t]) for t in y_ticks]
        ax.set_yticklabels(y_tick_labels)

    def set_x_tick_labels(self, ax: Axes):
        """
        Set the x-tick labels of the given axes to the column elements.

        Args:
            ax (Axes): The matplotlib Axes object to set the x-tick labels on.
        """
        if not ax.images:
            errstr = "No image found in the axes. Please run `ax.imshow()` first."
            raise RuntimeError(errstr)
        col_map = self.col_map()
        x_ticks = [int(round(t)) for t in ax.get_xticks()]
        x_tick_labels = [str(col_map[t]) for t in x_ticks]
        ax.set_xticklabels(x_tick_labels)

    def to_matrix(self) -> np.ndarray:
        """
        Convert the data into a matrix.

        Returns:
            np.ndarray: A 2D numpy array representing the matrix.
        """
        row_elements = self.get_row_elements()
        col_elements = self.get_col_elements()
        matrix = np.zeros((len(row_elements), len(col_elements)))
        for row_val, col_val, value in self._data:
            row_idx = row_elements.index(row_val)
            col_idx = col_elements.index(col_val)
            matrix[row_idx, col_idx] = value
        return matrix
