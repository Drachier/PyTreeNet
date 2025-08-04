"""
This module implements a function that plots the convergence of a result
over a given parameter range.
"""
from __future__ import annotations
from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

from ..line_config import LineConfig
from ..axis_config import AxisConfig
from ..configuration import (config_matplotlib_to_latex,
                             DocumentStyle,
                             set_size)
from ..util import (save_figure,
                    compute_alphas)


class ConvergingResults:
    """
    A class to hold convergence results for plotting.

    Attributes:
        values (list[npt.NDArray[np.float64]]): List of arrays containing
            convergence results.
        line_config (LineConfig): Configuration for the line style in the plot.
        conv_param (str): The parameter over which convergence is plotted.
        conv_param_values (list[Any]): Values of the convergence parameter.

    """

    def __init__(self,
                 values: list[npt.NDArray[np.float64]],
                 line_config: LineConfig,
                 conv_param_values: list[Any],
                 conv_param: str | None = None,
                 x_values: npt.NDArray[np.float64] | None = None
                 ):
        """
        Initialize a ConvergingResults instance.

        Args:
            values (list[npt.NDArray[np.float64]]): List of arrays containing
                convergence results should be ordered to match the
                corresponding values in `conv_param_values`.
            line_config (LineConfig): Configuration for the line style in the plot.
            conv_param_values (list[Any]): Values of the convergence parameter.
            conv_param (str | None, optional): The parameter over which convergence
                is plotted. Defaults to None.
            x_values (npt.NDArray[np.float64] | None, optional): X values for the
                plot. If None, it will be the convergence parameter values.
                Defaults to None.
        """
        if conv_param is None:
            conv_param = "parameter"
        if len(values) != len(conv_param_values):
            raise ValueError(
                f"Length of results and {conv_param} values does not match!")
        self.conv_param = conv_param
        self.values = values
        if x_values is None:
            x_values = np.array(conv_param_values)
        self.x_values = x_values
        if len(x_values) != len(values[0]):
            raise ValueError(
                f"Length of x_values does not match length of values arrays!")
        self.line_config = line_config
        self.conv_param_values = conv_param_values


def plot_convergence(results: list[ConvergingResults],
                     axis_config: AxisConfig,
                     exact_results: tuple[npt.NDArray[np.float64],
                                          npt.NDArray[np.float64], LineConfig] | None = None,
                     style: DocumentStyle = DocumentStyle.THESIS,
                     save_path: str | None = None
                     ) -> None:
    """
    Plots the convergence of results over a given parameter range.

    Args:
        results (list[ConvergingResults]): A list of ConvergingResults objects
            containing the data to plot.
        axis_config (AxisConfig): Configuration for the axes of the plot.
        exact_results (tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], LineConfig] | None, optional):
            A tuple containing the exact results x and y values, and a
            LineConfig for the exact results line. If None, no exact results
            will be plotted. Defaults to None.
        style (DocumentStyle, optional): The style to use for the plot.
            Defaults to DocumentStyle.THESIS.
        save_path (str | None, optional): Path to save the plot. If None,
            the plot will not be saved but shown instead. Defaults to None.
    """
    config_matplotlib_to_latex(style=style)
    size = set_size(style)
    fig, ax = plt.subplots(figsize=size)
    if exact_results is not None:
        x_exact, y_exact, exact_line_config = exact_results
        ax.plot(x_exact, y_exact,
                **exact_line_config.to_kwargs())
    for result in results:
        alpha = compute_alphas(len(result.conv_param_values))
        for j, value in enumerate(result.values):
            ax.plot(result.x_values,
                    value,
                    **result.line_config.to_kwargs(exclude={'label'}),
                    alpha=alpha[j])
        result.line_config.plot_legend(ax=ax)
    axis_config.apply_to_axis(ax=ax)
    save_figure(fig,
                filename=save_path)
