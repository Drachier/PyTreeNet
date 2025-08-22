"""
This module implements a function that plots the convergence of a result
over a given parameter range.
"""
from __future__ import annotations
from typing import Any, Self

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

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

    These are the results of multiple runs of a model, where each run
    corresponds to a different value of a convergence parameter.

    Attributes:
        values (list[npt.NDArray[np.float64]]): List of arrays containing
            convergence results. Each array corresponds to a different
            value of the convergence parameter.
        line_config (LineConfig): Configuration for the line style in the plot.
        conv_param_values (list[Any]): Values of the convergence parameter.
        conv_param (str | None): The parameter over which convergence is
            plotted. Defaults to "parameter".
        x_values (npt.NDArray[np.float64] | None): X values for the plot.
            If None, it will be the convergence parameter values.
            Defaults to None.
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
                "Length of x_values does not match length of values arrays!")
        self.line_config = line_config
        self.conv_param_values = conv_param_values

    @classmethod
    def from_array_list(cls,
                        values: list[npt.NDArray[np.float64]],
                        line_config: LineConfig,
                        conv_param_values: list[Any],
                        conv_param: str | None = None,
                        x_values: npt.NDArray[np.float64] | None = None
                        ) -> Self:
        """
        Create a ConvergingResults instance from a list of arrays.

        Args:
            values (list[npt.NDArray[np.float64]]): List of arrays containing
                convergence results.
            line_config (LineConfig): Configuration for the line style in the plot.
            conv_param_values (list[Any]): Values of the convergence parameter.
            conv_param (str | None, optional): The parameter over which convergence
                is plotted. Defaults to None.
            x_values (npt.NDArray[np.float64] | None, optional): X values for the
                plot. If None, it will be the convergence parameter values.
                Defaults to None.
        
        Returns:
            ConvergingResults: An instance of ConvergingResults.
        """
        return cls(values,
                   line_config,
                   conv_param_values=conv_param_values,
                   conv_param=conv_param,
                   x_values=x_values)

    @classmethod
    def from_dict(cls,
                  data: dict[Any, npt.NDArray[np.float64]],
                  line_config: LineConfig,
                  conv_param: str | None = None,
                  x_values: npt.NDArray[np.float64] | None = None
                  ) -> Self:
        """
        Create a ConvergingResults instance from a dictionary.

        Args:
            data (dict[Any, npt.NDArray[np.float64]]): Dictionary where keys
                are convergence parameter values and values are arrays of
                convergence results.
            line_config (LineConfig): Configuration for the line style in the plot.
            conv_param (str | None, optional): The parameter over which convergence
                is plotted. Defaults to None.
            x_values (npt.NDArray[np.float64] | None, optional): X values for the
                plot. If None, it will be the convergence parameter values.
        
        Returns:
            ConvergingResults: An instance of ConvergingResults.
        """
        sorted_items = sorted(data.items())
        conv_param_values, values = zip(*sorted_items)
        return cls.from_array_list(
            list(values),
            line_config,
            conv_param_values=list(conv_param_values),
            conv_param=conv_param,
            x_values=x_values
        )
    
    def limx(self) -> tuple[float, float]:
        """
        Get the x-axis limits for the plot.

        Returns:
            tuple[float, float]: The minimum and maximum x values.
        """
        return (np.min(self.x_values), np.max(self.x_values))

    def compute_alphas(self) -> list[float]:
        """
        Compute alpha values for the plot based on the number of convergence
        parameter values.

        Returns:
            list[float]: List of alpha values for the plot.
        """
        return compute_alphas(len(self.conv_param_values))

    def plot_one_on_axis(self,
                         index: int,
                         alpha: float = 1.0,
                         ax: Axes | None = None):
        """
        Plot one set of convergence results on the given axes.

        Args:
            index (int): Index of the result to plot.
            alpha (float, optional): Alpha value for the line.
                Defaults to 1.0.
            ax (Axes | None, optional): The matplotlib Axes object to plot on.
                If None, the current axes will be used. Defaults to None.
        """
        if ax is None:
            ax = plt.gca()
        ax.plot(self.x_values,
                self.values[index],
                alpha=alpha,
                **self.line_config.to_kwargs(exclude={'label'}))

    def plot_on_axis(self,
                     ax: Axes | None = None):
        """
        Plot all convergence results on the given axes.

        Args:
            ax (Axes | None): The matplotlib Axes object to plot on.
                If None, the current axes will be used. Defaults to None.
        """
        if ax is None:
            ax = plt.gca()
        alphas = self.compute_alphas()
        for index, alpha in enumerate(alphas):
            self.plot_one_on_axis(index=index, alpha=alpha, ax=ax)
        self.line_config.plot_legend(ax=ax)

class ReferenceResults:
    """
    A class to hold reference results for plotting.

    This is used to plot exact results in a convergence plot.

    Attributes:
        x (npt.NDArray[np.float64]): X values for the reference results.
        y (npt.NDArray[np.float64]): Y values for the reference results.
        line_config (LineConfig): Configuration for the line style in the plot.
    """

    def __init__(self,
                 x: npt.NDArray[np.float64],
                 y: npt.NDArray[np.float64],
                 line_config: LineConfig):
        """
        Initialize a ReferenceResults instance.

        Args:
            x (npt.NDArray[np.float64]): X values for the reference results.
            y (npt.NDArray[np.float64]): Y values for the reference results.
            line_config (LineConfig): Configuration for the line style in the plot.
        """
        self.x = x
        self.y = y
        self.line_config = line_config

    def limx(self) -> tuple[float, float]:
        """
        Get the x-axis limits for the reference results.

        Returns:
            tuple[float, float]: The minimum and maximum x values.
        """
        if len(self.x) == 0:
            raise ValueError("No x values provided for reference results!")
        return (np.min(self.x), np.max(self.x))

    def plot_on_axis(self,
                     ax: Axes | None = None):
        """
        Plot the reference results on the given axes.

        Args:
            ax (Axes | None): The matplotlib Axes object to plot on.
                If None, the current axes will be used.
        """
        if ax is None:
            ax = plt.gca()
        ax.plot(self.x, self.y, **self.line_config.to_kwargs())

def plot_convergence(results: list[ConvergingResults],
                     axis_config: AxisConfig,
                     exact_results: ReferenceResults | None = None,
                     style: DocumentStyle = DocumentStyle.THESIS,
                     save_path: str | None = None
                     ) -> None:
    """
    Plots the convergence of results over a given parameter range.

    Args:
        results (list[ConvergingResults]): A list of ConvergingResults objects
            containing the data to plot.
        axis_config (AxisConfig): Configuration for the axes of the plot.
        exact_results (ReferenceResults | None, optional): Reference results to plot.
        style (DocumentStyle, optional): The style to use for the plot.
            Defaults to DocumentStyle.THESIS.
        save_path (str | None, optional): Path to save the plot. If None,
            the plot will not be saved but shown instead. Defaults to None.
    """
    config_matplotlib_to_latex(style=style)
    size = set_size(style)
    fig, ax = plt.subplots(figsize=size)
    if exact_results is not None:
        exact_results.plot_on_axis(ax=ax)
    for result in results:
        result.plot_on_axis(ax=ax)
    axis_config.apply_to_axis(ax=ax)
    xlims = [result.limx() for result in results]
    try:
        xlims.append(exact_results.limx())
    except ValueError:
        pass
    ax.set_xlim(np.min([xlim[0] for xlim in xlims]),
                np.max([xlim[1] for xlim in xlims]))
    save_figure(fig,
                filename=save_path)
