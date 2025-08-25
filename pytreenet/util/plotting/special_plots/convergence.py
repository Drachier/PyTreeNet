"""
This module implements a function that plots the convergence of a result
over a given parameter range.
"""
from __future__ import annotations
from typing import Any, Self
from copy import deepcopy
from warnings import warn
from enum import Enum

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from ..line_config import LineConfig
from ..axis_config import AxisConfig
from ..configuration import (DocumentStyle,
                             figure_from_style)
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

    def with_new_values(self,
                        new_values: list[npt.NDArray[np.float64]],
                        deep: bool = False
                        ) -> Self:
        """
        Create a new ConvergingResults instance with the same configuration
        but different values.

        Args:
            new_values (list[npt.NDArray[np.float64]]): New list of arrays
                containing convergence results.
            deep (bool, optional): Whether to deep copy the other
                attributes. Defaults to False.

        Returns:
            ConvergingResults: A new instance of ConvergingResults with the
                same configuration but different values.

        """
        if deep:
            insrt = deepcopy
        else:
            insrt = lambda x: x
        return ConvergingResults(
            values=new_values,
            line_config=insrt(self.line_config),
            conv_param_values=insrt(self.conv_param_values),
            conv_param=insrt(self.conv_param),
            x_values=insrt(self.x_values)
        )

    def get_errors(self,
                   reference: "ReferenceResults",
                   deep: bool = False
                   ) -> Self:
        """
        Compute the absolute errors of the convergence results with respect
        to a set of reference results.

        Args:
            reference (ReferenceResults): The reference results to compare
                against.
            deep (bool, optional): Whether to deep copy the other attributes of
                the new instance. Defaults to False.

        Returns:
            ConvergingResults: A new instance of ConvergingResults containing the
            absolute errors.
        """
        errors = [np.abs(result - reference.y) for result in self.values]
        return self.with_new_values(errors, deep=deep)

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

def set_x_limits(ax: Axes,
                 results: list[ConvergingResults],
                 exact_results: ReferenceResults | None = None
                 ):
    """
    Set the x-axis limits of the given axes based on the provided results
    and reference results.

    Args:
        ax (Axes): The matplotlib Axes object to set the x-axis limits on.
        results (list[ConvergingResults]): A list of ConvergingResults objects
            to consider for setting the x-axis limits.
        exact_results (ReferenceResults | None, optional): Reference results
            to consider for setting the x-axis limits. Defaults to None.
    """
    xlims = [result.limx() for result in results]
    if exact_results is not None:
        try:
            xlims.append(exact_results.limx())
        except ValueError:
            pass
    ax.set_xlim(np.min([xlim[0] for xlim in xlims]),
                np.max([xlim[1] for xlim in xlims]))

def plot_convergence(results: list[ConvergingResults],
                     axis_config: AxisConfig,
                     exact_results: ReferenceResults | None = None,
                     style: DocumentStyle = DocumentStyle.THESIS,
                     ax: Axes | None = None,
                     save_path: str | None = None
                     ) -> Axes:
    """
    Plots the convergence of results over a given parameter range.

    Args:
        results (list[ConvergingResults]): A list of ConvergingResults objects
            containing the data to plot.
        axis_config (AxisConfig): Configuration for the axes of the plot.
        exact_results (ReferenceResults | None, optional): Reference results to plot.
        style (DocumentStyle, optional): The style to use for the plot.
            Defaults to DocumentStyle.THESIS.
        ax (Axes | None, optional): The matplotlib Axes object to plot on.
            If None, a new figure and axes will be created. Defaults to None.
        save_path (str | None, optional): Path to save the plot. If None,
            the plot will not be saved but shown instead. Defaults to None.

    Returns:
        Axes: The matplotlib Axes object containing the plot.
    """
    if ax is None:
        fig, ax = figure_from_style(style=style)
    else:
        # This avoids saving the figure
        fig = None
    if exact_results is not None:
        exact_results.plot_on_axis(ax=ax)
    for result in results:
        result.plot_on_axis(ax=ax)
    axis_config.apply_to_axis(ax=ax)
    set_x_limits(ax=ax,
                 results=results,
                 exact_results=exact_results)
    save_figure(fig,
                filename=save_path)
    return ax

def plot_error_convergence(results: list[ConvergingResults],
                           axis_config: AxisConfig,
                           exact_results: ReferenceResults,
                           style: DocumentStyle = DocumentStyle.THESIS,
                           ax: Axes | None = None,
                           save_path: str | None = None
                           ) -> Axes:
    """
    Plots the convergence of absolute errors of results over a given
    parameter range with respect to exact results.

    Args:
        results (list[ConvergingResults]): A list of ConvergingResults objects
            containing the data to plot.
        axis_config (AxisConfig): Configuration for the axes of the plot.
        exact_results (ReferenceResults): Reference results to compute errors against.
        style (DocumentStyle, optional): The style to use for the plot.
            Defaults to DocumentStyle.THESIS.
        ax (Axes | None, optional): The matplotlib Axes object to plot on.
            If None, a new figure and axes will be created. Defaults to None.
        save_path (str | None, optional): Path to save the plot. If None,
            the plot will not be saved but shown instead. Defaults to None.

    Returns:
        Axes: The matplotlib Axes object containing the plot.
    """
    if ax is None:
        fig, ax = figure_from_style(style=style)
    else:
        fig = None
    for result in results:
        errors = result.get_errors(reference=exact_results)
        errors.plot_on_axis(ax=ax)
    if axis_config.logy is False:
        axis_config.logy = True
        warn("Setting logy to True for error convergence plot.")
    axis_config.apply_to_axis(ax=ax)
    set_x_limits(ax=ax,
                 results=results,
                 exact_results=exact_results)
    save_figure(fig,
                filename=save_path)
    return ax

def plot_convergence_and_error(results: list[ConvergingResults],
                                axis_config: AxisConfig,
                                exact_results: ReferenceResults | None = None,
                                style: DocumentStyle = DocumentStyle.THESIS,
                                ax: Axes | None = None,
                                save_path: str | None = None
                                ) -> Axes:
    """
    Makes two plots horizontally next to each other.
    The left plot shows the convergence results, the right plot shows
    the absolute error convergence with respect to the exact results.
    """
    axis_config = deepcopy(axis_config)
    if ax is None:
        fig, ax = figure_from_style(style=style, subplots=(1, 2))
    else:
        fig = None
        if not isinstance(ax, np.ndarray) or ax.shape != (2,):
            errstr = "The provided axes must be a 1x2 array!"
            raise ValueError(errstr)
    plot_convergence(results,
                     axis_config,
                     exact_results=exact_results,
                     style=style,
                     ax=ax[0])
    axis_config.ylabel = f"Error of {axis_config.ylabel.lower()}"
    axis_config.make_legend = False
    axis_config.logy = True
    plot_error_convergence(results,
                           axis_config,
                           exact_results,
                           style=style,
                           ax=ax[1])
    save_figure(fig,
                filename=save_path)
    return ax

class WhichConvergence(Enum):
    """
    Enum to specify which convergence plots to create.
    """
    RESULTS = "results"
    ERRORS = "errors"
    BOTH = "both"

def plot_convergence_auto(results: list[ConvergingResults],
                          axis_config: AxisConfig,
                          exact_results: ReferenceResults | None = None,
                          style: DocumentStyle = DocumentStyle.THESIS,
                          which: WhichConvergence = WhichConvergence.BOTH,
                          ax: Axes | None = None,
                          save_path: str | None = None
                          ) -> Axes:
    """
    Automatically decides whether to plot convergence results, error
    convergence, or both based on the provided arguments.

    Args:
        results (list[ConvergingResults]): A list of ConvergingResults objects
            containing the data to plot.
        axis_config (AxisConfig): Configuration for the axes of the plot.
        exact_results (ReferenceResults | None, optional): Reference results to plot.
            If None, only convergence results will be plotted. Defaults to None.
        style (DocumentStyle, optional): The style to use for the plot.
            Defaults to DocumentStyle.THESIS.
        which (which_convergence, optional): Which plots to create.
            Defaults to which_convergence.BOTH.
        ax (Axes | None, optional): The matplotlib Axes object to plot on.
            If None, a new figure and axes will be created. Defaults to None.
        save_path (str | None, optional): Path to save the plot. If None,
            the plot will not be saved but shown instead. Defaults to None.

    Returns:
        Axes: The matplotlib Axes object containing the plot.
    """
    if which is WhichConvergence.RESULTS:
        return plot_convergence(
            results,
            axis_config,
            exact_results=exact_results,
            style=style,
            ax=ax,
            save_path=save_path
        )
    if which is WhichConvergence.ERRORS:
        return plot_error_convergence(
            results,
            axis_config,
            exact_results,
            style=style,
            ax=ax,
            save_path=save_path
        )
    if which is WhichConvergence.BOTH:
        return plot_convergence_and_error(
            results,
            axis_config,
            exact_results=exact_results,
            style=style,
            ax=ax,
            save_path=save_path
        )
    raise ValueError(f"Unknown value for which: {which}!")
