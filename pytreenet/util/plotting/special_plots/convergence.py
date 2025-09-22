"""
This module implements a function that plots the convergence of a result
over a given parameter range.
"""
from __future__ import annotations
from copy import deepcopy
from enum import Enum

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from ..axis_config import AxisConfig
from ..configuration import (DocumentStyle,
                             figure_from_style,
                             figure_double_plot)
from ..util import save_figure
from ..plotables.multiplot import ConvergingPlottable
from ..plotables.standard_plottable import StandardPlottable
from ..line_config import StyleMapping

def set_x_limits(ax: Axes,
                 results: list[ConvergingPlottable],
                 exact_results: StandardPlottable | None = None
                 ) -> None:
    """
    Set the x-axis limits of the given axes based on the provided results.

    Args:
        ax (Axes): The matplotlib Axes object to set the x-axis limits on.
        results (list[ConvergingPlottable]): A list of ConvergingPlottable objects
            containing the data to consider for setting the limits.
        exact_results (StandardPlottable | None, optional): Reference results to
            consider for setting the limits. Defaults to None.
    """
    x_mins = [result.x_limits()[0] for result in results]
    x_maxs = [result.x_limits()[1] for result in results]
    if exact_results is not None:
        x_mins.append(exact_results.x_limits()[0])
        x_maxs.append(exact_results.x_limits()[1])
    ax.set_xlim(min(x_mins), max(x_maxs))

def plot_convergence(results: list[ConvergingPlottable],
                     axis_config: AxisConfig,
                     exact_results: StandardPlottable | None = None,
                     style: DocumentStyle = DocumentStyle.THESIS,
                     ax: Axes | None = None,
                     save_path: str | None = None,
                     style_mapping: StyleMapping | None = None
                     ) -> Axes:
    """
    Plots the convergence of results over a given parameter range.

    Args:
        results (list[ConvergingPlottable]): A list of ConvergingPlottable objects
            containing the data to plot.
        axis_config (AxisConfig): Configuration for the axes of the plot.
        exact_results (StandardPlottable | None, optional): Reference results to plot.
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
    if style_mapping is not None:
        style_mapping.apply_legend()
    axis_config.apply_to_axis(ax=ax)
    set_x_limits(ax=ax,
                 results=results,
                 exact_results=exact_results)
    save_figure(fig,
                filename=save_path)
    plt.tight_layout()
    return ax

def plot_error_convergence(results: list[ConvergingPlottable],
                           axis_config: AxisConfig,
                           exact_results: StandardPlottable,
                           style: DocumentStyle = DocumentStyle.THESIS,
                           ax: Axes | None = None,
                           save_path: str | None = None
                           ) -> Axes:
    """
    Plots the convergence of absolute errors of results over a given
    parameter range with respect to exact results.

    Args:
        results (list[ConvergingPlottable]): A list of ConvergingPlottable objects
            containing the data to plot.
        axis_config (AxisConfig): Configuration for the axes of the plot.
        exact_results (StandardPlottable): Reference results to compute errors against.
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
    axis_config.set_attr_true_w_warning("logy")
    axis_config.apply_to_axis(ax=ax)
    set_x_limits(ax=ax,
                 results=results,
                 exact_results=exact_results)
    save_figure(fig,
                filename=save_path)
    plt.tight_layout()
    return ax

def plot_convergence_and_error(results: list[ConvergingPlottable],
                                axis_config: AxisConfig,
                                exact_results: StandardPlottable | None = None,
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
    fig, ax = figure_double_plot(style=style,
                                 axis=ax)
    plot_convergence(results,
                     axis_config,
                     exact_results=exact_results,
                     style=style,
                     ax=ax[0])
    if axis_config.ylabel is not None:
        axis_config.ylabel = f"Error of {axis_config.ylabel.lower()}"
    else:
        axis_config.ylabel = "Error"
    axis_config.make_legend = False
    axis_config.logy = True
    assert exact_results is not None, "Exact results must be provided to plot error convergence!"
    plot_error_convergence(results,
                           axis_config,
                           exact_results,
                           style=style,
                           ax=ax[1])
    save_figure(fig,
                filename=save_path)
    return ax

def plot_convergence_measure(results: list[ConvergingPlottable],
                             axis_config: AxisConfig,
                             style: DocumentStyle = DocumentStyle.THESIS,
                             ax: Axes | None = None,
                             save_path: str | None = None
                             ) -> Axes:
    """
    Plots the convergence of the convergence measure.

    Args:
        results (list[ConvergingPlottable]): A list of ConvergingPlottable objects
            containing the data to plot.
        axis_config (AxisConfig): Configuration for the axes of the plot.
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
        conv_measure = result.get_convergence()
        conv_measure.plot_on_axis(ax=ax)
    axis_config.set_attr_true_w_warning("logy")
    axis_config.apply_to_axis(ax=ax)
    set_x_limits(ax=ax,
                 results=results)
    save_figure(fig,
                filename=save_path)
    return ax

def plot_convergence_and_convmeasure(results: list[ConvergingPlottable],
                                        axis_config: AxisConfig,
                                        style: DocumentStyle = DocumentStyle.THESIS,
                                        ax: Axes | None = None,
                                        save_path: str | None = None
                                        ) -> Axes:
    """
    Makes two plots horizontally next to each other.

    The left plot shows the convergence results, the right plot shows
    the measure of convergence.

    Args:
        results (list[ConvergingPlottable]): The list of ConvergingPlottable objects
            containing the data to plot.
        axis_config (AxisConfig): Configuration for the axes of the plot.
        style (DocumentStyle, optional): The style to use for the plot.
            Defaults to DocumentStyle.THESIS.
        ax (Axes | None, optional): The matplotlib Axes object to plot on.
            If None, a new figure and axes will be created. Defaults to None.
        save_path (str | None, optional): Path to save the plot. If None,
            the plot will not be saved but shown instead. Defaults to None.

    Return:
        Axes: The matplotlib Axes object containing the plot.
    """
    axis_config = deepcopy(axis_config)
    fig, ax = figure_double_plot(style=style,
                                 axis=ax)
    plot_convergence(results,
                     axis_config=axis_config,
                     style=style,
                     ax=ax[0])
    axis_config.ylabel = "Conv"
    axis_config.make_legend = False
    axis_config.logy = True
    plot_convergence_measure(results,
                             axis_config=axis_config,
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
    CONVMEAS = "convergence_measure"
    RESANDERR = "res_and_err"
    RESANDCONVM = "res_and_convmeasure"
    BOTH = "res_and_err" # Alias for first use of both. Deprecated

def plot_convergence_auto(results: list[ConvergingPlottable],
                          axis_config: AxisConfig,
                          exact_results: StandardPlottable | None = None,
                          style: DocumentStyle = DocumentStyle.THESIS,
                          which: WhichConvergence = WhichConvergence.BOTH,
                          ax: Axes | None = None,
                          save_path: str | None = None,
                          style_mapping: StyleMapping | None = None
                          ) -> Axes:
    """
    Automatically decides whether to plot convergence results, error
    convergence, or both based on the provided arguments.

    Args:
        results (list[ConvergingPlottable]): A list of ConvergingPlottable objects
            containing the data to plot.
        axis_config (AxisConfig): Configuration for the axes of the plot.
        exact_results (StandardPlottable | None, optional): Reference results to plot.
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
    kwargs = {
        "style": style,
        "ax": ax,
        "save_path": save_path
    }
    if which is WhichConvergence.RESULTS:
        res =  plot_convergence(
            results,
            axis_config,
            exact_results=exact_results,
            style_mapping=style_mapping,
            **kwargs
        )
    elif which is WhichConvergence.ERRORS:
        res =  plot_error_convergence(
            results,
            axis_config,
            exact_results,
            **kwargs
        )
    elif which is WhichConvergence.RESANDERR:
        res =  plot_convergence_and_error(
            results,
            axis_config,
            exact_results=exact_results,
            **kwargs
        )
    elif which is WhichConvergence.CONVMEAS:
        res =  plot_convergence_measure(
            results,
            axis_config,
            **kwargs
        )
    elif which is WhichConvergence.RESANDCONVM:
        res =  plot_convergence_and_convmeasure(
            results,
            axis_config,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown value for which: {which}!")
    return res
