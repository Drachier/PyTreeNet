"""
This module implements a function that plots the convergence of a result
over a given parameter range.
"""
from __future__ import annotations
from typing import Any, Self
from copy import deepcopy, copy
from enum import Enum

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from ..line_config import LineConfig
from ..axis_config import AxisConfig
from ..configuration import (DocumentStyle,
                             figure_from_style,
                             figure_double_plot)
from ..util import (save_figure,
                    compute_alphas)

class ConvergingResults:
    """
    A class to hold convergence results for plotting.

    These are the results of multiple runs of a model, where each run
    corresponds to a different value of a convergence parameter.

    Attributes:
        values (list[npt.NDArray[np.floating]]): List of arrays containing
            convergence results. Each array corresponds to a different
            value of the convergence parameter.
        line_config (LineConfig): Configuration for the line style in the plot.
        conv_param_values (list[Any]): Values of the convergence parameter.
        conv_param (str | None): The parameter over which convergence is
            plotted. Defaults to "parameter".
        x_values (npt.NDArray[np.floating] | None): X values for the plot.
            If None, it will be the convergence parameter values.
            Defaults to None.
    """

    def __init__(self,
                 values: list[npt.NDArray[np.floating]],
                 line_config: LineConfig,
                 conv_param_values: list[Any],
                 conv_param: str | None = None,
                 x_values: npt.NDArray[np.floating] | None = None
                 ):
        """
        Initialize a ConvergingResults instance.

        Args:
            values (list[npt.NDArray[np.floating]]): List of arrays containing
                convergence results should be ordered to match the
                corresponding values in `conv_param_values`.
            line_config (LineConfig): Configuration for the line style in the plot.
            conv_param_values (list[Any]): Values of the convergence parameter.
            conv_param (str | None, optional): The parameter over which convergence
                is plotted. Defaults to None.
            x_values (npt.NDArray[np.floating] | None, optional): X values for the
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
                        values: list[npt.NDArray[np.floating]],
                        line_config: LineConfig,
                        conv_param_values: list[Any],
                        conv_param: str | None = None,
                        x_values: npt.NDArray[np.floating] | None = None
                        ) -> Self:
        """
        Create a ConvergingResults instance from a list of arrays.

        Args:
            values (list[npt.NDArray[np.floating]]): List of arrays containing
                convergence results.
            line_config (LineConfig): Configuration for the line style in the plot.
            conv_param_values (list[Any]): Values of the convergence parameter.
            conv_param (str | None, optional): The parameter over which convergence
                is plotted. Defaults to None.
            x_values (npt.NDArray[np.floating] | None, optional): X values for the
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
                  data: dict[Any, npt.NDArray[np.floating]],
                  line_config: LineConfig,
                  conv_param: str | None = None,
                  x_values: npt.NDArray[np.floating] | None = None
                  ) -> Self:
        """
        Create a ConvergingResults instance from a dictionary.

        Args:
            data (dict[Any, npt.NDArray[np.floating]]): Dictionary where keys
                are convergence parameter values and values are arrays of
                convergence results.
            line_config (LineConfig): Configuration for the line style in the plot.
            conv_param (str | None, optional): The parameter over which convergence
                is plotted. Defaults to None.
            x_values (npt.NDArray[np.floating] | None, optional): X values for the
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
                        new_values: list[npt.NDArray[np.floating]],
                        deep: bool = False
                        ) -> Self:
        """
        Create a new ConvergingResults instance with the same configuration
        but different values.

        Args:
            new_values (list[npt.NDArray[np.floating]]): New list of arrays
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
        return self.__class__(
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

    def get_convergence(self,
                        deep: bool = False
                        ) -> Self:
        r"""
        Find the convergence of the result with respect to the next result.
        
        Args:
            deep (bool, optional): Whether to deep copy the other attributes of
                the new instance. Defaults to False.
        
        Returns:
            ConvergingResults: A new instance of ConvergingResults containing the
                convergence measure of the results, i.e.

                .. math::

                    C(n) = \frac{|r_{n+1} - r_{n}|}{|p_{n+1} - p_{n}|},

                where \(r_n\) is the result at step \(n\) and \(p_n\) is the convergence
                parameter at step \(n\).
        """
        convergences = []
        for i, values in enumerate(self.values[:-1]):
            next_values = self.values[i + 1]
            param_diff = np.abs(self.conv_param_values[i + 1] -
                                self.conv_param_values[i])
            if param_diff == 0:
                errstr = "Series of parameters contains two equal consecutive values!"
                raise ZeroDivisionError(errstr)
            conv_val = np.abs(next_values - values) / param_diff
            # The first one will always be zero, so we skip it
            convergences.append(conv_val[1:])
        # We need to get rid of the last parameter value
        new = copy(self)
        new.conv_param_values = new.conv_param_values[:-1]
        new.x_values = new.x_values[1:]
        return new.with_new_values(convergences, deep=deep)

    def accumulated_results(self) -> npt.NDArray[np.floating]:
        """
        Accumulates each result into one value by averaging.

        Returns:
            npt.NDArray[np.floating]: The accumulated results. They are in the
                same order as the result chains, i.e. the first element
                corresponds to the first parameter value.
        """
        return np.array([np.mean(result) for result in self.values])

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
        x (npt.NDArray[np.floating]): X values for the reference results.
        y (npt.NDArray[np.floating]): Y values for the reference results.
        line_config (LineConfig): Configuration for the line style in the plot.
    """

    def __init__(self,
                 x: npt.NDArray[np.floating],
                 y: npt.NDArray[np.floating],
                 line_config: LineConfig):
        """
        Initialize a ReferenceResults instance.

        Args:
            x (npt.NDArray[np.floating]): X values for the reference results.
            y (npt.NDArray[np.floating]): Y values for the reference results.
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
    plt.tight_layout()
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
    axis_config.set_attr_true_w_warning("logy")
    axis_config.apply_to_axis(ax=ax)
    set_x_limits(ax=ax,
                 results=results,
                 exact_results=exact_results)
    save_figure(fig,
                filename=save_path)
    plt.tight_layout()
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

def plot_convergence_measure(results: list[ConvergingResults],
                             axis_config: AxisConfig,
                             style: DocumentStyle = DocumentStyle.THESIS,
                             ax: Axes | None = None,
                             save_path: str | None = None
                             ) -> Axes:
    """
    Plots the convergence of the convergence measure.

    Args:
        results (list[ConvergingResults]): A list of ConvergingResults objects
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

def plot_convergence_and_convmeasure(results: list[ConvergingResults],
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
        results (list[ConvergingResults]): The list of ConvergingResults objects
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
    kwargs = {
        "style": style,
        "ax": ax,
        "save_path": save_path
    }
    if which is WhichConvergence.RESULTS:
        return plot_convergence(
            results,
            axis_config,
            exact_results=exact_results,
            **kwargs
        )
    if which is WhichConvergence.ERRORS:
        return plot_error_convergence(
            results,
            axis_config,
            exact_results,
            **kwargs
        )
    if which is WhichConvergence.RESANDERR:
        return plot_convergence_and_error(
            results,
            axis_config,
            exact_results=exact_results,
            **kwargs
        )
    if which is WhichConvergence.CONVMEAS:
        return plot_convergence_measure(
            results,
            axis_config,
            **kwargs
        )
    if which is WhichConvergence.RESANDCONVM:
        return plot_convergence_and_convmeasure(
            results,
            axis_config,
            **kwargs
        )

    raise ValueError(f"Unknown value for which: {which}!")
