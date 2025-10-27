"""
This module implements a plottable that contains multiple lines.
"""
from __future__ import annotations
from copy import copy, deepcopy
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ..line_config import LineConfig
from ..util import compute_alphas
from .standard_plottable import Plottable, StandardPlottable

class ConvergingPlottable(Plottable):
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
                 x_values: npt.NDArray[np.floating] | None = None,
                 assoc_params: dict[str, Any] | None = None
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
            assoc_params (dict[str, Any] | None): Associated parameters for
                the plottable. Defaults to None.
        """
        super().__init__(line_config=line_config, assoc_params=assoc_params)
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
        self.conv_param_values = conv_param_values

    @classmethod
    def from_numpy_list(cls,
                        values: list[npt.NDArray[np.floating]],
                        line_config: LineConfig,
                        conv_param_values: list[Any],
                        conv_param: str | None = None,
                        x_values: npt.NDArray[np.floating] | None = None,
                        assoc_params: dict[str, Any] | None = None
                        ) -> Self:
        """
        Create a ConvergingResults instance from a list of numpy arrays.
        
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
            assoc_params (dict[str, Any] | None): Associated parameters for
                the plottable. Defaults to None.
        """
        return cls(values=values,
                   line_config=line_config,
                   conv_param_values=conv_param_values,
                   conv_param=conv_param,
                   x_values=x_values,
                   assoc_params=assoc_params)

    def assoc_subset(self, other, ignored_keys = None):
        if ignored_keys is None:
            ignored_keys = set()
        if self.conv_param is not None:
            ignored_keys = copy(ignored_keys)
            ignored_keys.add(self.conv_param)
        return super().assoc_subset(other, ignored_keys)

    def sort(self):
        """
        Sort the convergence results by the convergence parameter values.
        """
        sorted_indices = np.argsort(self.conv_param_values)
        self.conv_param_values = [self.conv_param_values[i] for i in sorted_indices]
        self.values = [self.values[i] for i in sorted_indices]

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
            x_values=insrt(self.x_values),
            assoc_params=insrt(self.assoc_params)
        )

    def get_errors(self,
                   reference: "StandardPlottable",
                   deep: bool = False
                   ) -> Self:
        """
        Compute the absolute errors of the convergence results with respect
        to a set of reference results.

        Args:
            reference (StandardPlottable): The reference results to compare
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

    def accumulated_results(self) -> StandardPlottable:
        """
        Accumulates each result into one value by averaging.

        Returns:
            StandardPlottable: The accumulated results. They are in the
                same order as the result chains, i.e. the first element
                corresponds to the first parameter value.
        """
        y_vals = np.array([np.mean(result) for result in self.values])
        return StandardPlottable(x=self.x_values,
                                 y=y_vals,
                                 line_config=deepcopy(self.line_config),
                                 assoc_params=copy(self.assoc_params)
                                 )

    def average_results(self) -> StandardPlottable:
        """
        Averages the result over the convergence parameter values.

        Returns:
            StandardPlottable: The averaged results.
        """
        y_vals = np.mean(np.array(self.values), axis=0)
        return StandardPlottable(x=self.x_values,
                                 y=y_vals,
                                 line_config=deepcopy(self.line_config),
                                 assoc_params=copy(self.assoc_params)
                                 )

    def variance_results(self) -> StandardPlottable:
        """
        Computes the variance of the results over the convergence
        parameter values.

        Returns:
            StandardPlottable: The variance of the results.
        """
        y_vals = np.var(np.array(self.values), axis=0)
        return StandardPlottable(x=self.x_values,
                                 y=y_vals,
                                 line_config=deepcopy(self.line_config),
                                 assoc_params=copy(self.assoc_params)
                                 )

    def min_results(self) -> StandardPlottable:
        """
        Computes the minimum of the results over the convergence
        parameter values.

        Returns:
            StandardPlottable: The minimum of the results.
        """
        y_vals = np.min(np.array(self.values), axis=0)
        return StandardPlottable(x=self.x_values,
                                 y=y_vals,
                                 line_config=deepcopy(self.line_config),
                                 assoc_params=copy(self.assoc_params)
                                 )

    def max_results(self) -> StandardPlottable:
        """
        Computes the maximum of the results over the convergence
        parameter values.

        Returns:
            StandardPlottable: The maximum of the results.
        """
        y_vals = np.max(np.array(self.values), axis=0)
        return StandardPlottable(x=self.x_values,
                                 y=y_vals,
                                 line_config=deepcopy(self.line_config),
                                 assoc_params=copy(self.assoc_params)
                                 )

    def difference_between(self,
                           indx0: int,
                           indx1: int,
                           abs: bool = True
                           ) -> StandardPlottable:
        """
        Computes the difference between result entries at two indices.

        Args:
            indx0 (int): Index of the first result entry.
            indx1 (int): Index of the second result entry.
            abs (bool, optional): Whether to take the absolute value of
                the difference. Defaults to True.

        Returns:
            StandardPlottable: The difference between the two result entries.
        """
        y_diff = np.zeros((len(self.values),), dtype=self.values[0].dtype)
        for i, vals in enumerate(self.values):
            if abs:
                y_diff[i] = np.abs(vals[indx1] - vals[indx0])
            else:
                y_diff[i] = vals[indx1] - vals[indx0]
        return StandardPlottable(x=self.conv_param_values,
                                    y=y_diff,
                                    line_config=deepcopy(self.line_config),
                                    assoc_params=copy(self.assoc_params)
                                    )

    def x_limits(self) -> tuple[float, float]:
        """
        Get the x-axis limits for the plot.

        Returns:
            tuple[float, float]: The minimum and maximum x values.
        """
        return (np.min(self.x_values), np.max(self.x_values))

    def y_limits(self) -> tuple[float, float]:
        """
        Get the y-axis limits for the plot.

        Returns:
            tuple[float, float]: The minimum and maximum y values across all
                convergence results.
        """
        all_values = np.concatenate(self.values)
        return (np.min(all_values), np.max(all_values))

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

    @classmethod
    def from_multiple_standards(cls,
                                standards: list[StandardPlottable],
                                line_config: LineConfig | None = None,
                                conv_param: str | None = None
                                ) -> Self:
        """
        Create a ConvergingResults instance from multiple StandardPlottable
        instances.

        Args:
            standards (list[StandardPlottable]): List of StandardPlottable
                instances to combine.
            line_config (LineConfig | None, optional): The line configuration
                for the plot. If None, a default LineConfig will be used.
                Defaults to None.

        Returns:
            ConvergingResults: An instance of ConvergingResults.
        """
        if conv_param is None and len(standards) < 2:
            raise ValueError("At least two standards must be provided!")
        if len(standards) == 1:
            single_plt = standards[0]
            return cls([single_plt.y],
                       single_plt.line_config,
                       [single_plt.assoc_params[conv_param]],
                       conv_param=conv_param,
                       x_values=single_plt.x,
                       assoc_params=copy(single_plt.assoc_params))
        elif len(standards) == 0:
            raise ValueError("At least one standard must be provided!")
        if line_config is None:
            line_config = standards[0].line_config
        values = [std.y for std in standards]
        # Check that all x values are the same
        first_x = standards[0].x
        for std in standards[1:]:
            if not np.allclose(std.x, first_x):
                raise ValueError("All standards must have the similar x values!")
        # Find differing associated parameter
        assoc_params0 = standards[0].assoc_params
        if conv_param is None:
            conv_param = ""
            assoc_params1 = standards[1].assoc_params
            for key in assoc_params0.keys():
                if assoc_params1[key] != assoc_params0[key]:
                    conv_param = key
                    break
            if conv_param == "":
                raise ValueError("No differing associated parameter found!")
        for std in standards[1:]:
            if not std.assoc_equal(standards[0], ignored_keys={conv_param}):
                raise ValueError("All standards must differ in the same parameter!")
        conv_param_values = [std.assoc_params[conv_param]
                             for std in standards]
        assoc_params = copy(assoc_params0)
        assoc_params.pop(conv_param)
        out = cls(values,
                   line_config,
                   conv_param_values,
                   conv_param=conv_param,
                   x_values=first_x,
                   assoc_params=assoc_params
                   )
        out.sort()
        return out
