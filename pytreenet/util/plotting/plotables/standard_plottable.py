"""
This module defines abstract bases classes for plotting.
"""
from __future__ import annotations
from copy import copy, deepcopy
from typing import Any, Callable, Self
from abc import ABC, abstractmethod
from inspect import signature

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from h5py import File

from ..line_config import (LineConfig, StyleMapping)
from ...experiment_util.sim_params import SimulationParameters
from ....time_evolution.results import Results

STANDARD_REFERENCE_FLAG = "EXACT"

class Plottable(ABC):
    """
    Abstract base class for objects that can be plotted.
    """

    def __init__(self,
                 line_config: LineConfig | None = None,
                 assoc_params: dict[str, Any] | None = None
                 ) -> None:
        """
        Initialize the Plottable object with a line configuration.

        Args:
            line_config (LineConfig | None, optional): The line configuration
                for the plot. Defaults to None.
        """
        if line_config is None:
            line_config = LineConfig()
        self.line_config = line_config
        if assoc_params is None:
            assoc_params = {}
        self.assoc_params = assoc_params

    def assoc_subset(self,
                     other: Plottable,
                     ignored_keys: set[str] | None = None
                     ) -> bool:
        """
        Check if the associated parameters of this plottable are a subset of
        another plottable's associated parameters.

        This can usually be used to check if two plottables are for the same
        parameter set, ignoring other associated parameters.

        Args:
            other (Plottable): The other plottable to compare against.
            ignored_keys (set[str] | None, optional): Keys to ignore in the
                comparison. Defaults to None.

        Returns:
            bool: True if this plottable's associated parameters are a subset
                of the other plottable's associated parameters, False otherwise.
        """
        if ignored_keys is None:
            ignored_keys = set()
        keys_to_check = set(self.assoc_params.keys()) - ignored_keys
        for key in keys_to_check:
            if key not in other.assoc_params:
                return False
            if self.assoc_params[key] != other.assoc_params[key]:
                return False
        return True

    def assoc_equal(self,
                    other: Plottable,
                    ignored_keys: set[str] | None = None
                    ) -> bool:
        """
        Check if the associated parameters of this plottable are equal to
        another plottable's associated parameters.

        Args:
            other (Plottable): The other plottable to compare against.
            ignored_keys (set[str] | None, optional): Keys to ignore in the
                comparison. Defaults to None.

        Returns:
            bool: True if this plottable's associated parameters are equal
                to the other plottable's associated parameters, False otherwise.
        """
        if ignored_keys is None:
            ignored_keys = set()
        is_subset = self.assoc_subset(other, ignored_keys)
        is_superset = other.assoc_subset(self, ignored_keys)
        return is_subset and is_superset

    def apply_style_mapping(self,
                           style_mapping: StyleMapping
                           ) -> None:
        """
        Apply the given style mapping to this plottable's line configuration.

        Args:
            style_mapping (StyleMapping): The style mapping to apply.
        """
        for parameter in style_mapping.get_parameters():
            if parameter in self.assoc_params:
                value = self.assoc_params[parameter]
                if style_mapping.value_valid(parameter, value):
                    style_mapping.apply_to_config(self.line_config,
                                                   parameter,
                                                   value)

    @abstractmethod
    def plot_on_axis(self,
                      ax):
        """
        Apply the plotting to the given axes.

        Args:
            ax: The matplotlib Axes object to apply the plotting to.
        """
        pass

    @abstractmethod
    def x_limits(self) -> tuple[float, float]:
        """
        Get the x-axis limits for the plot.

        Returns:
            tuple[float, float]: The (min, max) x-axis limits.
        """
        pass

    @abstractmethod
    def y_limits(self) -> tuple[float, float]:
        """
        Get the y-axis limits for the plot.

        Returns:
            tuple[float, float]: The (min, max) y-axis limits.
        """
        pass

class StandardPlottable(Plottable):
    """
    A class that is a single line plot.

    Attributes:
        x (npt.NDArray[np.floating]): X values for the plot.
        y (npt.NDArray[np.floating]): Y values for the plot.
        line_config (LineConfig): Configuration for the line style in the plot.
    """

    def __init__(self,
                 x: npt.NDArray[np.floating],
                 y: npt.NDArray[np.floating],
                 line_config: LineConfig | None = None,
                 assoc_params: dict[str, Any] | None = None
                 ) -> None:
        """
        Initialize a ReferenceResults instance.

        Args:
            x (npt.NDArray[np.floating]): X values for the reference results.
            y (npt.NDArray[np.floating]): Y values for the reference results.
            line_config (LineConfig): Configuration for the line style in the
                plot. Defaults to None.
            assoc_params (dict[str, Any] | None): Associated parameters for
                the plottable. Defaults to None.
        """
        super().__init__(line_config=line_config, assoc_params=assoc_params)
        self.x = x
        self.y = y

    def add_point(self,
                  x: float,
                  y: float) -> None:
        """
        Add a single point to the reference results.

        Args:
            x (float): The x value of the point to add.
            y (float): The y value of the point to add.
        """
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)

    def get_value(self,
                  x: float
                  ) -> float | None:
        """
        Get the y value corresponding to the given x value.

        Args:
            x (float): The x value to look for.
        
        Returns:
            float | None: The corresponding y value, or None if not found.
        """
        indices = np.where(self.x == x)[0]
        if len(indices) == 0:
            return None
        return self.y[indices[0]]

    def sort_by_x(self) -> None:
        """
        Sort the reference results by the x values.
        """
        sorted_indices = np.argsort(self.x)
        self.x = self.x[sorted_indices]
        self.y = self.y[sorted_indices]

    def x_limits(self) -> tuple[float, float]:
        """
        Get the x-axis limits for the reference results.

        Returns:
            tuple[float, float]: The minimum and maximum x values.
        """
        if len(self.x) == 0:
            return (float("inf"), float("-inf"))
        return (np.min(self.x), np.max(self.x))

    def y_limits(self) -> tuple[float, float]:
        """
        Get the y-axis limits for the reference results.

        Returns:
            tuple[float, float]: The minimum and maximum y values.
        """
        if len(self.y) == 0:
            return (float("inf"), float("-inf"))
        return (np.min(self.y), np.max(self.y))

    def start_end_difference(self,
                             absolute: bool = True
                             ) -> float:
        """
        Get the difference between the last and first y values.

        Args:
            absolute (bool): Whether to return the absolute difference.

        Returns:
            float: The difference between the last and first y values.
        """
        if len(self.y) < 2:
            return 0.0
        if absolute:
            return abs(self.y[-1] - self.y[0])
        return self.y[-1] - self.y[0]

    def plot_on_axis(self,
                     ax: Axes | None = None,
                     set_label: bool = True):
        """
        Plot the reference results on the given axes.

        Args:
            ax (Axes | None): The matplotlib Axes object to plot on.
                If None, the current axes will be used.
            set_label (bool): Whether to set the label for the line based on
                the line configuration. Defaults to True.
                This can be used to avoid creating a legend entry for this
                plottable.
        """
        if ax is None:
            ax = plt.gca()
        if set_label:
            kwargs = self.line_config.to_kwargs()
        else:
            kwargs = self.line_config.to_kwargs(exclude={"label"})
        ax.plot(self.x, self.y, **kwargs)

    def apply_numpy_to_y(self,
                         func: Callable[[npt.NDArray[np.floating]],
                                        npt.NDArray[np.floating]]
                         ) -> npt.NDArray[np.floating]:
        """
        Apply a numpy function to the y values.

        Args:
            func (Callable[[npt.NDArray[np.floating]],
                           npt.NDArray[np.floating]]): The numpy function to
                apply to the y values.
        
        Returns:
            npt.NDArray[np.floating]: The result of applying the function to
                the y values.
        """
        return func(self.y)

    def plot_error_bars(self,
                        y_min: StandardPlottable,
                        y_max: StandardPlottable,
                        ax: Axes | None = None
                        ) -> None:
        """
        Plot error bars on the current axes using the given min and max
        StandardPlottable objects.

        Args:
            y_min (StandardPlottable): The StandardPlottable object
                representing the minimum y values.
            y_max (StandardPlottable): The StandardPlottable object
                representing the maximum y values.
        """
        if ax is None:
            ax = plt.gca()
        yerr = np.stack([y_min.y, y_max.y], axis=0)
        ax.errorbar(self.x, self.y, yerr=yerr, **self.line_config.to_kwargs())

    @classmethod
    def from_simulation_result(cls,
                               result: Results,
                               sim_params: SimulationParameters,
                               extraction_function: Callable
                               ) -> Self:
        """
        Create a StandardPlottable from a simulation result and parameters.

        Args:
            result (Results): The simulation results.
            sim_params (SimulationParameters): The simulation parameters.
            extraction_function (Callable): A function to extract the relevant data.

        Returns:
            StandardPlottable: An instance of StandardPlottable.
        
        Raises:
            ValueError: If the extraction function does not have the correct
                signature.
        """
        sig = signature(extraction_function)
        if len(sig.parameters) == 1:
            res = extraction_function(result)
        elif len(sig.parameters) == 2:
            res = extraction_function(result, sim_params)
        else:
            raise ValueError("Extraction function must take 1 or 2 arguments!")
        try:
            times, data = res
        except (ValueError, TypeError) as e:
            raise ValueError("Extraction function must return a tuple of "
                             "(times, data)!") from e
        return cls(x=times,
                   y=data,
                   line_config=LineConfig(),
                   assoc_params=copy(sim_params.to_json_dict())
                   )

    def empty_clone(self) -> Self:
        """
        Create an empty clone of this StandardPlottable.

        Returns:
            StandardPlottable: An empty clone of this StandardPlottable.
        """
        return StandardPlottable(x=np.array([], dtype=float),
                                 y=np.array([], dtype=float),
                                 line_config=deepcopy(self.line_config),
                                 assoc_params=copy(self.assoc_params)
                                 )

    def len(self) -> int:
        """
        Get the length of the plottable data.

        Returns:
            int: The length of the x and y data arrays.
        """
        out = len(self.x)
        assert out == len(self.y)
        return out

    def truncate(self,
                 new_length: int
                 ) -> Self:
        """
        Truncate the plottable to the given new length.

        Args:
            new_length (int): The new length to truncate to.

        Returns:
            StandardPlottable: The truncated plottable.
        """
        newx = self.x[:new_length]
        newy = self.y[:new_length]
        return StandardPlottable(x=newx,
                                 y=newy,
                                 line_config=deepcopy(self.line_config),
                                 assoc_params=copy(self.assoc_params)
                                 )

    def save_to_h5(self, file: str | File):
        """
        Save the StandardPlottable to an HDF5 file.

        Args:
            file (str | File): The path to the HDF5 file or an open h5py File
                object.
        """
        if isinstance(file, str):
            with File(file, "w") as h5file:
                self.save_to_h5(h5file)
        else:
            file.create_dataset("x", data=self.x)
            file.create_dataset("y", data=self.y)
            for key, value in self.assoc_params.items():
                file.attrs[key] = value

    def interpolate_y(self,
                      new_x: npt.NDArray[np.floating]
                      ) -> Self:
        """
        Interpolate the y values to the given new x values.

        Args:
            new_x (npt.NDArray[np.floating]): The new x values to interpolate to.
        
        Returns:
            StandardPlottable: A new StandardPlottable with the interpolated
                y values.
        """
        new_y = np.interp(new_x, self.x, self.y)
        new = self.empty_clone()
        new.x = new_x
        new.y = new_y
        return new

def combine_equivalent_standard_plottables(x_vals: StandardPlottable,
                                           y_vals: StandardPlottable
                                           ) -> StandardPlottable:
    """
    Combine two StandardPlottable objects with the same x values into a new
    StandardPlottable object.

    Args:
        x_vals (StandardPlottable): The StandardPlottable object providing
            the new x values.
        y_vals (StandardPlottable): The StandardPlottable object providing
            the y values.

    Returns:
        StandardPlottable: A new StandardPlottable object with the y values
            from x_vals as new x values and the y values from y_vals as the
            new y values.
    """
    if not x_vals.assoc_subset(y_vals):
        raise ValueError("The associated parameters of the x values plottable"
                         " are not a subset of the y values plottable!")
    if len(x_vals.y) != len(y_vals.y):
        raise ValueError("The x and y values must have the same length!")
    return StandardPlottable(x=x_vals.y,
                             y=y_vals.y,
                             line_config=deepcopy(y_vals.line_config),
                             assoc_params=copy(y_vals.assoc_params)
                             )

def sort_by_style_mapping(plottables: list[Plottable],
                          style_mapping: StyleMapping
                          ) -> dict[tuple[tuple[str, Any], ...], list[Plottable]]:
    """
    Sorts the given plottables by the specified style mapping.

    Args:
        plottables (list[Plottable]): The list of plottables to sort.
        style_mapping (StyleMapping): The style mapping to use for sorting.

    Returns:
        dict[tuple[tuple[str, Any], ...], list[Plottable]]: A dictionary mapping
            style keys to lists of plottables that match the style.
    """
    sorted_plottables = {}
    for plottable in plottables:
        style_key = [(parameter,plottable.assoc_params[parameter])
                     for parameter in style_mapping.get_parameters()
                     if style_mapping.value_valid(parameter,plottable.assoc_params[parameter])]
        style_key = tuple(sorted(style_key))
        if style_key not in sorted_plottables:
            sorted_plottables[style_key] = []
        sorted_plottables[style_key].append(plottable)
    return sorted_plottables
