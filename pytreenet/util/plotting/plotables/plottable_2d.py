"""
A 2D plottable class for visualizing the relationship between two parameters.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Self, Callable

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..line_config import LineConfig
    from ...experiment_util.sim_params import SimulationParameters
    from ....time_evolution.results import Results


from .standard_plottable import Plottable


class Plottable2D(Plottable):

    def __init__(self,
                 point_coords: list[tuple[int, int]],
                 vals: npt.NDArray[np.floating],
                 assoc_params: dict[str, Any] | None = None
                 ) -> None:
        """
        Initializes a 2D plottable object.

        Args:
            point_coords (list[tuple[int, int]]):
                A list of tuples containing (x, y) coordinates of the points to be plotted.
            vals (npt.NDArray[np.floating]): The values associated with each (x, y) point,
                which will be visualized using color.
            assoc_params (dict[str, Any] | None): Optional dictionary of associated parameters
                that may be relevant for the plottable.

        """
        super().__init__(line_config=None, assoc_params=assoc_params)
        self.coords = point_coords
        self.vals = vals

    def find_x_points(self) -> list[int]:
        """
        Extracts the x-coordinates from the point coordinates.

        Returns:
            list[int]: A list of x-coordinates.
        """
        return sorted(set(coord[0] for coord in self.coords))

    def find_y_points(self) -> list[int]:
        """
        Extracts the y-coordinates from the point coordinates.

        Returns:
            list[int]: A list of y-coordinates.
        """
        return sorted(set(coord[1] for coord in self.coords))

    def get_value_matrix(self) -> npt.NDArray[np.floating]:
        """
        Constructs a 2D array of values corresponding to the (x, y) coordinates.

        Returns:
            npt.NDArray[np.floating]: A 2D array where the entry at (i, j) corresponds
                to the value associated with the point at (x_i, y_j).
        """
        x_points = self.find_x_points()
        y_points = self.find_y_points()
        value_matrix = np.zeros((len(x_points), len(y_points)))
        for (x, y), val in zip(self.coords, self.vals):
            x_index = x_points.index(x)
            y_index = y_points.index(y)
            value_matrix[x_index, y_index] = val
        return value_matrix

    def plot_on_axis(self, ax):
        """
        Plots the 2D plottable on the given axis.

        Args:
            ax: The matplotlib axis to plot on.
        """
        ax.imshow(self.get_value_matrix(), origin='lower')
        yticks = ax.get_yticks()
        xticks = ax.get_xticks()
        x_points = self.find_x_points()
        y_points = self.find_y_points()
        if len(x_points) < len(xticks):
            xticks = [i for i in range(len(x_points))]
        ax.set_xticks(xticks, [str(x_points[int(i)]) for i in xticks])
        if len(y_points) < len(yticks):
            yticks = [i for i in range(len(y_points))]
        ax.set_yticks(yticks, [str(y_points[int(i)]) for i in yticks])

    def add_point(self, x: int, y: int, val: float) -> None:
        """
        Adds a single point to the plottable.

        Args:
            x (int): The x-coordinate of the point.
            y (int): The y-coordinate of the point.
            val (float): The value associated with the point,
                which will be visualized using color.
        """
        self.coords.append((x, y))
        self.vals = np.append(self.vals, val)

    @classmethod
    def from_simulation_result(cls,
                               result: Results,
                               sim_params: SimulationParameters,
                               extraction_function: Callable[[
                                   Results, SimulationParameters], tuple[list[int], list[int], list[float]]]
                               ) -> Self:
        """
        Creates a Plottable2D object from simulation results.

        Args:
            result (Results): The results object containing the data from
                the simulation.
            sim_params (SimulationParameters): The parameters used for the simulation.
            extraction_function (Callable[[Results, SimulationParameters], tuple[list[int], list[int], list[float]]]):
                A function that takes the results and simulation parameters and
                returns a tuple of three lists (x_coords, y_coords, vals).

        Returns:
            Self: A new instance of Plottable2D containing the extracted
                (x, y, val) data from the simulation results.
        """
        x_coords, y_coords, vals = extraction_function(result, sim_params)
        point_coords = list(zip(x_coords, y_coords))
        vals_array = np.array(vals)
        return cls(point_coords=point_coords,
                   vals=vals_array,
                   assoc_params=sim_params.to_json_dict())

    def x_limits(self) -> tuple[float, float]:
        return (min(coord[0] for coord in self.coords), max(coord[0] for coord in self.coords))

    def y_limits(self) -> tuple[float, float]:
        return (min(coord[1] for coord in self.coords), max(coord[1] for coord in self.coords))


def average(plottables: list[Plottable2D]) -> Plottable2D:
    """
    Averages the values of multiple Plottable2D objects that have the same coordinates.

    Args:
        plottables (list[Plottable2D]): A list of Plottable2D objects to be averaged.

    Returns:
        Plottable2D: A new Plottable2D object with the same coordinates and the average values.
    """
    if not plottables:
        raise ValueError("The list of plottables cannot be empty!")

    # Assume all plottables have the same coordinates
    vals_matrix = np.array([plottable.get_value_matrix() for plottable in plottables])
    avg_vals = np.mean(vals_matrix, axis=0)
    x_coords = plottables[0].find_x_points()
    y_coords = plottables[0].find_y_points()
    mapping = {}
    for indx, x in enumerate(x_coords):
        for indy, y in enumerate(y_coords):
            mapping[(x, y)] = avg_vals[indx, indy]
    point_coords = list(mapping.keys())
    avg_vals_list = list(mapping.values())
    return Plottable2D(point_coords=point_coords,
                       vals=np.array(avg_vals_list),
                       assoc_params=plottables[0].assoc_params)
