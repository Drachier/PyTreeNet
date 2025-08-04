"""
This module provides a configuration class to set up axis properties for
plotting.
"""
from dataclasses import dataclass

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

@dataclass
class AxisConfig:
    """
    Configuration for axis properties in a plot.

    Attributes:
        title (str): Title of the axis.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        make_legend (bool): Whether to create a legend for the plot.
        
    """
    title: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    make_legend: bool = False

    def apply_to_axis(self,
                      ax: Axes | None = None):
        """
        Apply the axis configuration to the given axes.
        
        Args:
            ax (Axes | None): The matplotlib Axes object to apply the
                configuration to. If None, the current axes will be used.
        """
        if ax is None:
            ax = plt.gca()
        if self.title is not None:
            ax.set_title(self.title)
        if self.xlabel is not None:
            ax.set_xlabel(self.xlabel)
        if self.ylabel is not None:
            ax.set_ylabel(self.ylabel)
        if self.make_legend:
            ax.legend()

    def apply_with_plt(self):
        """
        Apply the axis configuration using the current pyplot state.
        """
        self.apply_to_axis(ax=plt.gca())

    def set_x_to_time(self):
        """
        Set the x-axis label to time.

        This is a convenience method to set the x-axis label to a common
        label used in time series plots.
        """
        self.xlabel = "Time $t$"
