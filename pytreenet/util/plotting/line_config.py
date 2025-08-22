"""
This module implements a class that can be used to configure a single line plot
in matplotlib.
"""
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ...special_ttn.special_states import TTNStructure

@dataclass
class LineConfig:
    """
    Configuration for a single line in a plot.
    
    Attributes:
        color (str): Color of the line.
        linestyle (str): Style of the line (e.g., 'solid', 'dashed').
        linewidth (float): Width of the line.
        label (str): Label for the line, used in legends.
    """
    color: str | None = None
    linestyle: str | None = None
    linewidth: float | None = None
    label: str | None = None
    marker: str | None = None

    def to_kwargs(self, exclude: set[str] | None = None
                  ) -> dict:
        """
        Convert the LineConfig to a dictionary of keyword arguments for
        matplotlib's plot function.

        Args:
            exclude (set[str]): Set of attributes to exclude from the
                keyword arguments.
        
        Returns:
            dict: Dictionary of keyword arguments.
        """
        if exclude is None:
            exclude = set()
        kwargs = {}
        if self.color is not None and 'color' not in exclude:
            kwargs['color'] = self.color
        if self.linestyle is not None and 'linestyle' not in exclude:
            kwargs['linestyle'] = self.linestyle
        if self.linewidth is not None and 'linewidth' not in exclude:
            kwargs['linewidth'] = self.linewidth
        if self.label is not None and 'label' not in exclude:
            kwargs['label'] = self.label
        if self.marker is not None and 'marker' not in exclude:
            kwargs['marker'] = self.marker
        return kwargs

    def plot_legend(self,
                    ax: Axes | None = None,
                    label: str | None = None):
        """
        Plot the legend for the line configuration on the given axes.
        
        Args:
            ax (Axes | None): The matplotlib Axes object to plot the legend on.
                If None, the current axes will be used.
            label (str | None): The label for the legend. If None, the label
                from the LineConfig will be used.
        """
        if ax is None:
            ax = plt.gca()
        if label is None:
            if self.label is None:
                raise ValueError("No label provided for legend!")
            label = self.label
        kwargs = self.to_kwargs()
        kwargs['label'] = label
        ax.plot([], [],
                **kwargs)

def config_from_ttn_structure(
    ttn_structure: TTNStructure,
    use_attributes: set[str]
    ) -> LineConfig:
    """
    Create a LineConfig from a TTNStructure and a set determining which
    attributes to use.

    Args:
        ttn_structure (TTNStructure): The TTN structure to base the
            configuration on.
        use_attributes (set[str]): Set indicating which attributes to use
            as defined by the TTNStructure.

    Returns:
        LineConfig: A LineConfig object with the specified attributes.
    """
    config = LineConfig()
    if 'color' in use_attributes:
        config.color = ttn_structure.colour()
    if 'linestyle' in use_attributes:
        config.linestyle = ttn_structure.linestyle()
    if 'label' in use_attributes:
        config.label = ttn_structure.label()
    if 'marker' in use_attributes:
        config.marker = ttn_structure.marker()
    return config
