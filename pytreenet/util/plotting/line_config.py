"""
This module implements a class that can be used to configure a single line plot
in matplotlib.
"""
from __future__ import annotations
from typing import Any, Self
from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ...special_ttn.special_states import TTNStructure
from ..experiment_util.metadata_file import MetadataFilter

class StyleOption(Enum):
    """
    Enumeration of style options for line configuration.

    These are the options that are used to differentiate different lines in a
    plot. Other options like linewidth are not included as they are used
    globally for all lines in a plot.
    """
    COLOR = 'color'
    LINESTYLE = 'linestyle'
    MARKER = 'marker'

    def map_range(self) -> tuple[str]:
        """
        Get the range of values for the style option.

        Returns:
            tuple[str]: The range of values for the style option.
        """
        if self == StyleOption.COLOR:
            return ("tab:blue", "tab:orange", "tab:green", "tab:red",
                    "tab:purple", "tab:brown", "tab:pink", "tab:gray",
                    "tab:olive", "tab:cyan", "black")
        if self == StyleOption.LINESTYLE:
            return ("solid", "dashed", "dotted", "dashdot")
        if self == StyleOption.MARKER:
            return ("o", "s", "D", "^", "v", "<", ">", "p", "*", "h")
        raise ValueError("Invalid StyleOption!")

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

class StyleMapping:
    """
    A map that associates parameter keys to a style option and parameter
    values to a specific style value.
    """

    def __init__(self,
                 param_to_option: dict[str, StyleOption] | None = None,
                 param_value_to_style: dict[str, dict[Any, str]] | None = None
                 ) -> None:
        """
        Initialize a StyleMapping object.

        Args:
            param_to_option (dict[str, StyleOption] | None): A mapping from
                parameter keys to style options. If None, an empty mapping
                will be used. Defaults to None.
            param_value_to_style (dict[str, dict[Any, Any]] | None): A mapping
                from parameter keys to a mapping of parameter values to style
                values. If None, an empty mapping will be used. Defaults to
                None.
        """
        if len(set(param_to_option.values())) != len(param_to_option):
            errstr = "Each StyleOption can only be used once!"
            raise ValueError(errstr)
        self.param_to_option = param_to_option or {}
        self.param_value_to_style = param_value_to_style or {}

    def get_style_mapping(self,
                         param_key: str
                         ) -> dict[Any, str]:
        """
        Get the style mapping for a given parameter key.

        Args:
            param_key (str): The parameter key.
        
        Returns:
            dict[Any, str]: The style mapping associated with the parameter key.
        """
        if param_key not in self.param_value_to_style:
            errstr = f"No style mapping for parameter '{param_key}'!"
            raise KeyError(errstr)
        return self.param_value_to_style[param_key]

    def get_style_value(self,
                        param_key: str,
                        param_value: Any
                        ) -> str:
        """
        Get the style value for a given parameter key and value.

        Args:
            param_key (str): The parameter key.
            param_value (Any): The parameter value.
        
        Returns:
            str: The style value associated with the parameter key and value.
        """
        if param_key not in self.param_value_to_style:
            errstr = f"No style mapping for parameter '{param_key}'!"
            raise KeyError(errstr)
        style_map = self.get_style_mapping(param_key)
        if param_value not in style_map:
            errstr = (f"No style value for parameter '{param_key}' "
                      f"with value '{param_value}'!")
            raise KeyError(errstr)
        return style_map[param_value]

    def get_style_option(self,
                         param_key: str
                         ) -> StyleOption:
        """
        Get the style option for a given parameter key.

        Args:
            param_key (str): The parameter key.
        
        Returns:
            StyleOption: The style option associated with the parameter key.
        """
        if param_key not in self.param_to_option:
            errstr = f"No style option for parameter '{param_key}'!"
            raise KeyError(errstr)
        return self.param_to_option[param_key]

    def option_in_use(self,
                      style_option: StyleOption
                      ) -> bool:
        """
        Check if a style option is in use.

        Args:
            style_option (StyleOption): The style option to check.

        Returns:
            bool: True if the style option is in use, False otherwise.
        """
        return style_option in self.param_to_option.values()

    def add_mapping(self,
                    param_key: str,
                    style_option: StyleOption,
                    value_to_style: dict[Any, str]
                    ) -> None:
        """
        Add a mapping from a parameter key to a style option and a mapping
        from parameter values to style values.

        Args:
            param_key (str): The parameter key.
            style_option (StyleOption): The style option to associate with
                the parameter key.
            value_to_style (dict[Any, str]): A mapping from parameter values
                to style values.
        """
        if self.option_in_use(style_option):
            errstr = f"Style option '{style_option}' is already in use!"
            raise ValueError(errstr)
        self.param_to_option[param_key] = style_option
        self.param_value_to_style[param_key] = value_to_style

    def change_mapping(self,
                       param_key: str,
                       style_option: StyleOption,
                       value_to_style: dict[Any, str]
                       ) -> None:
        """
        Change or add the mapping from parameter values to style values for
        a given parameter key.

        Args:
            param_key (str): The parameter key.
            style_option (StyleOption): The style option to associate with
                the parameter key.
            value_to_style (dict[Any, str]): The new mapping from parameter
                values to style values.
        """
        self.param_to_option[param_key] = style_option
        self.param_value_to_style[param_key] = value_to_style

    def change_value_mapping(self,
                             param_key: str,
                             value_to_style: dict[Any, str]
                             ) -> None:
        """
        Change or add the mapping from parameter values to style values for
        a given parameter key.

        Args:
            param_key (str): The parameter key.
            value_to_style (dict[Any, str]): The new mapping from parameter
                values to style values.
        """
        if param_key not in self.param_to_option:
            errstr = f"No style option for parameter '{param_key}'!"
            raise KeyError(errstr)
        self.param_value_to_style[param_key] = value_to_style

    @classmethod
    def from_filter_and_choice(cls,
                               md_filter: MetadataFilter,
                               style_choice: dict[str, StyleOption]
                               ) -> Self:
        """
        Create a StyleMapping from a MetadataFilter and a style choice.

        Args:
            md_filter (MetadataFilter): The metadata filter to use.
            style_choice (dict[str, StyleOption]): A mapping from parameter
                keys to style options.

        Returns:
            StyleMapping: The created StyleMapping object.
        """
        mapping = cls()
        for key, option in style_choice.items():
            if not md_filter.filters_parameter(key):
                errstr = (f"Parameter '{key}' in style choice is not in the "
                          f"metadata filter!")
                raise KeyError(errstr)
            values = md_filter.get_criterium(key)
            potential_style_values = option.map_range()
            if len(values) > len(potential_style_values):
                errstr = (f"Not enough style values for parameter '{key}' "
                          f"with {len(values)} values!")
                raise ValueError(errstr)
            # We need to use enumerate, as there may be fewer values than
            # unique values, so we cannot use zip.
            value_to_style = {v: potential_style_values[i]
                              for i, v in enumerate(values)}
            mapping.add_mapping(key, option, value_to_style)
        return mapping
