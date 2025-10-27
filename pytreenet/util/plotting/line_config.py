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
                return
            label = self.label
        kwargs = self.to_kwargs()
        kwargs['label'] = label
        ax.plot([],[],
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
                 param_to_option: dict[str, list[StyleOption]] | None = None,
                 param_value_to_style: dict[str, dict[Any, list[str]]] | None = None
                 ) -> None:
        """
        Initialize a StyleMapping object.

        Args:
            param_to_option (dict[str, list[StyleOption]] | None): A mapping from
                parameter keys to style options. If None, an empty mapping
                will be used. Defaults to None.
            param_value_to_style (dict[str, dict[Any, list[str]]] | None): A mapping
                from parameter keys to a mapping of parameter values to style
                values. If None, an empty mapping will be used. Defaults to
                None.
        """
        if param_to_option is None:
            param_to_option = {}
        if param_value_to_style is None:
            param_value_to_style = {}
        if len(set(param_to_option.values())) != len(param_to_option):
            errstr = "Each StyleOption can only be used once!"
            raise ValueError(errstr)
        for key, val_map in param_value_to_style.items():
            if key not in param_to_option:
                errstr = (f"Parameter '{key}' in value-to-style mapping is not "
                          f"in parameter-to-option mapping!")
                raise KeyError(errstr)
            for val, style_val in val_map.items():
                if len(style_val) != len(param_to_option[key]):
                    errstr = (f"Parameter '{key}' has {len(style_val)} style "
                              f"values for value '{val}', but {len(param_to_option[key])} "
                              f"are required!")
                    raise ValueError(errstr)
        self.param_to_option = param_to_option
        self.param_value_to_style = param_value_to_style

    def get_parameters(self) -> set[str]:
        """
        Get the set of parameter keys in the mapping.

        Returns:
            set[str]: The set of parameter keys.
        """
        return set(self.param_to_option.keys())

    def value_valid(self,
                    param_key: str,
                    value: Any
                    ) -> bool:
        """
        Check if a value is valid for a given parameter key.

        Args:
            param_key (str): The parameter key.
            value (Any): The value to check.

        Returns:
            bool: True if the value is valid, False otherwise.
        """
        if param_key not in self.get_parameters():
            errstr = f"No style mapping for parameter '{param_key}'!"
            raise KeyError(errstr)
        return value in self.param_value_to_style[param_key]

    def get_style_mapping(self,
                         param_key: str
                         ) -> dict[Any, list[str]]:
        """
        Get the style mapping for a given parameter key.

        Args:
            param_key (str): The parameter key.
        
        Returns:
            dict[Any, list[str]]: The style mapping associated with the parameter key.
        """
        if param_key not in self.param_value_to_style:
            errstr = f"No style mapping for parameter '{param_key}'!"
            raise KeyError(errstr)
        return self.param_value_to_style[param_key]

    def get_style_value(self,
                        param_key: str,
                        param_value: Any,
                        style_option: StyleOption
                        ) -> str:
        """
        Get the style value for a given parameter key and value.

        Args:
            param_key (str): The parameter key.
            param_value (Any): The parameter value.
            style_option (StyleOption): The style option to get the value for.
        
        Returns:
            str: The style value associated with the parameter key and value
                for the given style option.
        """
        if param_key not in self.param_value_to_style:
            errstr = f"No style mapping for parameter '{param_key}'!"
            raise KeyError(errstr)
        style_map = self.get_style_mapping(param_key)
        if param_value not in style_map:
            errstr = (f"No style value for parameter '{param_key}' "
                      f"with value '{param_value}'!")
            raise KeyError(errstr)
        option_index = self.param_to_option[param_key].index(style_option)
        return style_map[param_value][option_index]

    def apply_to_config(self,
                        line_config: LineConfig,
                        param_key: str,
                        param_value: Any
                        ) -> None:
        """
        Apply the style mapping to the given line configuration.

        Args:
            line_config (LineConfig): The line configuration to modify.
            param_key (str): The parameter key.
            param_value (Any): The parameter value.
        """
        style_options = self.get_style_options(param_key)
        for style_option in style_options:
            style_arg = style_option.value
            style_value = self.get_style_value(param_key,
                                               param_value,
                                               style_option)
            setattr(line_config, style_arg, style_value)

    def create_config(self,
                      keys_values: dict[str, Any],
                      allow_missing: bool = True
                      ) -> LineConfig:
        """
        Create a LineConfig from the given parameter keys and values.

        Args:
            keys_values (dict[str, Any]): A mapping from parameter keys to
                parameter values.
            allow_missing (bool): Whether to allow missing parameter keys
                in the mapping. If False, an error will be raised if a
                parameter key is not found in the mapping. Defaults to True.
        
        Returns:
            LineConfig: The created LineConfig object.
        """
        line_config = LineConfig()
        for key, value in keys_values.items():
            if key in self.get_parameters():
                self.apply_to_config(line_config, key, value)
            else:
                if not allow_missing:
                    errstr = f"No style mapping for parameter '{key}'!"
                    raise KeyError(errstr)
        return line_config

    def get_style_options(self,
                          param_key: str
                          ) -> list[StyleOption]:
        """
        Get the style options for a given parameter key.

        Args:
            param_key (str): The parameter key.
        
        Returns:
            list[StyleOption]: The style option associated with the parameter key.
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
        for options in self.param_to_option.values():
            if style_option in options:
                return True
        return False

    def set_no_lines(self,
                     param_key: str
                     ) -> None:
        """
        For the given parameter key, set no lines to be drawn.

        Args:
            param_key (str): The parameter key.
        """
        if self.option_in_use(StyleOption.LINESTYLE):
            errstr = "Style option 'linestyle' is already in use!"
            raise ValueError(errstr)
        # Set the style option to the linestyle ""
        if param_key not in self.param_to_option:
            self.param_to_option[param_key] = [StyleOption.LINESTYLE]
        else:
            self.param_to_option[param_key].append(StyleOption.LINESTYLE)
        # Set the style value for all parameter values to ""
        value_map = self.param_value_to_style.get(param_key, {})
        for val in value_map:
            value_map[val].append("")
        self.param_value_to_style[param_key] = value_map

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
        if param_key in self.param_to_option:
            self.param_to_option[param_key].append(style_option)
            curr_map = self.param_value_to_style[param_key]
            for val, style_val in value_to_style.items():
                if val in curr_map:
                    curr_map[val].append(style_val)
                else:
                    raise KeyError(f"Value '{val}' not in existing mapping!")
        else:
            self.param_to_option[param_key] = [style_option]
            self.param_value_to_style[param_key] = {key: [val]
                                                    for key, val in value_to_style.items()}

    def apply_legend(self,
                     ax: Axes | None = None,
                     ignore: set[tuple[str,Any], str] | None = None
                     ) -> None:
        """
        Apply the legend for all style mappings to the given axes.

        Args:
            ax (Axes | None): The matplotlib Axes object to plot the legend on.
                If None, the current axes will be used.
            ignore (set[tuple(str, Any)] | None): A set of parameter key-value
                pairs to ignore when applying the legend. If None, no pairs
                will be ignored. Defaults to None.
        """
        if ignore is None:
            ignore = set()
        if ax is None:
            ax = plt.gca()
        for param_key, value_map in self.param_value_to_style.items():
            style_options = self.get_style_options(param_key)
            for param_value, style_values in value_map.items():
                if not (param_key, param_value) in ignore and param_key not in ignore:
                    line_config = LineConfig()
                    for style_option, style_value in zip(style_options,
                                                        style_values):
                        setattr(line_config, style_option.value, style_value)
                    if param_key == "ttns_structure":
                        label = TTNStructure(param_value).label()
                    else:
                        label = f"{param_key}={param_value}"
                    line_config.label = label
                    if StyleOption.COLOR not in style_options:
                        line_config.color = "black"
                    line_config.plot_legend(ax=ax)

    @classmethod
    def from_filter_and_choice(cls,
                               md_filter: MetadataFilter,
                               style_choice: dict[str, StyleOption | list[StyleOption]]
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
        for key, options in style_choice.items():
            if not isinstance(options, list):
                options = [options]
            for option in options:
                if not md_filter.filters_parameter(key):
                    errstr = (f"Parameter '{key}' in style choice is not in the "
                            f"metadata filter!")
                    raise KeyError(errstr)
                values = md_filter.get_criterium(key)
                if key == "ttns_structure":
                    style_values = [config_from_ttn_structure(TTNStructure(v), {option.value})
                                    for v in values]
                    value_to_style = {v: getattr(s, option.value)
                                    for v, s in zip(values, style_values)}
                else:
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
