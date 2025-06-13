"""
This module contains functions commonly used for time dependent ssimulations.
"""
from typing import Self, Callable
from dataclasses import dataclass

from math import sin, cos, pi

@dataclass
class ControlTimeParameters:
    """
    Parameters defining the time intervals of a control window.

    Attributes:
        start_time (float): The start time of the control window.
        start_end_time (float): The end time of the startup phase.
        shutdown_start_time (float): The start time of the shutdown phase.
        end_time (float): The end time of the control window.
    """
    start_time: float = 0.0
    start_end_time: float = 1.0
    shutdown_start_time: float = 2.0
    end_time: float = 4.0

    def __postinit__(self):
        assert self.start_time <= self.start_end_time, \
            "Start time must be less than or equal to start end time!"
        assert self.start_end_time <= self.shutdown_start_time, \
            "Start end time must be less than or equal to shutdown start time!"
        assert self.shutdown_start_time <= self.end_time, \
            "Shutdown start time must be less than or equal to end time!"

    def startup_time_window(self) -> float:
        """
        Returns the time it takes for the start up.
        """
        return self.start_end_time - self.start_time

    def shutdown_time_window(self) -> float:
        """
        Returns the time it takes for the shutdown.
        """
        return self.end_time - self.shutdown_start_time

    def linear_startup(self,
                       end_value: Callable | float = 1.0
                       ) -> Callable:
        """
        Returns a linear function for the startup phase of the control window.

        Parameters:
            end_value (Callable | float): The end value of the linear function.
                If a Callable is provided, it should take a float as input and
                return a float. If a float is provided, it will be used as the
                end value.

        Retrurns:
            Callable: A linear function that can be used as a startup function
                    for the control window defined by this object.
        """
        def linear_startup(t: float) -> float:
            denominator = self.startup_time_window()
            if denominator == 0:
                return 0.0
            enumerator = t - self.start_time
            if isinstance(end_value, float):
                factor = end_value
            else:
                factor = end_value(self.start_end_time)
            return factor * enumerator / denominator
        return linear_startup

    def linear_shutdown(self,
                        end_value: Callable | float = 1.0
                        ) -> Self:
        """
        Returns a linear function for the shutdown phase of the control window.

        Parameters:
            end_value (Callable | float): The end value of the linear function.
                If a Callable is provided, it should take a float as input and
                return a float. If a float is provided, it will be used as the
                end value.
        
        Returns:
            Callable: A linear function that can be used as a shutdown function
                    for the control window defined by this object.
        """
        def linear_shutdown(t: float) -> float:
            denominator = self.shutdown_time_window()
            if denominator == 0:
                return 0.0
            enumerator = t - self.shutdown_start_time
            if isinstance(end_value, float):
                factor = end_value
            else:
                factor = end_value(self.start_end_time)
            return factor * (1 - enumerator / denominator)
        return linear_shutdown

    def trig_startup(self,
                      end_value: Callable | float = 1.0
                      ) -> Callable:
        """
        Returns a trigonometric function for the startup phase.

        It increases from 0 to the end value in a sinusoidal manner.

        Parameters:
            end_value (Callable | float): The end value of the trigonometric
                function. If a Callable is provided, it should take a float as
                input and return a float. If a float is provided, it will be
                used as the end value.
            
        Returns:
            Callable: A trigonometric function that can be used as a startup
                    function for the control window defined by this object.
        """
        def trig_startup(t: float) -> float:
            tdiff = self.startup_time_window()
            if tdiff == 0:
                return 0.0
            if isinstance(end_value, float):
                factor = end_value
            else:
                factor = end_value(self.start_end_time)
            return factor * ( sin(pi / 2 * (t - self.start_time) / tdiff )) ** 2
        return trig_startup

    def trig_shutdown(self,
                        end_value: Callable | float = 1.0
                        ) -> Callable:
        """
        Returns a trigonometric function for the shutdown phase.

        Parameters:
            end_value (Callable | float): The end value of the trigonometric
                function. If a Callable is provided, it should take a float as
                input and return a float. If a float is provided, it will be
                used as the end value.
        
        Returns:
            Callable: A trigonometric function that can be used as a shutdown
                    function for the control window defined by this object.
        """
        def trig_shutdown(t: float) -> float:
            tdiff = self.shutdown_time_window()
            if tdiff == 0:
                return 0.0
            if isinstance(end_value, float):
                factor = end_value
            else:
                factor = end_value(self.start_end_time)
            return factor * (cos(pi / 2 * (t - self.shutdown_start_time) / tdiff)) ** 2
        return trig_shutdown

class ControlWindowFunction:
    """
    A class representing a control window function that can be turned and shut
    off in a controlled manner.

    The control window is divided into three phases: startup, middle, and
    shutdown. Each phase has a specific function that defines the behavior of
    the control window during that phase. Outside of these phases, the
    control window is turned off, i.e., the function returns 0.

    Attributes:
        startup (Callable): The function for the startup phase.
        middle (Callable): The function for the middle phase.
        shutdown (Callable): The function for the shutdown phase.
        control_times (ControlTimeParameters): The time intervals of the control
            window.
    """

    def __init__(self,
                 startup: Callable,
                 middle: Callable,
                 shutdown: Callable,
                 control_times: ControlTimeParameters):
        """
        Initializes the ControlWindowFunction with the given parameters.

        Args:
            startup (Callable): The function for the startup phase.
            middle (Callable): The function for the middle phase.
            shutdown (Callable): The function for the shutdown phase.
            control_times (ControlTimeParameters): The time intervals of the control window.
        """
        self.startup = startup
        self.middle = middle
        self.shutdown = shutdown
        self.control_times = control_times

    def __call__(self, t: float) -> float:
        """
        Calls the control window function with the given time.

        Args:
            t (float): The time at which to evaluate the function.

        Returns:
            float: The value of the control window function at time t.
        """
        if self.control_times.start_time <= t < self.control_times.start_end_time:
            return self.startup(t)
        if self.control_times.start_end_time <= t <= self.control_times.shutdown_start_time:
            return self.middle(t)
        if self.control_times.shutdown_start_time < t <= self.control_times.end_time:
            return self.shutdown(t)
        return 0.0

    @classmethod
    def constant_middle(cls,
                        startup: Callable,
                        shutdown: Callable,
                        control_times: ControlTimeParameters,
                        strength: float = 1.0,
                        ) -> Self:
        """
        Generates a ControlWindowFunction with a constant middle phase.
        """
        return cls(
            startup=startup,
            middle=lambda _: strength,
            shutdown=shutdown,
            control_times=control_times
        )

    @classmethod
    def instant_on_off(cls,
                        middle: Callable,
                        start_time: float = 0.0,
                        end_time: float = 1.0,
                        ) -> Self:
        """
        Generates a ControlWindowFunction with an instant on-off behavior.
        """
        timeparams = ControlTimeParameters(
            start_time=start_time,
            start_end_time=start_time,
            shutdown_start_time=end_time,
            end_time=end_time
        )
        return cls(
            startup=lambda _: 0.0,
            middle=middle,
            shutdown=lambda _: 0.0,
            control_times=timeparams
        )
        