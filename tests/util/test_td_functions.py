"""
This module tests various time dependent functions and classes.
"""
import unittest

from math import pi, sin, cos

from pytreenet.util.td_functions import (ControlTimeParameters,
                                         ControlWindowFunction)

class TestControlTimeParameters(unittest.TestCase):
    """
    Test the ControlTimeParameters class.
    """
    def setUp(self) -> None:
        self.params = ControlTimeParameters(
            start_time=0.0,
            start_end_time=1.0,
            shutdown_start_time=2.0,
            end_time=4.0
        )

    def test_startup_time_window(self) -> None:
        """
        Test the startup_time_window method.
        """
        self.assertEqual(self.params.startup_time_window(), 1.0)

    def test_shutdown_time_window(self) -> None:
        """
        Test the shutdown_time_window method.
        """
        self.assertEqual(self.params.shutdown_time_window(), 2.0)

    def test_linear_startup(self) -> None:
        """
        Test the linear_startup method.
        """
        found = self.params.linear_startup(end_value=3.0)
        def correct(t: float) -> float:
            return 3.0 * t
        times = [0.0, 0.5, 1.0]
        for t in times:
            self.assertEqual(found(t), correct(t))

    def test_linear_shutdown(self) -> None:
        """
        Test the linear_shutdown method.
        """
        found = self.params.linear_shutdown(end_value=3.0)
        def correct(t: float) -> float:
            return 3.0 * (4.0 - t) / 2.0
        times = [2.0, 2.5, 3.0, 3.5, 4.0]
        for t in times:
            self.assertEqual(found(t), correct(t))

    def test_trig_startup(self) -> None:
        """
        Test the trig_startup method.
        """
        found = self.params.trig_startup(end_value=3.0)
        def correct(t: float) -> float:
            return 3.0 * (sin(pi / 2 * t)) ** 2
        times = [0.0, 0.5, 1.0]
        for t in times:
            self.assertEqual(found(t), correct(t))

    def test_trig_shutdown(self) -> None:
        """
        Test the trig_shutdown method.
        """
        found = self.params.trig_shutdown(end_value=3.0)
        def correct(t: float) -> float:
            return 3.0 * (cos(pi / 2 * (t-2)/ 2)) ** 2
        times = [2.0, 2.5, 3.0, 3.5, 4.0]
        for t in times:
            self.assertEqual(found(t), correct(t))

class TestControlWindowFunctionInit(unittest.TestCase):
    """
    Test the ControlWindowFunction class.
    """

    def test_init(self) -> None:
        """
        Test the initialization of the ControlWindowFunction class.
        """
        params = ControlTimeParameters(
            start_time=0.0,
            start_end_time=1.0,
            shutdown_start_time=2.0,
            end_time=4.0
        )
        functions = ControlWindowFunction(lambda _: 0.5,
                                          lambda _: 1,
                                          lambda _: 0.75,
                                          params)
        # Before window
        self.assertEqual(functions(-1.0), 0)
        # Startup phase
        self.assertEqual(functions(0.0), 0.5)
        self.assertEqual(functions(0.5), 0.5)
        # Middle phase
        self.assertEqual(functions(1.0), 1)
        self.assertEqual(functions(1.5), 1)
        self.assertEqual(functions(2.0), 1)
        # Shutdown phase
        self.assertEqual(functions(2.5), 0.75)
        self.assertEqual(functions(3.0), 0.75)
        self.assertEqual(functions(4.0), 0.75)
        # After window
        self.assertEqual(functions(5.0), 0)
    