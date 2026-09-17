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

    def test_linear_startup_zero_window(self) -> None:
        """
        Test the linear_startup method with a zero startup time window.
        """
        params = ControlTimeParameters(
            start_time=0.0,
            start_end_time=0.0,
            shutdown_start_time=2.0,
            end_time=4.0
        )
        found = params.linear_startup(end_value=3.0)
        times = [0.0, 0.5, 1.0]
        for t in times:
            self.assertEqual(found(t), 0.0)

    def test_linear_startup_callable(self) -> None:
        """
        Test the linear_startup method with a callable end_value.
        """
        found = self.params.linear_startup(end_value=lambda t: 2.0 * t)
        def correct(t: float) -> float:
            return 2.0 * t
        times = [0.0, 0.5, 1.0]
        for t in times:
            self.assertEqual(found(t), correct(t))

    def test_linear_shutdown(self) -> None:
        """
        Test the linear_shutdown method.
        """
        found = self.params.linear_shutdown(initial_value=3.0)
        def correct(t: float) -> float:
            return 3.0 * (4.0 - t) / 2.0
        times = [2.0, 2.5, 3.0, 3.5, 4.0]
        for t in times:
            self.assertEqual(found(t), correct(t))

    def test_linear_shutdown_zero_window(self) -> None:
        """
        Test the linear_shutdown method with a zero shutdown time window.
        """
        params = ControlTimeParameters(
            start_time=0.0,
            start_end_time=1.0,
            shutdown_start_time=2.0,
            end_time=2.0
        )
        found = params.linear_shutdown(initial_value=3.0)
        times = [2.0, 2.5, 3.0, 3.5, 4.0]
        for t in times:
            self.assertEqual(found(t), 0.0)

    def test_linear_shutdown_callable(self) -> None:
        """
        Test the linear_shutdown method with a callable initial_value.
        """
        found = self.params.linear_shutdown(initial_value=lambda t: 3.0 / 2.0 * t)
        def correct(t: float) -> float:
            return (t -2.0) * -1.5 + 3.0
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

    def test_trig_startup_zero_window(self) -> None:
        """
        Test the trig_startup method with a zero startup time window.
        """
        params = ControlTimeParameters(
            start_time=0.0,
            start_end_time=0.0,
            shutdown_start_time=2.0,
            end_time=4.0
        )
        found = params.trig_startup(end_value=3.0)
        times = [0.0, 0.5, 1.0]
        for t in times:
            self.assertEqual(found(t), 0.0)

    def test_trig_startup_callable(self) -> None:
        """
        Test the trig_startup method with a callable end_value.
        """
        found = self.params.trig_startup(end_value=lambda t: 2.0 * t)
        def correct(t: float) -> float:
            return 2.0 * t
        times = [0.0, 0.5, 1.0]
        for t in times:
            self.assertAlmostEqual(found(t), correct(t))

    def test_trig_shutdown(self) -> None:
        """
        Test the trig_shutdown method.
        """
        found = self.params.trig_shutdown(initial_value=3.0)
        def correct(t: float) -> float:
            return 3.0 * (cos(pi / 2 * (t-2)/ 2)) ** 2
        times = [2.0, 2.5, 3.0, 3.5, 4.0]
        for t in times:
            self.assertEqual(found(t), correct(t))

    def test_trig_shutdown_zero_window(self) -> None:
        """
        Test the trig_shutdown method with a zero shutdown time window.
        """
        params = ControlTimeParameters(
            start_time=0.0,
            start_end_time=1.0,
            shutdown_start_time=2.0,
            end_time=2.0
        )
        found = params.trig_shutdown(initial_value=3.0)
        times = [2.0, 2.5, 3.0, 3.5, 4.0]
        for t in times:
            self.assertEqual(found(t), 0.0)

    def test_trig_shutdown_callable(self) -> None:
        """
        Test the trig_shutdown method with a callable initial_value.
        """
        found = self.params.trig_shutdown(initial_value=lambda t: 3.0 / 2.0 * t)
        def correct(t: float) -> float:
            return 3.0 * (cos(pi / 2 * (t-2)/ 2)) ** 2
        times = [2.0, 2.5, 3.0, 3.5, 4.0]
        for t in times:
            self.assertAlmostEqual(found(t), correct(t))

    def test_start_greater_startend(self) -> None:
        """
        Test that an error is raised if start_time is greater than start_end_time.
        """
        with self.assertRaises(ValueError):
            ControlTimeParameters(
                start_time=1.0,
                start_end_time=0.0,
                shutdown_start_time=2.0,
                end_time=4.0)

    def test_startend_greater_shutdownstart(self) -> None:
        """
        Test that an error is raised if start_end_time is greater than shutdown_start_time.
        """
        with self.assertRaises(ValueError):
            ControlTimeParameters(
                start_time=0.0,
                start_end_time=3.0,
                shutdown_start_time=2.0,
                end_time=4.0)

    def test_shutdownstart_greater_end(self) -> None:
        """
        Test that an error is raised if shutdown_start_time is greater than end_time.
        """
        with self.assertRaises(ValueError):
            ControlTimeParameters(
                start_time=0.0,
                start_end_time=1.0,
                shutdown_start_time=5.0,
                end_time=4.0)

    def test_all_times_equal(self) -> None:
        """
        Test that no error is raised if all times are equal.
        """
        try:
            ControlTimeParameters(
                start_time=1.0,
                start_end_time=1.0,
                shutdown_start_time=1.0,
                end_time=1.0)
        except ValueError:
            self.fail("ControlTimeParameters raised ValueError unexpectedly!")

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

    def test_constant_middle_classmethod(self) -> None:
        """
        Test the constant_middle class method of ControlWindowFunction.
        """
        params = ControlTimeParameters(
            start_time=0.0,
            start_end_time=1.0,
            shutdown_start_time=2.0,
            end_time=4.0
        )
        functions = ControlWindowFunction.constant_middle(lambda _: 0.5,
                                                          lambda _: 0.75,
                                                          params)
        # Before window
        self.assertEqual(functions(-1.0), 0)
        # Startup phase
        self.assertEqual(functions(0.0), 0.5)
        self.assertEqual(functions(0.5), 0.5)
        # Middle phase
        self.assertEqual(functions(1.0), 1.0)
        self.assertEqual(functions(1.5), 1.0)
        self.assertEqual(functions(2.0), 1.0)
        # Shutdown phase
        self.assertEqual(functions(2.5), 0.75)
        self.assertEqual(functions(3.0), 0.75)
        self.assertEqual(functions(4.0), 0.75)
        # After window
        self.assertEqual(functions(5.0), 0)

    def test_constant_middle_custom_strength(self):
        """
        Test the constant_middle class method of ControlWindowFunction with a
        custom strength.
        """
        params = ControlTimeParameters(
            start_time=0.0,
            start_end_time=1.0,
            shutdown_start_time=2.0,
            end_time=4.0
        )
        functions = ControlWindowFunction.constant_middle(lambda _: 0.5,
                                                          lambda _: 0.75,
                                                          params,
                                                          strength=2.0)
        # Before window
        self.assertEqual(functions(-1.0), 0)
        # Startup phase
        self.assertEqual(functions(0.0), 0.5)
        self.assertEqual(functions(0.5), 0.5)
        # Middle phase
        self.assertEqual(functions(1.0), 2.0)
        self.assertEqual(functions(1.5), 2.0)
        self.assertEqual(functions(2.0), 2.0)
        # Shutdown phase
        self.assertEqual(functions(2.5), 0.75)
        self.assertEqual(functions(3.0), 0.75)
        self.assertEqual(functions(4.0), 0.75)
        # After window
        self.assertEqual(functions(5.0), 0)

    def test_instant_on_off_classmethod(self):
        """
        Test the instant_on_off class method of ControlWindowFunction.
        """
        functions = ControlWindowFunction.instant_on_off(lambda t: 2.0 * t)
        # Before window
        self.assertEqual(functions(-1.0), 0)
        # At Start-Up
        self.assertEqual(functions(0.0), 0)
        # Middle phase
        self.assertEqual(functions(0.2), 2.0*0.2)
        self.assertEqual(functions(0.4), 2.0*0.4)
        self.assertEqual(functions(0.6), 2.0*0.6)
        # At Shutdown
        self.assertEqual(functions(1.0), 1.0*2.0)
        # After window
        self.assertEqual(functions(5.0), 0)

    def test_instant_on_off_classmethod_customstartend(self):
        """
        Test the instant_on_off class method of ControlWindowFunction with a
        custom start and end time.
        """
        functions = ControlWindowFunction.instant_on_off(lambda t: 2.0 * t,
                                                         start_time=1.0,
                                                         end_time=3.0)
        # Before window
        self.assertEqual(functions(0.0), 0)
        # At Start-Up
        self.assertEqual(functions(1.0), 2.0*1.0)
        # Middle phase
        self.assertEqual(functions(1.5), 2.0*1.5)
        self.assertEqual(functions(2.0), 2.0*2.0)
        self.assertEqual(functions(2.5), 2.0*2.5)
        # At Shutdown
        self.assertEqual(functions(3.0), 2.0*3.0)
        # After window
        self.assertEqual(functions(4.0), 0)
