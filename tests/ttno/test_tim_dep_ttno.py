"""
This module tests the TimeDependentTTNO class and its supporting classes.
and functions.
"""
import unittest
from copy import deepcopy
from math import sin, pi

import numpy as np

from pytreenet.random.random_matrices import crandn
from pytreenet.random.random_ttno import generate_three_layer_ttno
from pytreenet.ttno.time_dep_ttno import (TimeDependentTTNO,
                                          FactorUpdatable,
                                          DiscreetTimeTTNO)

class TestFactorUpdatable(unittest.TestCase):
    """
    Tests the FactorUpdatable class.
    """

    def setUp(self):
        self.node_id = "node_1"
        self.indices = (0,2,3,slice(None), slice(None))
        shape = (3, 4, 5, 2, 2)
        self.initial_values = crandn(shape)
        self.factor_function = sin
        self.factor_updatable = FactorUpdatable(self.node_id,
                                                self.indices,
                                                deepcopy(self.initial_values),
                                                self.factor_function)
    def test_init(self):
        """
        Tests a simple initialization of the FactorUpdatable class and
        its getter methods.
        """
        self.assertEqual(0, self.factor_updatable.current_time)
        self.assertEqual(self.node_id, self.factor_updatable.node_id)
        self.assertEqual((slice(0,1),slice(2,3),slice(3,4),slice(None),slice(None)),
                         self.factor_updatable.indices)
        np.testing.assert_array_equal(self.initial_values,
                                      self.factor_updatable.initial_values)
        np.testing.assert_array_equal(self.factor_function(0) * self.initial_values,
                                        self.factor_updatable.current_values)
        self.assertEqual(self.factor_function,
                         self.factor_updatable.factor_function)

    def test_update_current_time(self):
        """
        Tests the update_current_time method of the FactorUpdatable class.
        """
        self.factor_updatable.update_current_time(0.1)
        self.assertEqual(0.1, self.factor_updatable.current_time)

    def test_get_zero_time_values(self):
        """
        Tests the get_zero_time_values method of the FactorUpdatable class.
        """
        np.testing.assert_array_equal(self.factor_updatable.get_zero_time_values(),
                                      self.factor_function(0) * self.initial_values)

    def test_update(self):
        """
        Tests the update method of the FactorUpdatable class.
        """
        self.factor_updatable.update(0.1)
        np.testing.assert_array_equal(self.factor_updatable.current_values,
                                      self.factor_function(0.1) * self.initial_values)
        self.assertEqual(self.factor_updatable.current_time, 0.1)

    def test_update_with_pihalf(self):
        """
        As sin(pi/2) = 1, the current values should be equal to the initial values.
        """
        self.factor_updatable.update(np.pi/2)
        np.testing.assert_array_equal(self.initial_values,
                                      self.factor_updatable.current_values)

class TestTimeDependentTTNOSinupdate(unittest.TestCase):
    """
    Tests the TimeDependentTTNO class with a sin update function.
    """

    def setUp(self):
        self.ttno = generate_three_layer_ttno()
        self.factor_function = sin
        self.time_step_size = pi / 2
        # Generate updatables
        ## 1
        node_id = "child_ly10"
        self.indices1 = (slice(0,1), slice(3,4), slice(None), slice(None))
        shape = (1, 1, 5, 5)
        self.initial_values1 = crandn(shape)
        factor_updatable1 = FactorUpdatable(node_id, self.indices1,
                                                self.initial_values1,
                                                self.factor_function)
        ## 2
        node_id = "child_ly12"
        self.indices2 = (slice(1,2), slice(None), slice(None))
        shape = (1, 2, 2)
        self.initial_values2 = np.asarray([[10,20],[30,40]]).reshape(1,2,2) #crandn(shape)
        factor_updatable2 = FactorUpdatable(node_id, self.indices2,
                                                self.initial_values2,
                                                self.factor_function)
        # Create the TimeDependentTTNO object
        self.updatables = [factor_updatable1, factor_updatable2]
        self.time_dep_ttno = TimeDependentTTNO(self.updatables,
                                               deepcopy(self.ttno))

    def test_init(self):
        """
        After the initilization, the time dependent TTNO should be equal to the
        original TTNO, as sin(0) = 0.
        """
        self.assertEqual(self.time_dep_ttno, self.ttno)

    def test_ttno_is_changed(self):
        """
        Time stepping should change the TTNO.
        """
        self.time_dep_ttno.update(self.time_step_size)
        # Check that the time dependent TTNO is not equal to the original TTNO
        self.assertNotEqual(self.time_dep_ttno, self.ttno)

    def test_reset(self):
        """
        By time stepping and resetting the time dependent TTNO, it should be equal
        to the original TTNO.
        """
        self.time_dep_ttno.update(self.time_step_size)
        self.time_dep_ttno.reset_tensors()
        # Check that the time dependent TTNO is equal to the original TTNO
        self.assertEqual(self.time_dep_ttno, self.ttno)

    def test_sin_update(self):
        """
        As sin(pi/2) = 1, the current TTNO should be equal to the original TTNO
        plus the initial values of the updatables.
        """
        self.time_dep_ttno.update(self.time_step_size)
        ## We know which tensors are changed and they are changed to the initial values
        # Tensor 1
        tensor1 = self.ttno.tensors["child_ly10"][self.indices1]
        correct1 = tensor1 + self.initial_values1 # As factor is just 1
        found = self.time_dep_ttno.tensors["child_ly10"][self.indices1]
        np.testing.assert_allclose(correct1, found)
        # Tensor 2
        tensor2 = self.ttno.tensors["child_ly12"][self.indices2]
        correct2 = tensor2 + self.initial_values2
        found = self.time_dep_ttno.tensors["child_ly12"][self.indices2]
        np.testing.assert_allclose(correct2, found)

    def test_two_steps(self):
        """
        As sin(pi) = 0, the current TTNO should be equal to the original TTNO.
        """
        self.time_dep_ttno.update(self.time_step_size)
        self.time_dep_ttno.update(self.time_step_size)
        # Check that the time dependent TTNO is equal to the original TTNO
        self.assertEqual(self.time_dep_ttno, self.ttno)

class TestTimeDependentTTNODoubleUpdate(unittest.TestCase):
    """
    Tests that two updatales acting on the same part of the tensor add up.
    """

    def setUp(self):
        self.ttno = generate_three_layer_ttno()
        self.time_step_size = 1
        # Generate updatables
        node_id = "child_ly10"
        self.indices1 = (slice(0,1), slice(3,4), slice(None), slice(None))
        shape = (1, 1, 5, 5)
        self.initial_values1 = crandn(shape)
        ## 1
        self.factor_function1 = lambda t: t
        factor_updatable1 = FactorUpdatable(node_id, self.indices1,
                                                self.initial_values1,
                                                self.factor_function1)
        ## 2
        self.factor_function2 = lambda t: 2 * t
        factor_updatable2 = FactorUpdatable(node_id, self.indices1,
                                                self.initial_values1,
                                                self.factor_function2)
        # Create the TimeDependentTTNO object
        self.updatables = [factor_updatable1, factor_updatable2]
        self.time_dep_ttno = TimeDependentTTNO(self.updatables,
                                               deepcopy(self.ttno))

    def test_doubleupdate(self):
        """
        Tests that two updatables acting on the same part of the tensor add up.
        """
        self.time_dep_ttno.update(self.time_step_size)
        # Check that the updated tensor has 3 times the initial values
        tensor = self.ttno.tensors["child_ly10"][self.indices1]
        correct = tensor + 3 * self.initial_values1
        found = self.time_dep_ttno.tensors["child_ly10"][self.indices1]
        np.testing.assert_allclose(correct, found)

class TestDiscreetTimeDepTTNO(unittest.TestCase):
    """
    Test the discreet time dependent TTNO.
    """

    def setUp(self) -> None:
        self.ttnos = [generate_three_layer_ttno()
                      for _ in range(3)]
        self.dt = 0.1
        self.discreet_ttno = DiscreetTimeTTNO(deepcopy(self.ttnos),
                                              self.dt)

    def test_init(self):
        """
        Test the initialisation.
        """
        self.assertEqual(self.discreet_ttno,
                         self.ttnos[0])
        for i, ttno in enumerate(self.ttnos):
            self.assertEqual(ttno, self.discreet_ttno.ttnos[i])
        self.assertEqual(0,self.discreet_ttno.current_time_step)
        self.assertEqual(0,self.discreet_ttno.current_time)
        self.assertEqual(self.dt,self.discreet_ttno.dt)
        self.assertTrue(isinstance(self.discreet_ttno, DiscreetTimeTTNO))

    def test_update_no_change(self):
        """
        Test the that the update method does not change the TTNO, if the time
        step is too small.
        """
        time_step = 0.05
        self.discreet_ttno.update(time_step)
        self.assertEqual(self.ttnos[0], self.discreet_ttno)
        self.assertEqual(0,self.discreet_ttno.current_time_step)
        self.assertEqual(0.05,self.discreet_ttno.current_time)

    def test_update_exacthit(self):
        """
        Test the that the update method changes the TTNO, if the time step is
        exactly equal to the dt.
        """
        time_step = self.dt
        self.discreet_ttno.update(time_step)
        self.assertNotEqual(self.ttnos[0], self.discreet_ttno)
        self.assertEqual(self.ttnos[1], self.discreet_ttno)
        self.assertEqual(1,self.discreet_ttno.current_time_step)
        self.assertEqual(self.dt,self.discreet_ttno.current_time)

    def test_update_largerthandt(self):
        """
        Test the that the update method changes the TTNO, if the time step is
        larger than the dt.
        """
        time_step = 0.12
        self.discreet_ttno.update(time_step)
        self.assertNotEqual(self.ttnos[0], self.discreet_ttno)
        self.assertEqual(self.ttnos[1], self.discreet_ttno)
        self.assertEqual(1,self.discreet_ttno.current_time_step)
        self.assertEqual(time_step,self.discreet_ttno.current_time)

    def test_update_multiple_steps(self):
        """
        Test the that the update method changes the TTNO, even if the time
        steps do not exactly match the dt.
        """
        time_step = 0.075
        self.discreet_ttno.update(time_step)
        self.assertEqual(self.ttnos[0], self.discreet_ttno)
        self.assertEqual(0,self.discreet_ttno.current_time_step)
        self.assertEqual(time_step,self.discreet_ttno.current_time)
        self.discreet_ttno.update(time_step)
        self.assertNotEqual(self.ttnos[0], self.discreet_ttno)
        self.assertEqual(self.ttnos[1], self.discreet_ttno)
        self.assertEqual(1,self.discreet_ttno.current_time_step)
        self.assertEqual(2*time_step,self.discreet_ttno.current_time)

    def test_reset(self):
        """
        Test the reset method.
        """
        time_step = self.dt
        self.discreet_ttno.update(time_step)
        self.discreet_ttno.reset()
        self.assertEqual(self.ttnos[0], self.discreet_ttno)
        self.assertEqual(0,self.discreet_ttno.current_time_step)
        self.assertEqual(0,self.discreet_ttno.current_time)

if __name__ == "__main__":
    unittest.main()
