"""
Tests for the crandn module in pytreenet.util.

This module provides comprehensive unit tests for:
- RandomDistribution enum and its generating_function method
- crandn function for generating complex random arrays
- crandn_like function for generating complex arrays matching input shape
"""

import unittest

import numpy as np
import numpy.random as npr
import numpy.testing as npt

from pytreenet.util.crandn import (
    RandomDistribution,
    crandn,
    crandn_like
)


class TestRandomDistribution(unittest.TestCase):
    """Tests for the RandomDistribution enum."""

    def test_enum_values(self):
        """Test that enum values are correctly defined."""
        self.assertEqual(RandomDistribution.NORMAL.value, "normal")
        self.assertEqual(RandomDistribution.UNIFORM.value, "uniform")

    def test_enum_members(self):
        """Test that enum has expected members."""
        members = list(RandomDistribution)
        self.assertEqual(len(members), 2)
        self.assertIn(RandomDistribution.NORMAL, members)
        self.assertIn(RandomDistribution.UNIFORM, members)

    def test_generating_function_normal(self):
        """Test that NORMAL distribution returns rng.normal."""
        rng = npr.default_rng(42)
        gen_func = RandomDistribution.NORMAL.generating_function(rng)
        # Check that it's the same function by comparing the function names
        self.assertEqual(gen_func.__name__, 'normal')
        # Alternatively, check that both are bound to the same rng object
        self.assertIs(gen_func.__self__, rng)

    def test_generating_function_uniform(self):
        """Test that UNIFORM distribution returns rng.uniform."""
        rng = npr.default_rng(42)
        gen_func = RandomDistribution.UNIFORM.generating_function(rng)
        # Check that it's the same function by comparing the function names
        self.assertEqual(gen_func.__name__, 'uniform')
        # Check that both are bound to the same rng object
        self.assertIs(gen_func.__self__, rng)

    def test_generating_function_returns_callable(self):
        """Test that generating_function returns a callable."""
        rng = npr.default_rng(42)
        for dist in RandomDistribution:
            gen_func = dist.generating_function(rng)
            self.assertTrue(callable(gen_func))

    def test_generating_function_with_different_rng(self):
        """Test generating_function with different RNG instances."""
        rng1 = npr.default_rng(42)
        rng2 = npr.default_rng(123)
        
        func1 = RandomDistribution.NORMAL.generating_function(rng1)
        func2 = RandomDistribution.NORMAL.generating_function(rng2)
        
        # Should be different objects since they come from different RNGs
        self.assertIsNot(func1, func2)


class Test_crandn_BasicFunctionality(unittest.TestCase):
    """Tests for basic crandn functionality."""

    def test_output_dtype_is_complex(self):
        """Test that output is complex dtype."""
        result = crandn(10)
        self.assertTrue(np.iscomplexobj(result))
        self.assertEqual(result.dtype, np.complex128)

    def test_output_shape_single_int(self):
        """Test output shape with single integer size."""
        result = crandn(10)
        self.assertEqual(result.shape, (10,))

    def test_output_shape_tuple(self):
        """Test output shape with tuple size."""
        result = crandn((5, 10))
        self.assertEqual(result.shape, (5, 10))

    def test_output_shape_multidimensional(self):
        """Test output shape with multidimensional tuple."""
        result = crandn((3, 4, 5))
        self.assertEqual(result.shape, (3, 4, 5))

    def test_output_shape_with_args(self):
        """Test output shape with additional args."""
        result = crandn(5, 10, 3)
        self.assertEqual(result.shape, (5, 10, 3))

    def test_output_shape_single_int_with_args(self):
        """Test output shape with single int and additional args."""
        result = crandn(5, 10)
        self.assertEqual(result.shape, (5, 10))

    def test_empty_array(self):
        """Test generation of empty array."""
        result = crandn(0)
        self.assertEqual(result.shape, (0,))
        self.assertEqual(len(result), 0)

    def test_empty_multidimensional(self):
        """Test generation of empty multidimensional array."""
        result = crandn((0, 5))
        self.assertEqual(result.shape, (0, 5))


class Test_crandn_ArgumentValidation(unittest.TestCase):
    """Tests for argument validation in crandn."""

    def test_tuple_in_args_raises_error(self):
        """Test that passing a tuple in *args raises ValueError."""
        with self.assertRaises(ValueError) as context:
            crandn(5, (10, 20))
        self.assertIn("All additional dimensions (*args) must be integers", str(context.exception))
        self.assertIn("tuple", str(context.exception))

    def test_mixed_int_tuple_args_raises_error(self):
        """Test that mixing integers and tuples in *args raises ValueError."""
        with self.assertRaises(ValueError) as context:
            crandn(5, 10, (20, 30))
        self.assertIn("All additional dimensions (*args) must be integers", str(context.exception))
        self.assertIn("position 1", str(context.exception))
        self.assertIn("tuple", str(context.exception))

    def test_string_in_args_raises_error(self):
        """Test that passing a string in *args raises ValueError."""
        with self.assertRaises(ValueError) as context:
            crandn(5, "invalid")
        self.assertIn("All additional dimensions (*args) must be integers", str(context.exception))
        self.assertIn("position 0", str(context.exception))
        self.assertIn("str", str(context.exception))

    def test_float_in_args_raises_error(self):
        """Test that passing a float in *args raises ValueError."""
        with self.assertRaises(ValueError) as context:
            crandn(5, 3.14)
        self.assertIn("All additional dimensions (*args) must be integers", str(context.exception))
        self.assertIn("position 0", str(context.exception))
        self.assertIn("float", str(context.exception))

    def test_none_in_args_raises_error(self):
        """Test that passing None in *args raises ValueError."""
        with self.assertRaises(ValueError) as context:
            crandn(5, None)
        self.assertIn("All additional dimensions (*args) must be integers", str(context.exception))
        self.assertIn("position 0", str(context.exception))
        self.assertIn("NoneType", str(context.exception))

    def test_valid_int_args_no_error(self):
        """Test that valid integer args do not raise errors."""
        # This should work fine
        result = crandn(5, 10, 3)
        self.assertEqual(result.shape, (5, 10, 3))

    def test_tuple_size_with_int_args_no_error(self):
        """Test that tuple size with valid int args works."""
        # When size is a tuple and args are integers, only size is used
        result = crandn((5, 10), 3)
        # Based on current implementation, args are ignored when size is tuple
        self.assertEqual(result.shape, (5, 10))


class Test_crandn_Seeding(unittest.TestCase):
    """Tests for crandn seeding behavior."""

    def test_same_seed_produces_same_output(self):
        """Test that same seed produces same output."""
        result1 = crandn(10, seed=42)
        result2 = crandn(10, seed=42)
        npt.assert_array_equal(result1, result2)

    def test_different_seed_produces_different_output(self):
        """Test that different seeds produce different outputs."""
        result1 = crandn(10, seed=42)
        result2 = crandn(10, seed=123)
        self.assertFalse(np.array_equal(result1, result2))

    def test_no_seed_produces_random_output(self):
        """Test that no seed produces different outputs on different calls."""
        # This might rarely fail due to random chance, but very unlikely
        result1 = crandn(10)
        result2 = crandn(10)
        # We check that they're not exactly equal (extremely unlikely for random)
        self.assertFalse(np.array_equal(result1, result2))

    def test_seed_none_behavior(self):
        """Test that seed=None behaves as expected."""
        result1 = crandn(5, seed=None)
        result2 = crandn(5, seed=None)
        # With seed=None, each call should use a different RNG state
        # So results should be different (extremely unlikely to be the same)
        self.assertFalse(np.array_equal(result1, result2))


class Test_crandn_Distributions(unittest.TestCase):
    """Tests for different distribution types and their statistical properties in crandn."""

    def test_distribution_string_value(self):
        """Test that distribution enum values are strings."""
        self.assertIsInstance(RandomDistribution.NORMAL.value, str)
        self.assertIsInstance(RandomDistribution.UNIFORM.value, str)

    def test_normal_distribution_default(self):
        """Test that NORMAL is the default distribution."""
        result = crandn(1000, seed=42)
        # For standard complex normal, the real and imaginary parts
        # should have mean ~0 and std ~0.5 (since we divide by sqrt(2))
        # With 1000 samples, we use a more lenient tolerance
        self.assertAlmostEqual(np.mean(result.real), 0, delta=0.1)
        self.assertAlmostEqual(np.mean(result.imag), 0, delta=0.1)
        self.assertAlmostEqual(np.std(result.real), 1/np.sqrt(2), delta=0.15)
        self.assertAlmostEqual(np.std(result.imag), 1/np.sqrt(2), delta=0.15)

    def test_normal_distribution_explicit(self):
        """Test explicit NORMAL distribution."""
        result = crandn(1000, seed=42, distribution=RandomDistribution.NORMAL)
        self.assertAlmostEqual(np.mean(result.real), 0, delta=0.1)
        self.assertAlmostEqual(np.mean(result.imag), 0, delta=0.1)

    def test_normal_distribution_mean_zero(self):
        """Test that complex normal has mean near zero."""
        # Use a larger sample for more accurate statistical testing
        result = crandn(10000, seed=42)
        mean = np.mean(result)
        # Mean should be very close to 0
        self.assertAlmostEqual(mean.real, 0, places=1)
        self.assertAlmostEqual(mean.imag, 0, places=1)

    def test_normal_distribution_variance(self):
        """Test that complex normal has correct variance."""
        # For standard complex normal: real and imag ~ N(0, 0.5)
        # because we divide by sqrt(2): N(0,1)/sqrt(2) = N(0, 0.5)
        result = crandn(10000, seed=42)
        var_real = np.var(result.real)
        var_imag = np.var(result.imag)
        expected_var = 0.5
        self.assertAlmostEqual(var_real, expected_var, places=1)
        self.assertAlmostEqual(var_imag, expected_var, places=1)

    def test_normal_distribution_independent_components(self):
        """Test that real and imaginary parts are independent."""
        # For large sample, correlation between real and imag should be near 0
        result = crandn(10000, seed=42)
        correlation = np.corrcoef(result.real, result.imag)[0, 1]
        self.assertAlmostEqual(correlation, 0, places=1)

    def test_normal_with_loc(self):
        """Test NORMAL distribution with loc parameter."""
        loc = 5.0
        result = crandn(1000, distribution=RandomDistribution.NORMAL, loc=loc)
        # Mean should be around loc / sqrt(2) for real and imag parts
        # Actually, (real + i*imag)/sqrt(2) where real~N(loc,1), imag~N(loc,1)
        # So mean should be around (loc + i*loc)/sqrt(2) = loc*(1+i)/sqrt(2)
        expected_mean_real = loc / np.sqrt(2)
        expected_mean_imag = loc / np.sqrt(2)
        self.assertAlmostEqual(np.mean(result.real), expected_mean_real, places=0)
        self.assertAlmostEqual(np.mean(result.imag), expected_mean_imag, places=0)

    def test_normal_with_scale(self):
        """Test NORMAL distribution with scale parameter."""
        scale = 2.0
        result = crandn(10000, distribution=RandomDistribution.NORMAL, scale=scale, seed=42)
        # Std should be around scale / sqrt(2) for real and imag parts
        expected_std = scale / np.sqrt(2)
        # Use a larger sample and more lenient tolerance due to statistical variance
        self.assertAlmostEqual(np.std(result.real), expected_std, places=0)
        self.assertAlmostEqual(np.std(result.imag), expected_std, places=0)

    def test_complex_values_are_complex(self):
        """Test that all generated values are complex."""
        result = crandn(100, seed=42)
        self.assertTrue(np.all(np.iscomplex(result)))
        # Check that imaginary parts are non-zero (for normal distribution)
        # Note: there's a tiny chance this could fail, but very unlikely
        self.assertTrue(np.any(result.imag != 0))

    def test_uniform_distribution(self):
        """Test UNIFORM distribution."""
        # For uniform distribution with default parameters (0,1)
        # The result should be in range [0, 1/sqrt(2)] + i[0, 1/sqrt(2)]
        result = crandn(1000, distribution=RandomDistribution.UNIFORM)
        
        # All values should be complex
        self.assertTrue(np.all(np.iscomplex(result)))
        
        # Real and imaginary parts should be in [0, 1/sqrt(2)]
        max_expected = 1/np.sqrt(2)
        self.assertTrue(np.all(result.real >= 0))
        self.assertTrue(np.all(result.real <= max_expected))
        self.assertTrue(np.all(result.imag >= 0))
        self.assertTrue(np.all(result.imag <= max_expected))

    def test_uniform_distribution_with_kwargs(self):
        """Test UNIFORM distribution with custom parameters."""
        # Test with custom low and high values
        result = crandn(1000, distribution=RandomDistribution.UNIFORM, low=-1, high=1)
        
        # The result should be (uniform(-1,1) + i*uniform(-1,1)) / sqrt(2)
        # So each component should be in [-1/sqrt(2), 1/sqrt(2)]
        max_expected = 1/np.sqrt(2)
        self.assertTrue(np.all(result.real >= -max_expected))
        self.assertTrue(np.all(result.real <= max_expected))
        self.assertTrue(np.all(result.imag >= -max_expected))
        self.assertTrue(np.all(result.imag <= max_expected))

    def test_uniform_with_custom_bounds(self):
        """Test UNIFORM distribution with custom bounds."""
        low, high = -2.0, 3.0
        result = crandn(1000, distribution=RandomDistribution.UNIFORM, low=low, high=high)
        
        # Each component should be in [low/sqrt(2), high/sqrt(2)]
        scale = 1 / np.sqrt(2)
        self.assertTrue(np.all(result.real >= low * scale))
        self.assertTrue(np.all(result.real <= high * scale))
        self.assertTrue(np.all(result.imag >= low * scale))
        self.assertTrue(np.all(result.imag <= high * scale))


class Test_crandn_EdgeCases(unittest.TestCase):
    """Tests for edge cases in crandn."""

    def test_size_one(self):
        """Test generation of single element."""
        result = crandn(1)
        self.assertEqual(result.shape, (1,))
        self.assertTrue(np.iscomplexobj(result))

    def test_scalar_size_with_args(self):
        """Test scalar size with multiple args."""
        result = crandn(1, 1, 1)
        self.assertEqual(result.shape, (1, 1, 1))

    def test_tuple_size_with_args(self):
        """Test tuple size with additional args - args are ignored when size is tuple."""
        # When size is a tuple, additional args are currently ignored in crandn
        result = crandn((2, 3), 4)
        # Based on current implementation, args are ignored when size is tuple
        self.assertEqual(result.shape, (2, 3))

    def test_large_dimensions(self):
        """Test generation with large dimensions."""
        result = crandn(10, 10, 10, 10)
        self.assertEqual(result.shape, (10, 10, 10, 10))

    def test_very_large_single_dimension(self):
        """Test generation with very large single dimension."""
        result = crandn(1000000)
        self.assertEqual(result.shape, (1000000,))


class Test_crandn_like(unittest.TestCase):
    """Tests for crandn_like function."""

    def test_basic_functionality(self):
        """Test basic crandn_like functionality."""
        array = np.zeros((5, 10))
        result = crandn_like(array)
        self.assertEqual(result.shape, (5, 10))
        self.assertTrue(np.iscomplexobj(result))

    def test_matches_input_shape(self):
        """Test that output shape matches input array shape."""
        for shape in [(5,), (3, 4), (2, 3, 4), (10, 10, 10)]:
            array = np.zeros(shape)
            result = crandn_like(array)
            self.assertEqual(result.shape, shape)

    def test_with_seed(self):
        """Test crandn_like with seed parameter."""
        array = np.zeros((5, 5))
        result1 = crandn_like(array, seed=42)
        result2 = crandn_like(array, seed=42)
        npt.assert_array_equal(result1, result2)

    def test_with_different_seeds(self):
        """Test crandn_like with different seeds."""
        array = np.zeros((5, 5))
        result1 = crandn_like(array, seed=42)
        result2 = crandn_like(array, seed=123)
        self.assertFalse(np.array_equal(result1, result2))

    def test_with_distribution(self):
        """Test crandn_like with different distribution."""
        array = np.zeros((10, 10))
        result = crandn_like(array, distribution=RandomDistribution.UNIFORM)
        self.assertEqual(result.shape, (10, 10))
        # Check that values are in expected range for uniform
        self.assertTrue(np.all(result.real >= 0))
        self.assertTrue(np.all(result.imag >= 0))

    def test_with_args(self):
        """Test crandn_like with additional args."""
        array = np.zeros((5, 5))
        # Additional args are passed to crandn, but since array.shape is a tuple,
        # they will be ignored (same behavior as crandn with tuple size)
        result = crandn_like(array, 3)
        self.assertEqual(result.shape, (5, 5))

    def test_with_kwargs(self):
        """Test crandn_like with additional kwargs."""
        array = np.zeros((10, 10))
        result = crandn_like(array, distribution=RandomDistribution.NORMAL, loc=1.0)
        self.assertEqual(result.shape, (10, 10))
        # Mean should be affected by loc parameter
        self.assertAlmostEqual(np.mean(result.real), 1/np.sqrt(2), places=0)

    def test_complex_input_array(self):
        """Test crandn_like with complex input array."""
        array = np.array([1+2j, 3+4j, 5+6j])
        result = crandn_like(array)
        self.assertEqual(result.shape, (3,))
        self.assertTrue(np.iscomplexobj(result))

    def test_empty_input_array(self):
        """Test crandn_like with empty array."""
        array = np.array([])
        result = crandn_like(array)
        self.assertEqual(result.shape, (0,))

    def test_zero_shape_array(self):
        """Test crandn_like with zero in shape."""
        array = np.zeros((0, 5))
        result = crandn_like(array)
        self.assertEqual(result.shape, (0, 5))


class Test_crandn_Normalization(unittest.TestCase):
    """Tests for the normalization factor in crandn."""

    def test_normalization_factor(self):
        """Test that normalization factor is applied correctly."""
        # For standard complex normal, we expect:
        # result = (normal() + i*normal()) / sqrt(2)
        # So the variance of each component should be 1/2
        result = crandn(10000, seed=42)
        var_real = np.var(result.real, ddof=1)  # Sample variance
        var_imag = np.var(result.imag, ddof=1)
        expected_var = 1.0 / 2.0
        self.assertAlmostEqual(var_real, expected_var, places=1)
        self.assertAlmostEqual(var_imag, expected_var, places=1)

    def test_normalization_consistency(self):
        """Test that normalization is consistent across different sizes."""
        # Generate multiple arrays and check that variance is consistent
        for size in [100, 1000, 10000]:
            result = crandn(size, seed=42)
            var_real = np.var(result.real, ddof=1)
            var_imag = np.var(result.imag, ddof=1)
            expected_var = 1.0 / 2.0
            self.assertAlmostEqual(var_real, expected_var, places=0)
            self.assertAlmostEqual(var_imag, expected_var, places=0)


class Test_crandn_TypeConsistency(unittest.TestCase):
    """Tests for type consistency in crandn output."""

    def test_always_returns_ndarray(self):
        """Test that crandn always returns numpy array."""
        result = crandn(10)
        self.assertIsInstance(result, np.ndarray)

    def test_always_complex_dtype(self):
        """Test that output always has complex dtype."""
        for size in [1, 10, (5, 5), (2, 3, 4)]:
            result = crandn(size)
            self.assertTrue(np.iscomplexobj(result))

    def test_dtype_consistency_across_sizes(self):
        """Test dtype consistency across different sizes."""
        result1 = crandn(10)
        result2 = crandn((10, 10))
        result3 = crandn(5, 5, 5)
        self.assertEqual(result1.dtype, result2.dtype)
        self.assertEqual(result2.dtype, result3.dtype)


class Test_crandn_Integration(unittest.TestCase):
    """Integration tests for crandn functions."""

    def test_crandn_and_crandn_like_consistency(self):
        """Test that crandn_like produces same results as crandn with same shape."""
        shape = (5, 10)
        array = np.zeros(shape)
        
        # Use same seed for both
        result1 = crandn(shape, seed=42)
        result2 = crandn_like(array, seed=42)
        
        npt.assert_array_equal(result1, result2)

    def test_multiple_calls_with_same_seed(self):
        """Test multiple calls with same seed produce identical results."""
        seeds = [42, 123, 456]
        size = (10, 10)
        
        for seed in seeds:
            results = [crandn(size, seed=seed) for _ in range(5)]
            for i in range(1, len(results)):
                npt.assert_array_equal(results[0], results[i])

    def test_workflow_with_real_array(self):
        """Test a typical workflow with real arrays."""
        # Create a real array
        real_array = np.random.random((10, 20))
        
        # Generate complex array with same shape
        complex_array = crandn_like(real_array)
        
        # Verify properties
        self.assertEqual(complex_array.shape, real_array.shape)
        self.assertTrue(np.iscomplexobj(complex_array))
        self.assertFalse(np.isrealobj(complex_array))

    def test_chaining_with_different_distributions(self):
        """Test chaining calls with different distributions."""
        shape = (5, 5)
        
        normal_result = crandn(shape, seed=42, distribution=RandomDistribution.NORMAL)
        uniform_result = crandn(shape, seed=42, distribution=RandomDistribution.UNIFORM)
        
        # They should have different values (different distributions)
        self.assertFalse(np.array_equal(normal_result, uniform_result))
        
        # But same shape
        self.assertEqual(normal_result.shape, uniform_result.shape)

if __name__ == '__main__':
    unittest.main()
