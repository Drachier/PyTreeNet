"""
Tests for the std_utils module in pytreenet.util.
"""

import unittest

import numpy as np
import numpy.testing as npt
from scipy.linalg import expm

from pytreenet.util.std_utils import (
    copy_object,
    sort_dictionary,
    compare_lists_by_value,
    compare_lists_by_identity,
    permute_iterator,
    find_permutation,
    is_broadcastable,
    int_to_slice,
    fast_exp_action,
    positivise_range,
    average_data,
    identity_mapping,
    inverse_bijective_finite_map
)


class Test_copy_object(unittest.TestCase):
    """Tests for copy_object function."""

    def test_deep_copy_list(self):
        original = [[1, 2], [3, 4]]
        copied = copy_object(original, deep=True)
        self.assertEqual(copied, original)
        self.assertIsNot(copied, original)
        self.assertIsNot(copied[0], original[0])

    def test_shallow_copy_list(self):
        original = [[1, 2], [3, 4]]
        copied = copy_object(original, deep=False)
        self.assertEqual(copied, original)
        self.assertIsNot(copied, original)
        self.assertIs(copied[0], original[0])

    def test_deep_copy_dict(self):
        original = {'a': [1, 2], 'b': [3, 4]}
        copied = copy_object(original, deep=True)
        self.assertEqual(copied, original)
        self.assertIsNot(copied, original)
        self.assertIsNot(copied['a'], original['a'])

    def test_shallow_copy_dict(self):
        original = {'a': [1, 2], 'b': [3, 4]}
        copied = copy_object(original, deep=False)
        self.assertEqual(copied, original)
        self.assertIsNot(copied, original)
        self.assertIs(copied['a'], original['a'])


class Test_sort_dictionary(unittest.TestCase):
    """Tests for sort_dictionary function."""

    def test_sort_by_ascending_values(self):
        d = {'a': 3, 'b': 1, 'c': 2}
        sorted_d = sort_dictionary(d)
        self.assertEqual(list(sorted_d.keys()), ['b', 'c', 'a'])

    def test_sort_empty_dict(self):
        self.assertEqual(sort_dictionary({}), {})

    def test_sort_with_duplicate_values(self):
        d = {'a': 1, 'b': 1, 'c': 0}
        sorted_d = sort_dictionary(d)
        self.assertEqual(list(sorted_d.values()), [0, 1, 1])


class Test_compare_lists_by_value(unittest.TestCase):
    """Tests for compare_lists_by_value function."""

    def test_equal_lists_different_order(self):
        list1 = [1, 2, 3]
        list2 = [3, 2, 1]
        self.assertTrue(compare_lists_by_value(list1, list2))

    def test_equal_lists_same_order(self):
        list1 = [1, 2, 3]
        list2 = [1, 2, 3]
        self.assertTrue(compare_lists_by_value(list1, list2))

    def test_different_lengths(self):
        list1 = [1, 2, 3]
        list2 = [1, 2]
        self.assertFalse(compare_lists_by_value(list1, list2))

    def test_different_elements(self):
        list1 = [1, 2, 3]
        list2 = [1, 2, 4]
        self.assertFalse(compare_lists_by_value(list1, list2))

    def test_with_duplicates(self):
        list1 = [1, 2, 2, 3]
        list2 = [3, 2, 1, 2]
        self.assertTrue(compare_lists_by_value(list1, list2))

    def test_empty_lists(self):
        self.assertTrue(compare_lists_by_value([], []))


class Test_compare_lists_by_identity(unittest.TestCase):
    """Tests for compare_lists_by_identity function."""

    def test_same_objects_same_order(self):
        obj1 = [1, 2]
        obj2 = [3, 4]
        list1 = [obj1, obj2]
        list2 = [obj1, obj2]
        self.assertTrue(compare_lists_by_identity(list1, list2))

    def test_same_values_different_objects(self):
        list1 = [[1, 2], [3, 4]]
        list2 = [[1, 2], [3, 4]]
        self.assertFalse(compare_lists_by_identity(list1, list2))

    def test_different_lengths(self):
        list1 = [object()]
        list2 = []
        self.assertFalse(compare_lists_by_identity(list1, list2))

    def test_different_order(self):
        obj1 = [1, 2]
        obj2 = [3, 4]
        list1 = [obj1, obj2]
        list2 = [obj2, obj1]
        self.assertFalse(compare_lists_by_identity(list1, list2))

    def test_empty_lists(self):
        self.assertTrue(compare_lists_by_identity([], []))


class Test_permute_iterator(unittest.TestCase):
    """Tests for permute_iterator function."""

    def test_permute_iterator(self):
        self.assertEqual(permute_iterator((1, 2, 3), [2, 1, 0]), (3, 2, 1))
        self.assertEqual(permute_iterator((1, 2, 3), [0, 1, 2]), (1, 2, 3))
        self.assertEqual(permute_iterator((1, 2, 3), [0, 2, 1]), (1, 3, 2))

    def test_permute_iterator_empty(self):
        self.assertEqual(permute_iterator((), []), ())

    def test_permute_iterator_single(self):
        self.assertEqual(permute_iterator((1,), [0]), (1,))

    def test_permute_iterator_error(self):
        self.assertRaises(AssertionError, permute_iterator, (1, 2, 3), [0, 1, 2, 3])
        self.assertRaises(AssertionError, permute_iterator, (1, 2, 3), [0, 1])

    def test_permute_iterator_list(self):
        # Test that it works with lists too
        self.assertEqual(list(permute_iterator([1, 2, 3], [2, 1, 0])), [3, 2, 1])


class Test_find_permutation(unittest.TestCase):
    """Tests for find_permutation function."""

    def test_simple_permutation(self):
        list1 = ['a', 'b', 'c']
        list2 = ['c', 'b', 'a']
        perm = find_permutation(list1, list2)
        self.assertEqual(perm, [2, 1, 0])

    def test_identity_permutation(self):
        list1 = [1, 2, 3]
        list2 = [1, 2, 3]
        perm = find_permutation(list1, list2)
        self.assertEqual(perm, [0, 1, 2])

    def test_error_on_different_lengths(self):
        self.assertRaises(AssertionError, find_permutation, [1, 2], [1, 2, 3])


class Test_is_broadcastable(unittest.TestCase):
    """Tests for is_broadcastable function."""

    def test_same_shape(self):
        self.assertTrue(is_broadcastable((3, 4), (3, 4)))

    def test_one_dimension_one(self):
        self.assertTrue(is_broadcastable((1, 4), (3, 4)))
        self.assertTrue(is_broadcastable((3, 4), (1, 4)))
        self.assertTrue(is_broadcastable((3, 1), (3, 4)))
        self.assertTrue(is_broadcastable((3, 4), (3, 1)))

    def test_both_dimensions_one(self):
        self.assertTrue(is_broadcastable((1, 1), (3, 4)))

    def test_not_broadcastable(self):
        self.assertFalse(is_broadcastable((3, 4), (5, 6)))
        self.assertFalse(is_broadcastable((3, 4), (3, 5)))

    def test_different_dimensions(self):
        self.assertTrue(is_broadcastable((1,), (3, 4)))
        self.assertFalse(is_broadcastable((2,), (3, 4)))

    def test_empty_shapes(self):
        self.assertTrue(is_broadcastable((), ()))
        self.assertTrue(is_broadcastable((), (3,)))


class Test_int_to_slice(unittest.TestCase):
    """Tests for int_to_slice function."""

    def test_positive_index(self):
        s = int_to_slice(5)
        self.assertEqual(s, slice(5, 6))

    def test_zero_index(self):
        s = int_to_slice(0)
        self.assertEqual(s, slice(0, 1))

    def test_negative_index(self):
        s = int_to_slice(-3)
        self.assertEqual(s, slice(-3, -2))


class Test_fast_exp_action(unittest.TestCase):
    """Tests for fast_exp_action function."""

    def setUp(self):
        self.matrix = np.array([[0, 1], [-1, 0]])  # Rotation by 90 degrees
        self.vector = np.array([1, 0])
        self.res = expm(self.matrix) @ self.vector  # Expected result using scipy's expm

    def test_expm_mode(self):
        result = fast_exp_action(self.matrix, self.vector, mode="expm")
        self.assertEqual(result.shape, self.vector.shape)
        npt.assert_almost_equal(result, self.res)

    def test_chebyshev_mode(self):
        result = fast_exp_action(self.matrix, self.vector, mode="chebyshev")
        self.assertEqual(result.shape, self.vector.shape)
        npt.assert_almost_equal(result, self.res)

    def test_sparse_mode(self):
        result = fast_exp_action(self.matrix, self.vector, mode="sparse")
        # sparse mode returns a 2D array, just check it has the right number of elements
        self.assertEqual(result.size, self.vector.size)
        npt.assert_almost_equal(result[:,0], self.res)

    def test_none_mode(self):
        result = fast_exp_action(self.matrix, self.vector, mode="none")
        npt.assert_almost_equal(result, self.vector)

    def test_default_mode(self):
        result = fast_exp_action(self.matrix, self.vector)
        self.assertEqual(result.shape, self.vector.shape)
        npt.assert_almost_equal(result, self.res)

    def test_invalid_mode(self):
        self.assertRaises(NotImplementedError, 
                         fast_exp_action, self.matrix, self.vector, mode="invalid")


class Test_positivise_range(unittest.TestCase):
    """Tests for positivise_range function."""

    def test_all_positive(self):
        rng = range(1, 3)
        result = positivise_range(rng, 10)
        self.assertEqual(result, range(1, 3))

    def test_negative_start(self):
        # range(-2, 3) with size=10: start=-2->8, stop=3->3
        rng = range(-2, 3)
        result = positivise_range(rng, 10)
        self.assertEqual(result, range(8, 3))

    def test_negative_stop(self):
        rng = range(1, -2)
        result = positivise_range(rng, 10)
        self.assertEqual(result, range(1, 8))

    def test_both_negative(self):
        rng = range(-3, -1)
        result = positivise_range(rng, 10)
        self.assertEqual(result, range(7, 9))

    def test_with_step(self):
        rng = range(0, 10, 2)
        result = positivise_range(rng, 20)
        self.assertEqual(result, range(0, 10, 2))

    def test_negative_with_step(self):
        # range(-2, 5, 2) with size=10: start=-2->8, stop=5->5
        rng = range(-2, 5, 2)
        result = positivise_range(rng, 10)
        self.assertEqual(result, range(8, 5, 2))


class Test_average_data(unittest.TestCase):
    """Tests for average_data function."""

    def test_simple_average(self):
        data = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
        result = average_data(data)
        np.testing.assert_array_almost_equal(result, [3, 4])

    def test_single_array(self):
        data = [np.array([1, 2, 3])]
        result = average_data(data)
        np.testing.assert_array_almost_equal(result, [1, 2, 3])

    def test_empty_input(self):
        self.assertRaises(ValueError, average_data, [])

    def test_different_shapes(self):
        data = [np.array([1, 2]), np.array([3, 4])]
        result = average_data(data)
        np.testing.assert_array_almost_equal(result, [2, 3])


class Test_identity_mapping(unittest.TestCase):
    """Tests for identity_mapping function."""

    def test_integer(self):
        self.assertEqual(identity_mapping(42), 42)

    def test_string(self):
        self.assertEqual(identity_mapping("hello"), "hello")

    def test_list(self):
        lst = [1, 2, 3]
        self.assertEqual(identity_mapping(lst), lst)

    def test_none(self):
        self.assertIsNone(identity_mapping(None))


class Test_inverse_bijective_finite_map(unittest.TestCase):
    """Tests for inverse_bijective_finite_map function."""

    def test_dict_mapping(self):
        mapping = {'a': 1, 'b': 2, 'c': 3}
        inverse = inverse_bijective_finite_map(mapping)
        self.assertEqual(inverse(1), 'a')
        self.assertEqual(inverse(2), 'b')
        self.assertEqual(inverse(3), 'c')

    def test_dict_mapping_subset(self):
        mapping = {'a': 1, 'b': 2, 'c': 3}
        inverse = inverse_bijective_finite_map(mapping, keys_to_invert=['a', 'b'])
        self.assertEqual(inverse(1), 'a')
        self.assertEqual(inverse(2), 'b')
        self.assertRaises(KeyError, inverse, 3)

    def test_callable_mapping(self):
        def double(x):
            return x * 2
        keys = [1, 2, 3]
        inverse = inverse_bijective_finite_map(double, keys_to_invert=keys)
        self.assertEqual(inverse(2), 1)
        self.assertEqual(inverse(4), 2)
        self.assertEqual(inverse(6), 3)

    def test_callable_without_keys_raises(self):
        def double(x):
            return x * 2
        self.assertRaises(ValueError, 
                         inverse_bijective_finite_map, double)


if __name__ == '__main__':
    unittest.main()
