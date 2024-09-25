import unittest

from pytreenet.util.std_utils import permute_tuple

class Test_permute_tuple(unittest.TestCase):

    def test_permute_tuple(self):
        self.assertEqual(permute_tuple((1, 2, 3), [2, 1, 0]), (3, 2, 1))
        self.assertEqual(permute_tuple((1, 2, 3), [0, 1, 2]), (1, 2, 3))
        self.assertEqual(permute_tuple((1, 2, 3), [0, 2, 1]), (1, 3, 2))

    def test_permute_tuple_empty(self):
        self.assertEqual(permute_tuple((), []), ())

    def test_permute_tuple_single(self):
        self.assertEqual(permute_tuple((1,), [0]), (1,))

    def test_permute_tuple_error(self):
        self.assertRaises(AssertionError, permute_tuple, (1, 2, 3), [0, 1, 2, 3])
        self.assertRaises(AssertionError, permute_tuple, (1, 2, 3), [0, 1])

if __name__ == '__main__':
    unittest.main()