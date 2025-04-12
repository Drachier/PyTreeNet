import unittest

from pytreenet.util.std_utils import permute_iterator

class Test_permute_iterator(unittest.TestCase):

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

if __name__ == '__main__':
    unittest.main()