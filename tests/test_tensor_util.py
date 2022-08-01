import unittest
import pytreenet as ptn

class TestTreeTensorNetwork(unittest.TestCase):

    def test_matricization(self):
        tensor1 = ptn.crandn((2,3,4,5))
        output_legs = (1,3)
        input_legs = (0,2)

        matrix = ptn.tensor_matricization(tensor1, output_legs, input_legs)

        self.assertEqual(matrix.shape, (3*5,2*4))


if __name__ == "__main__":
    unittest.main()