import unittest
import pytreenet as ptn
import numpy as np

class TestTensorNode(unittest.Testcase):

    def setUp(self):
        tensor1 = ptn.crandn((2,3,4))
        tensor2 = ptn.crandn((2,3,4,5))


if __name__ == "__main__":
    unittest.main()
